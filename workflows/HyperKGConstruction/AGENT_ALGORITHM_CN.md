# HyperKGConstruction Agent 算法说明

本文档解释 `workflows/HyperKGConstruction` 中每个 agent 的职责、输入输出和核心算法逻辑。

这个 workflow 的定位是：在旧 KARMA workflow 已经产出 `summary + local KG + unified KG` 之后，构建一个 textual-rich、source-grounded、role-aware 的 HyperKG。它不会重新做 PDF ingestion、阅读、摘要、实体抽取、关系抽取或 unified KG construction。

---

## 1. 总体流程

输入来自 `KARMA/output`：

```text
KARMA/output/
  summaries/
    *_summaries.json
  local_kg/
    *_kg.json
  unified_kg/
    unified_kg.json
```

当前数据特点：

- `summaries/*.json` 是每篇文章一个 `list[str]`，每个字符串视为一个 segment-level summary。
- `local_kg/*.json` 是 article-level local KG，不是真正的 segment-level KG。
- `unified_kg/unified_kg.json` 是跨文章合并后的 canonical entity / triple backbone。

整体算法：

```text
summaries + local KGs + unified KG
        |
        v
EvidencePacketAgent
        |
        v
ClaimSplitterAgent
        |
        v
HyperedgeComposerAgent
        |
        v
HyperedgeCriticAgent
        |
        v
HyperedgeMergerAgent
        |
        v
HyperKGWriterAgent
        |
        v
JSONL HyperKG + review queue + optional vector indexes
```

核心思想：

```text
summary    -> textual-rich claim source
local KG   -> article/segment structural constraint
unified KG -> canonical entity backbone
embedding  -> candidate retrieval, entity linking, merge pre-ranking
LLM        -> claim splitting, role assignment, faithfulness/merge judgment
```

---

## 2. Pipeline Controller

文件：`pipeline.py`

`HyperKGPipeline` 不是 agent，它负责把 6 个 agent 串起来。

### 2.1 初始化逻辑

Pipeline 初始化时会创建：

```text
EvidencePacketAgent
ClaimSplitterAgent
HyperedgeComposerAgent
HyperedgeCriticAgent
HyperedgeMergerAgent
HyperKGWriterAgent
```

同时也会创建共享基础设施：

```text
EmbeddingService    可选
VectorIndexAdapter  可选
```

`EmbeddingService` 不是 agent。它不生成知识，只做文本向量化。

### 2.2 运行逻辑

`run()` 的核心流程：

```python
packets = EvidencePacketAgent.process(...)
claims = ClaimSplitterAgent.process(packets)
evidence_hyperedges = HyperedgeComposerAgent.process(claims, packet_map)
reviewed_hyperedges, critic_reviews = HyperedgeCriticAgent.process(...)
accepted = [h for h in reviewed_hyperedges if h.decision == "ACCEPT"]
canonical_hyperedges, merge_reviews = HyperedgeMergerAgent.process(accepted)
run_stats = HyperKGWriterAgent.process(...)
```

最后返回：

```text
packets
claims
evidence_hyperedges
canonical_hyperedges
review_items
run_stats
```

---

## 3. EvidencePacketAgent

文件：`agents/evidence_packet.py`

### 3.1 目标

把已有的 summary、local KG、unified KG 对齐成统一输入包 `SegmentEvidencePacket`。

它不做新抽取。它只做：

- summary 标准化
- local KG 匹配
- article-level KG 到 summary-level packet 的裁剪
- local entity 到 unified KG canonical entity 的链接
- unresolved entity 记录

### 3.2 输入

```python
summaries: list[dict] | list[str]
local_kgs: list[dict] | dict[str, dict]
unified_kg: dict
metadata: dict | None
```

在当前 `KARMA/output` 中，CLI 会把每条 summary 转成：

```json
{
  "article_id": "...",
  "segment_id": "seg0001",
  "summary_id": "SUM:article:seg0001",
  "summary_text": "..."
}
```

### 3.3 输出

```python
list[SegmentEvidencePacket]
```

每个 packet 形如：

```python
SegmentEvidencePacket(
    packet_id="P:article_id:segment_id",
    article_id="...",
    segment_id="seg0001",
    summary_id="SUM:...",
    summary_text="...",
    local_entities=[...],
    local_triples=[...],
    canonical_entity_map={"local mention": "canonical entity id"},
    unresolved_entities=[...],
    metadata={...},
    warnings=[...],
)
```

### 3.4 真实 KARMA 数据适配逻辑

当前 local KG 是 article-level，而不是 segment-level。因此如果直接把整篇 KG 交给后续 LLM，会有两个问题：

- prompt 过长
- summary 和 KG 噪声不匹配

所以 `EvidencePacketAgent` 会对每条 summary 做 local KG filtering。

#### 3.4.1 Summary-level local KG filtering

算法：

```text
输入：article-level local KG + summary_text

1. 读取文章 KG 中所有 entities 和 triples。
2. 对每个 entity mention 计算 lexical match score：
   - mention 直接出现在 summary 中 -> 高分
   - 单词边界匹配 -> 高分
   - 太短 mention 忽略
3. 对每个 triple 计算 triple score：
   - head 出现在 summary 中
   - tail 出现在 summary 中
   - head/tail 都出现则更高
   - relation 文本也出现则加分
   - confidence/relevance/clarity 给一个小加权
4. 按分数排序，取 top max_local_triples_per_packet。
5. 把选中 triple 的 head/tail 加回 local_entities。
6. 按实体分数排序，取 top max_local_entities_per_packet。
7. 如果完全没有匹配，则 fallback 到 capped article KG。
```

默认上限：

```text
max_local_entities_per_packet = 80
max_local_triples_per_packet = 60
```

CLI 可以调：

```bash
--max-local-entities 80
--max-local-triples 60
```

metadata 会记录：

```text
hyperkg_filter_mode
hyperkg_original_local_entity_count
hyperkg_original_local_triple_count
hyperkg_packet_entity_count
hyperkg_packet_triple_count
```

### 3.5 Entity linking 逻辑

对每个 local entity mention，链接到 unified KG canonical entity。

优先级：

```text
1. entity_id exact match
2. normalized_id exact match
3. ontology_id exact match
4. canonical_name exact match
5. alias exact match
6. lowercase normalized exact match
7. fuzzy match
8. dense embedding retrieval, if enabled
```

如果启用 embedding：

```text
1. 为 unified KG canonical entities 建立 canonical_entity vector index。
2. 对 unresolved local entity mention 编码。
3. 在 canonical_entity namespace 中 top-k search。
4. top-1 score >= entity_linking_threshold 则自动链接。
5. 否则把候选写入 unresolved_entities。
```

### 3.6 Review 触发

以下情况会进入 review queue：

- local KG 缺失
- summary 为空
- entity 无法链接
- dense candidate 存在但低于 auto-link threshold
- local KG filtering 没有匹配，只能 fallback 到 capped article KG

---

## 4. ClaimSplitterAgent

文件：`agents/claim_splitter.py`

### 4.1 目标

把一个 segment summary 拆成多个 atomic claims。

一个 summary 往往包含多个知识点，例如：

```text
Risperidone reduced irritability in children with ASD,
but caused weight gain in some participants.
```

这至少应拆成两条 claim：

```text
1. Risperidone reduced irritability in children with ASD.
2. Risperidone caused weight gain in some participants.
```

### 4.2 输入

```python
SegmentEvidencePacket
```

Prompt template 会拿到这些 placeholders：

```text
{summary_text}
{local_entities}
{local_triples}
{canonical_entity_map}
{metadata}
```

### 4.3 输出

```python
list[AtomicClaim]
```

每条 claim：

```python
AtomicClaim(
    claim_id="C:article:segment:01",
    packet_id="P:article:segment",
    claim_text="...",
    claim_type="INTERVENTION_OUTCOME",
    candidate_entities=[...],
    candidate_triples=[...],
    source_summary_id="...",
    source_segment_id="...",
    metadata={...},
)
```

### 4.4 算法逻辑

```text
1. 如果 summary_text 为空，直接返回空列表，并进入 review queue。
2. 从 prompts.toml 读取 system_prompt 和 prompt_template。
3. 用 packet 信息渲染 prompt。
4. 调用 OpenRouter LLM。
5. 解析 LLM JSON。
6. 支持这些 JSON 形状：
   - list
   - {"claims": [...]}
   - {"atomic_claims": [...]}
   - {"items": [...]}
7. 每条 claim 标准化为 AtomicClaim。
8. 最多保留 max_claims_per_summary 条。
9. 如果 JSON 解析失败或没有 claim，加入 review queue。
```

### 4.5 Prompt 管理

代码中不写 prompt 内容。

对应 TOML section：

```toml
[hyperkg_claim_splitter]
system_prompt = ""
prompt_template = ""
```

如果 `prompt_template` 为空，agent 会抛出 `PromptConfigurationError`。

---

## 5. HyperedgeComposerAgent

文件：`agents/hyperedge_composer.py`

### 5.1 目标

把 `AtomicClaim` 组合成正式的 source-level `EvidenceHyperedge`。

它负责把一句文本 claim 转成 n-ary hyperedge：

```text
claim text
claim type
canonical entities
entity roles
qualifiers
triple projections
source provenance
initial scores
```

### 5.2 输入

```python
claim: AtomicClaim
packet: SegmentEvidencePacket
```

Prompt template placeholders：

```text
{claim_json}
{packet_json}
{summary_text}
{local_entities}
{local_triples}
{canonical_entity_map}
```

### 5.3 输出

```python
EvidenceHyperedge
```

核心字段：

```python
EvidenceHyperedge(
    evidence_hyperedge_id="EH:article:segment:01",
    claim_text="...",
    claim_type="INTERVENTION_OUTCOME",
    entities=[
        HyperedgeEntity(entity_id="...", mention="...", role="intervention"),
        HyperedgeEntity(entity_id="...", mention="...", role="outcome"),
    ],
    qualifiers={...},
    triple_projections=[...],
    source={
        "article_id": "...",
        "segment_id": "...",
        "summary_id": "...",
        "section": "..."
    },
    scores={...},
    decision="CANDIDATE"
)
```

### 5.4 算法逻辑

```text
1. 从 TOML 读取 composer prompt。
2. 把 claim、packet、summary、local KG、canonical map 渲染到 prompt。
3. 调用 OpenRouter LLM。
4. 解析 JSON object。
5. 将 entities 转成 HyperedgeEntity。
6. 将 triple_projections 转成 TripleProjection。
7. 自动补 source.article_id / segment_id / summary_id。
8. 自动补初始 scores：
   - entity_linking
   - local_kg_agreement
   - nary_completeness
9. 构造 embedding_text，但不在 composer 内调用 embedding model。
10. 做 deterministic validation。
```

### 5.5 Validation 规则

Composer 后会做硬校验：

```text
1. hyperedge 至少有 min_entities_per_hyperedge 个实体，默认 2。
2. 每个 entity 必须有 entity_id、mention、role。
3. claim_type 不能为空，否则设为 OTHER。
4. source.article_id / segment_id / summary_id 不能为空。
5. triple projection 的 head/tail 必须出现在 hyperedge.entities 中。
6. 如果 entity_id 不在 packet.canonical_entity_map 中，标记 REVIEW。
```

不满足时：

```text
decision = REVIEW
warnings 添加具体原因
review_queue 添加 HYPEREDGE_VALIDATION item
```

### 5.6 Fallback

如果 LLM 输出不是 JSON object：

```text
1. 用 claim_text + packet.canonical_entity_map 构造一个 fallback hyperedge。
2. 所有 entity role 设为 unknown。
3. decision = REVIEW。
4. 加入 review queue。
```

---

## 6. HyperedgeCriticAgent

文件：`agents/hyperedge_critic.py`

### 6.1 目标

检查 `EvidenceHyperedge` 是否可靠，决定它是否进入 canonical merge。

输出 decision：

```text
ACCEPT
REVIEW
REJECT
```

### 6.2 输入

```python
hyperedge: EvidenceHyperedge
packet: SegmentEvidencePacket
```

Prompt template placeholders：

```text
{hyperedge_json}
{summary_text}
{local_entities}
{local_triples}
{canonical_entity_map}
```

### 6.3 评分维度

Critic 检查 5 个维度：

```text
faithfulness
  claim 是否被 summary 支持

entity_linking
  entities 是否正确链接到 unified KG

local_kg_agreement
  hyperedge 的实体/投影是否和 local KG 一致

scope_completeness
  是否保留 population、severity、dose、comparator、timepoint 等 scope 信息

relation_correctness
  是否关系方向正确，是否处理 negation / no significant effect
```

### 6.4 Integration score

聚合公式：

```python
integration_score = (
    0.30 * faithfulness
    + 0.20 * entity_linking
    + 0.20 * local_kg_agreement
    + 0.15 * scope_completeness
    + 0.15 * relation_correctness
)
```

### 6.5 Decision threshold

默认：

```text
ACCEPT if integration_score >= 0.75
REVIEW if 0.55 <= integration_score < 0.75
REJECT if integration_score < 0.55
```

### 6.6 算法逻辑

```text
1. 根据 hyperedge.source 找到对应 packet。
2. 从 TOML 读取 critic prompt。
3. 把 hyperedge、summary、local KG、canonical map 渲染到 prompt。
4. 调用 OpenRouter LLM。
5. 解析 JSON。
6. 提取或补齐五个 score。
7. 计算 integration_score。
8. 如果 LLM 给出合法 decision，则使用 LLM decision。
9. 否则根据 threshold 自动决定 ACCEPT / REVIEW / REJECT。
10. 如果 decision=REVIEW 或存在 warnings/violations，写入 review queue。
```

---

## 7. HyperedgeMergerAgent

文件：`agents/hyperedge_merger.py`

### 7.1 目标

把多个 accepted evidence hyperedges 合并成 global-level `CanonicalHyperedge`。

Evidence hyperedge 是 segment-level：

```text
EH:article:segment:claim
```

Canonical hyperedge 是跨 segment / article 聚合后的 claim cluster：

```text
CH:claim_type:hash(...)
```

### 7.2 输入

```python
evidence_hyperedges: list[EvidenceHyperedge]
existing_canonical_hyperedges: list[CanonicalHyperedge] | None
```

只处理：

```text
decision == "ACCEPT"
```

### 7.3 输出

```python
tuple[list[CanonicalHyperedge], list[review_item]]
```

### 7.4 Blocking key

先用 deterministic blocking，避免所有 hyperedges 两两比较。

Evidence hyperedge blocking key：

```python
(
    claim_type,
    sorted(core_entity_ids),
    sorted(core_projection_relations)
)
```

core roles 包括：

```text
intervention
risk_factor
mechanism
outcome
adverse_outcome
condition
population
assessment_tool
```

如果一个 evidence hyperedge 和某个 canonical hyperedge blocking key 完全一致，则直接 merge。

### 7.5 Dense candidate retrieval

如果 blocking key 没有命中，并且 embedding 已启用：

```text
1. 为 canonical hyperedges 建 canonical_hyperedge vector index。
2. 把 evidence hyperedge 的 embedding_text 编码。
3. 在 canonical_hyperedge namespace 检索 top-k。
4. 先按 claim_type filter。
5. 对候选做 compatibility check。
```

### 7.6 Merge threshold

默认：

```text
auto_merge_similarity = 0.90
llm_merge_similarity = 0.78
max_merge_candidates = 10
```

逻辑：

```text
if similarity >= auto_merge_similarity and deterministic filters pass:
    merge
elif similarity >= llm_merge_similarity:
    call LLM merger prompt
else:
    create new canonical hyperedge
```

### 7.7 Compatibility check

目前 deterministic filter 包括：

```text
1. claim_type 必须一致。
2. core_entity_ids 至少要有 overlap。
3. ADVERSE_EFFECT 不和 INTERVENTION_OUTCOME 直接合并。
```

避免错误合并，例如：

```text
Risperidone reduces irritability.
Risperidone causes weight gain.
```

这两条实体可能相近，但 claim type / outcome 不同，不应该合并。

### 7.8 LLM merge decision

当 dense similarity 进入不确定区间时，调用 merger prompt。

Prompt placeholders：

```text
{evidence_hyperedge_json}
{candidate_canonical_hyperedges_json}
{merge_criteria}
```

LLM 可返回：

```text
MERGE
NEW
RELATED_TO
DIFFERENT_SCOPE
CONTRADICTS
```

处理方式：

```text
MERGE           -> 合并到 candidate canonical hyperedge
NEW             -> 新建 canonical hyperedge
RELATED_TO      -> 记录 related_hyperedges
DIFFERENT_SCOPE -> 记录 related_hyperedges
CONTRADICTS     -> 记录 conflicts
```

### 7.9 CanonicalHyperedge 生成

新建 canonical hyperedge 时：

```text
canonical_hyperedge_id = CH:{claim_type}:{hash(core_entities + relations + claim_text)}
canonical_claim = evidence_hyperedge.claim_text
member_evidence_hyperedges = [EH id]
support_count = 1
core_entity_ids = core role entities
qualifier_summary = qualifiers aggregation
scope_summary = sources / segments / projection relations / claim samples
confidence_summary = integration_score summary
```

合并时更新：

```text
member_evidence_hyperedges
support_count
core_entity_ids
qualifier_summary
scope_summary
confidence_summary
embedding_text
```

---

## 8. HyperKGWriterAgent

文件：`agents/hyperkg_writer.py`

### 8.1 目标

把最终 HyperKG 写成本地 artifact。

第一版使用 JSONL + local vector index，不依赖 Neo4j。

### 8.2 输入

```python
evidence_hyperedges
canonical_hyperedges
packets
review_items
```

### 8.3 输出目录

```text
output_dir/
  evidence_hyperedges.jsonl
  canonical_hyperedges.jsonl
  incidence_edges.jsonl
  triple_projections.jsonl
  summary_links.jsonl
  review_queue.jsonl
  run_stats.json

  vector_indexes/
    canonical_entity.index
    canonical_entity.metadata.jsonl
    canonical_entity.vectors.npy
    summary.index
    summary.metadata.jsonl
    summary.vectors.npy
    evidence_hyperedge.index
    evidence_hyperedge.metadata.jsonl
    evidence_hyperedge.vectors.npy
    canonical_hyperedge.index
    canonical_hyperedge.metadata.jsonl
    canonical_hyperedge.vectors.npy
    manifest.json
```

### 8.4 写出内容

#### evidence_hyperedges.jsonl

每行一个 `EvidenceHyperedge`。

保留：

```text
claim_text
claim_type
entities
qualifiers
triple_projections
source
scores
decision
warnings
embedding_text
vector_id
```

#### canonical_hyperedges.jsonl

每行一个 `CanonicalHyperedge`。

保留：

```text
canonical_claim
claim_type
member_evidence_hyperedges
support_count
core_entity_ids
qualifier_summary
scope_summary
confidence_summary
related_hyperedges
conflicts
```

#### incidence_edges.jsonl

entity-hyperedge 二部图边：

```json
{
  "entity_id": "...",
  "evidence_hyperedge_id": "...",
  "role": "intervention",
  "mention": "...",
  "linking_confidence": 0.96
}
```

#### triple_projections.jsonl

兼容旧 KG / GraphRAG 的 binary projection：

```json
{
  "triple_projection_id": "TP:...",
  "evidence_hyperedge_id": "...",
  "head_entity_id": "...",
  "relation": "reduces",
  "tail_entity_id": "...",
  "support": "...",
  "confidence": 0.88
}
```

#### summary_links.jsonl

hyperedge 到 summary provenance：

```json
{
  "evidence_hyperedge_id": "...",
  "article_id": "...",
  "segment_id": "...",
  "summary_id": "..."
}
```

#### review_queue.jsonl

所有需要人工审核的 item。

#### run_stats.json

统计：

```text
packet_count
evidence_hyperedge_count
accepted_evidence_hyperedge_count
review_evidence_hyperedge_count
rejected_evidence_hyperedge_count
canonical_hyperedge_count
incidence_edge_count
triple_projection_count
review_item_count
output_dir
timing_sec
```

### 8.5 Vector index 写出

如果 embedding enabled：

```text
summary
evidence_hyperedge
canonical_hyperedge
canonical_entity
```

都会被写入 vector index。

JSONL 中不保存大向量，只保存 vector id / embedding text / metadata。真正向量在 `.vectors.npy` 中。

---

## 9. EmbeddingService

文件：`embedding_service.py`

### 9.1 目标

提供 frozen dense text encoder。

它不是 agent，不生成知识，不做最终判断。

### 9.2 用途

当前主要用于：

```text
1. EvidencePacketAgent
   local entity -> canonical entity candidate retrieval

2. HyperedgeMergerAgent
   evidence hyperedge -> canonical hyperedge candidate retrieval

3. HyperKGWriterAgent
   写 summary / evidence / canonical vector indexes
```

### 9.3 配置

```python
EmbeddingConfig(
    enabled=True,
    model_name="",
    model_path=None,
    device="cuda",
    batch_size=64,
    normalize=True,
    max_length=512,
    use_fp16=True,
    trust_remote_code=False,
)
```

如果没有配置 `embedding_model_name` 或 `embedding_model_path`，workflow 会自动 fallback 到 deterministic mode。

---

## 10. VectorIndexAdapter

文件：`vector_index.py`

### 10.1 目标

MVP 版本的本地向量索引。

功能：

```text
add(namespace, ids, texts, vectors, metadata)
search(namespace, query_text/query_vector, top_k, filters)
save(output_dir)
```

### 10.2 Search 算法

```text
1. 对 query vector 和 corpus vectors 做 L2 normalization。
2. 用 matrix multiplication 计算 cosine similarity。
3. argpartition 找 top-k。
4. 按 score 排序。
5. 返回 id/text/metadata/score。
```

### 10.3 Persistence

每个 namespace 写：

```text
{namespace}.index
{namespace}.metadata.jsonl
{namespace}.vectors.npy
```

再写总 manifest：

```text
manifest.json
```

---

## 11. Prompt 和 OpenRouter

### 11.1 Prompt 管理

所有 LLM prompt 都在：

```text
workflows/HyperKGConstruction/prompts.toml
```

当前留白：

```toml
[hyperkg_claim_splitter]
system_prompt = ""
prompt_template = ""

[hyperkg_hyperedge_composer]
system_prompt = ""
prompt_template = ""

[hyperkg_hyperedge_critic]
system_prompt = ""
prompt_template = ""

[hyperkg_hyperedge_merger]
system_prompt = ""
prompt_template = ""
```

代码不会 hardcode prompt。

如果 `prompt_template` 为空，LLM-backed agent 会报：

```text
PromptConfigurationError
```

### 11.2 OpenRouter API

文件：`llm.py`

默认从 `.env` 加载：

```text
OPENROUTER_API_KEY=...
```

默认 base URL：

```text
https://openrouter.ai/api/v1
```

可选覆盖：

```text
OPENROUTER_BASE_URL=...
```

CLI 覆盖：

```bash
--api-key ...
--api-base-url ...
--env-file path/to/.env
```

---

## 12. Review Queue 逻辑

review queue 是人工审核入口。

常见类型：

```text
EMPTY_SUMMARY
LLM_JSON_PARSE_FAILED
MISSING_PACKET
HYPEREDGE_VALIDATION
CRITIC_REVIEW
UNCERTAIN_MERGE
UNRESOLVED_ENTITY
PACKET_WARNING
VECTOR_INDEX_FAILED
```

每个 review item 包含：

```json
{
  "review_item_id": "REV:000001",
  "type": "...",
  "object_id": "...",
  "article_id": "...",
  "segment_id": "...",
  "summary_text": "...",
  "object_json": {},
  "reason": "...",
  "suggested_action": "..."
}
```

---

## 13. 当前实现的关键取舍

### 13.1 为什么要裁剪 local KG

当前 `KARMA/output/local_kg` 是 article-level graph。每篇文章可能有几十到上百个实体和三元组，个别文章更大。

如果每条 summary 都带整篇 KG：

```text
prompt 太长
LLM 注意力被噪声干扰
claim/entity role 更容易错配
成本更高
```

所以当前实现用 lexical matching 把 article KG 转成 summary-relevant packet。

### 13.2 为什么 embedding 不是 agent

Embedding 只做 candidate retrieval：

```text
entity candidate retrieval
canonical hyperedge candidate retrieval
vector index construction
```

最终判断仍由 deterministic rule 或 LLM agent 完成。

### 13.3 为什么只 merge ACCEPT

`REVIEW` 和 `REJECT` 的 evidence hyperedges 可能存在：

```text
faithfulness 问题
entity linking 问题
关系方向问题
scope 缺失
```

直接合并会污染 canonical HyperKG。因此第一版只把 `ACCEPT` 输入 merger。

---

## 14. 运行方式

在填好 `prompts.toml` 后运行：

```bash
python -m workflows.HyperKGConstruction run-karma-output \
  --karma-output-dir KARMA/output \
  --output-dir output/hyperkg
```

如果想限制 prompt 中 local KG 的大小：

```bash
python -m workflows.HyperKGConstruction run-karma-output \
  --karma-output-dir KARMA/output \
  --output-dir output/hyperkg \
  --max-local-entities 50 \
  --max-local-triples 30
```

如果启用 embedding：

```bash
python -m workflows.HyperKGConstruction run-karma-output \
  --karma-output-dir KARMA/output \
  --output-dir output/hyperkg \
  --embedding-model BAAI/bge-large-en-v1.5 \
  --embedding-device cuda
```

