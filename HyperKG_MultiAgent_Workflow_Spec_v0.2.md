# HyperKG Multi-Agent Workflow Implementation Spec

版本：v0.2  
目标读者：Codex / implementation engineer  
适用范围：基于已经生成好的 segment-level summary、local KG 和 unified KG，构建 textual-rich HyperKG。  
重要约束：本 workflow 不重新做 PDF ingestion、segment reading、summarization、entity extraction、relationship extraction 或 unified KG construction。

---

## 1. 目标

本项目需要新增一个独立的 HyperKG construction layer。它接收已经采集完成的数据：

```text
segment summaries
local KGs
unified KG
article / segment metadata
```

并输出：

```text
evidence hyperedges
canonical hyperedges
entity-hyperedge incidence graph
triple projections
provenance links
review queue
vector indexes for retrieval / merging
optional graph-store export
```

核心设计原则：

```text
summary      -> textual-rich claim source
local KG     -> local structural constraint
unified KG   -> canonical entity backbone
embedding    -> candidate generation, semantic merge, retrieval-ready index
HyperKG      -> role-aware, scope-aware, source-grounded claim graph
```

---

## 2. 是否需要 dense embedding model

结论：**需要，但不要把它设计成新的 agent。**

Dense embedding 不负责生成知识，也不负责做最终判断；它应该作为一个基础设施层 `EmbeddingService` / `VectorIndexAdapter` 被 6 个 agents 调用。这样可以保持 workflow 简单，同时利用 GPU 提升 entity linking、hyperedge merging 和后续 retrieval readiness。

### 2.1 为什么需要

HyperKG 的核心难点不是单个 segment 内生成 hyperedge，而是跨 segment / article 的语义对齐：

```text
Functional communication training reduced aggression in children with ID.
FCT decreased aggressive behavior among children with intellectual disability.
```

这两句话在 surface form 上不同，但应该进入同一个 canonical hyperedge。只靠 deterministic rule 会漏掉很多同义表达；只靠 LLM 两两比较会非常贵，也不稳定。因此需要 dense embeddings 做 candidate retrieval 和 semantic pre-ranking。

Dense embeddings 主要用于三处：

```text
1. EvidencePacketAgent:
   local entity mention -> unified KG canonical entity candidate retrieval

2. HyperedgeMergerAgent:
   evidence hyperedge -> candidate canonical hyperedge retrieval / clustering

3. HyperKGWriterAgent:
   entity / evidence hyperedge / canonical hyperedge / summary vector indexes
```

### 2.2 不需要做什么

这不是 HypKG-style hypergraph transformer 训练，也不是 GNN 或 KG embedding training。第一版不要做：

```text
hypergraph neural network
hypergraph transformer training
ComplEx / TransE / CompGCN training
DeepWalk training
supervised contrastive training
```

第一版只需要做 **frozen dense encoder + vector index**：

```text
text -> embedding vector -> top-k candidate retrieval -> LLM or deterministic decision
```

### 2.3 推荐策略

优先实现一个通用 `EmbeddingService`，模型名称从配置读取，不要硬编码。建议支持：

```text
embedding_model_name: str
embedding_model_path: str | None
embedding_device: "cuda" | "cpu"
embedding_batch_size: int
embedding_normalize: bool
embedding_dimension: int | None
```

MVP 可以只使用一个 dense text embedding model。后续如果 entity linking 错误较多，再加入第二个 biomedical entity-linking encoder。

推荐默认策略：

```text
MVP:
  one dense text embedding model for entities, summaries, evidence hyperedges, canonical hyperedges

Optional later:
  separate biomedical entity embedding model for entity linking
  same general dense model for claim / hyperedge retrieval
```

### 2.4 和 LLM 的关系

Embedding 负责 **召回候选**，LLM 负责 **结构化判断**。

```text
embedding:
  fast, cheap, parallelizable, high recall candidate generation

LLM:
  role assignment, qualifier extraction, faithfulness check, uncertain merge decision
```

不要让 LLM 对所有 hyperedges 做全量两两比较。正确做法是：

```text
dense embedding top-k candidate retrieval
  -> deterministic filters
  -> LLM only for uncertain candidate pairs
```

---

## 3. 与旧 KG workflow 的边界

不要在 HyperKG workflow 中重复实现旧 KG pipeline。以下模块已经属于旧 workflow，不应该在新 workflow 中重新执行：

```text
IngestionAgent
ReaderAgent
SummarizerAgent
EntityExtractionAgent
RelationshipExtractionAgent
SchemaAlignmentAgent
ConflictResolutionAgent
EvaluatorAgent
```

HyperKG workflow 只处理旧 workflow 的产物：

```text
summary + local KG + unified KG -> HyperKG
```

新 workflow 的定位是：

```text
post-KG hyperedge construction and integration layer
```

---

## 4. 简化版 Multi-Agent Workflow

不要超过 7 个 agents。本实现建议使用 6 个 agents：

```text
Input:
  segment summaries
  local KGs
  unified KG

Infrastructure:
  EmbeddingService
  VectorIndexAdapter
  ReviewQueue

Agents:
  1. EvidencePacketAgent
  2. ClaimSplitterAgent
  3. HyperedgeComposerAgent
  4. HyperedgeCriticAgent
  5. HyperedgeMergerAgent
  6. HyperKGWriterAgent

Output:
  textual-rich HyperKG
```

整体流程：

```text
summary + local KG + unified KG
        ↓
[1] EvidencePacketAgent
        ↓
[2] ClaimSplitterAgent
        ↓
[3] HyperedgeComposerAgent
        ↓
[4] HyperedgeCriticAgent
        ↓
[5] HyperedgeMergerAgent
        ↓
[6] HyperKGWriterAgent
        ↓
Textual-rich HyperKG + vector indexes
```

可以另外实现一个 `HyperKGPipeline` 或 `HyperKGOrchestrator` 类用于调度，但它不算 agent，不需要 prompt。

---

## 5. 推荐代码目录结构

建议把 HyperKG 作为独立子包，避免和旧 KG agents 混在一起：

```text
karma/
  agents/
    hyperkg/
      __init__.py
      pipeline.py
      config.py

      evidence_packet/
        __init__.py
        agent.py

      claim_splitter/
        __init__.py
        agent.py

      hyperedge_composer/
        __init__.py
        agent.py

      hyperedge_critic/
        __init__.py
        agent.py

      hyperedge_merger/
        __init__.py
        agent.py

      hyperkg_writer/
        __init__.py
        agent.py

  core/
    hyperkg_data_structures.py
    embedding_service.py
    vector_index.py
    review_queue.py
```

也可以把 `hyperkg_data_structures.py` 合并进现有 `data_structures.py`，但为了减少对旧 KG pipeline 的影响，建议单独建文件。

`embedding_service.py` 和 `vector_index.py` 是基础设施，不是 agent。

---

## 6. Prompt loading 约束

不要在 Python 代码中硬编码 prompt。所有需要 LLM 的 agent 都必须从 `prompts.toml` 加载配置。

推荐 TOML section 名称：

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

`EvidencePacketAgent` 和 `HyperKGWriterAgent` 默认可以是 deterministic agent，不需要 prompt。如果实现时需要 LLM，也必须加对应 TOML section：

```toml
[hyperkg_evidence_packet]
system_prompt = ""
prompt_template = ""

[hyperkg_writer]
system_prompt = ""
prompt_template = ""
```

代码中加载方式应保持和旧 agents 一致：

```python
config = get_agent_config("hyperkg_claim_splitter")
system_prompt = config.get("system_prompt", "")
self.prompt_template = config.get("prompt_template", "")
```

建议：如果 `prompt_template` 缺失或为空，LLM agent 应该 raise 明确错误，而不是在代码里 fallback 到长 prompt。这样可以保证 prompt 完全由 TOML 管理。

---

## 7. EmbeddingService and VectorIndexAdapter

### 7.1 作用

`EmbeddingService` 负责把文本批量编码成 dense vectors。`VectorIndexAdapter` 负责添加向量、保存向量、top-k 查询。

它们不是 agents，不需要 prompt。

### 7.2 推荐配置

```python
@dataclass
class EmbeddingConfig:
    enabled: bool = True
    model_name: str = ""
    model_path: str | None = None
    device: str = "cuda"
    batch_size: int = 64
    normalize: bool = True
    max_length: int = 512
    use_fp16: bool = True
    trust_remote_code: bool = False
```

建议 `model_name` 由外部 config 或 environment variable 提供。不要把具体模型写死在 agent 文件里。

### 7.3 推荐接口

```python
class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return shape = (len(texts), dim)."""
        ...

    def encode_one(self, text: str) -> np.ndarray:
        ...
```

```python
class VectorIndexAdapter:
    def add(
        self,
        namespace: str,
        ids: list[str],
        texts: list[str],
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        ...

    def search(
        self,
        namespace: str,
        query_text: str | None = None,
        query_vector: np.ndarray | None = None,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        ...

    def save(self, output_dir: str) -> None:
        ...
```

### 7.4 推荐 namespaces

```text
canonical_entity
summary
evidence_hyperedge
canonical_hyperedge
```

### 7.5 Embedding 文本模板

不要直接把原始 JSON dump 给 embedding model。需要构造稳定、简洁的 embedding text。

```python
def entity_embedding_text(entity: dict) -> str:
    return f"{name}. Type: {entity_type}. Aliases: {aliases}. Description: {description}."
```

```python
def evidence_hyperedge_embedding_text(h: EvidenceHyperedge) -> str:
    return "\n".join([
        f"Claim: {h.claim_text}",
        f"Type: {h.claim_type}",
        "Entities: " + "; ".join(
            f"{e.mention} ({e.role}, {e.entity_type})" for e in h.entities
        ),
        "Qualifiers: " + compact_json(h.qualifiers),
    ])
```

```python
def canonical_hyperedge_embedding_text(ch: CanonicalHyperedge) -> str:
    return "\n".join([
        f"Canonical claim: {ch.canonical_claim}",
        f"Type: {ch.claim_type}",
        "Core entities: " + ", ".join(ch.core_entity_ids),
        "Scope: " + compact_json(ch.scope_summary),
        "Qualifiers: " + compact_json(ch.qualifier_summary),
    ])
```

### 7.6 向量存储建议

JSONL 里不要保存大向量。推荐：

```text
JSONL objects store vector_id / embedding_text / embedding_model
actual vectors stored in vector index or .npy / .faiss files
```

MVP 可以用 in-memory cosine search + `.npy` 保存。后续再切换 FAISS、Qdrant、Milvus 或 Neo4j vector index。

### 7.7 GPU 使用建议

实现时支持 batching：

```text
entity embeddings: batch encode canonical entity texts
summary embeddings: batch encode summary_text
hyperedge embeddings: batch encode claim_text + roles + qualifiers
```

如果 GPU memory 不足，降低 batch_size。不要在每个 agent 内重复加载模型；由 `HyperKGPipeline` 初始化一个共享 `EmbeddingService`，传给需要的 agents。

---

## 8. 核心数据结构

### 8.1 SegmentEvidencePacket

`SegmentEvidencePacket` 是每个 segment 的标准输入包。它把 summary、local KG、unified KG alignment 和 metadata 放在一起。

```python
@dataclass
class SegmentEvidencePacket:
    packet_id: str
    article_id: str
    segment_id: str
    summary_id: str
    summary_text: str

    local_entities: list[dict]
    local_triples: list[dict]

    canonical_entity_map: dict[str, str]
    unresolved_entities: list[dict]

    metadata: dict
    warnings: list[str] = field(default_factory=list)
```

示例：

```json
{
  "packet_id": "P:article123:seg045",
  "article_id": "article123",
  "segment_id": "seg045",
  "summary_id": "SUM:article123:seg045",
  "summary_text": "Risperidone reduced irritability in children with ASD and severe ID...",
  "local_entities": [],
  "local_triples": [],
  "canonical_entity_map": {
    "Risperidone": "E:MEDICATION:risperidone",
    "ASD": "E:DISORDER:autism_spectrum_disorder"
  },
  "metadata": {
    "section": "results",
    "year": 2021,
    "doi": "..."
  }
}
```

### 8.2 AtomicClaim

`AtomicClaim` 是从 summary 拆出来的最小 claim。一个 summary 可以对应多个 atomic claims。

```python
@dataclass
class AtomicClaim:
    claim_id: str
    packet_id: str
    claim_text: str
    claim_type: str
    candidate_entities: list[str]
    candidate_triples: list[dict]
    source_summary_id: str
    source_segment_id: str
    metadata: dict
```

推荐 claim type：

```text
GENE_PHENOTYPE
NEUROBIOLOGICAL_MECHANISM
RISK_ASSOCIATION
CAUSAL_MECHANISM
INTERVENTION_OUTCOME
ADVERSE_EFFECT
ASSESSMENT_DIAGNOSTIC
CARE_ACCESS_SYSTEM
SOCIAL_CONTEXT_OUTCOME
EDUCATIONAL_INTERVENTION
OTHER
```

### 8.3 HyperedgeEntity

```python
@dataclass
class HyperedgeEntity:
    entity_id: str
    mention: str
    role: str
    entity_type: str | None = None
    linking_confidence: float = 1.0
```

推荐 role set：

```text
population
condition
intervention
mechanism
risk_factor
outcome
adverse_outcome
assessment_tool
severity
developmental_stage
comparator
dose
timepoint
statistical_evidence
care_setting
social_context
unknown
```

### 8.4 TripleProjection

```python
@dataclass
class TripleProjection:
    head_entity_id: str
    relation: str
    tail_entity_id: str
    support: str
    confidence: float = 0.5
```

注意：triple projection 只是为了和原 KG / GraphRAG 兼容。不要把 hyperedge 中所有 entity pair 都投影成 triple。只有 summary 或 local KG 明确支持的关系才应该投影。

### 8.5 EvidenceHyperedge

`EvidenceHyperedge` 是 segment-level 的 source-grounded hyperedge。

```python
@dataclass
class EvidenceHyperedge:
    evidence_hyperedge_id: str
    claim_text: str
    claim_type: str

    entities: list[HyperedgeEntity]
    qualifiers: dict
    triple_projections: list[TripleProjection]

    source: dict
    scores: dict
    decision: str = "CANDIDATE"
    warnings: list[str] = field(default_factory=list)

    embedding_text: str | None = None
    vector_id: str | None = None
```

### 8.6 CanonicalHyperedge

`CanonicalHyperedge` 是跨 segment / article 聚合后的 global-level hyperedge。

```python
@dataclass
class CanonicalHyperedge:
    canonical_hyperedge_id: str
    canonical_claim: str
    claim_type: str

    member_evidence_hyperedges: list[str]
    support_count: int

    core_entity_ids: list[str]
    qualifier_summary: dict
    scope_summary: dict
    confidence_summary: dict

    related_hyperedges: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)

    embedding_text: str | None = None
    vector_id: str | None = None
```

---

## 9. Agent 1: EvidencePacketAgent

### 9.1 作用

把已有的 summary、local KG、unified KG 对齐成标准输入包。这个 agent 不做新的 extraction。

### 9.2 输入

```python
summaries: list[dict]
local_kgs: list[dict] | dict[str, dict]
unified_kg: dict
metadata: dict | None
embedding_service: EmbeddingService | None
vector_index: VectorIndexAdapter | None
```

summary 最少需要包含：

```text
article_id
segment_id
summary_id
summary_text
section optional
```

local KG 最少需要包含：

```text
entities
triples
article_id / segment_id
```

unified KG 最少需要包含：

```text
canonical_entities
aliases
ontology_ids optional
canonical_triples optional
```

### 9.3 处理逻辑

1. 按 `article_id + segment_id` 找到 summary 对应的 local KG。
2. 读取 local entities 和 local triples。
3. 用 unified KG 做 canonical entity map，优先顺序：
   - normalized_id exact match
   - ontology_id exact match
   - canonical name exact match
   - alias exact match
   - lowercase normalized match
   - dense embedding top-k candidate retrieval
   - fuzzy match as last fallback
4. 记录 unresolved entities。
5. 生成 `SegmentEvidencePacket`。
6. 如果 local KG 缺失，不要报错终止，而是生成 packet 并添加 warning。

### 9.4 Dense embedding usage

如果 `embedding_service.enabled == True`：

```text
1. 为 unified KG canonical entities 建立 canonical_entity vector index。
2. 对 unresolved local entity mention 构造 embedding text。
3. 在 canonical_entity namespace 中 top-k search。
4. 如果 top-1 similarity >= entity_linking_threshold，则自动链接。
5. 如果 top-k 中有候选但分数不足，则放入 unresolved_entities 并记录 candidates。
```

推荐输出 unresolved candidate 结构：

```json
{
  "mention": "FCT",
  "local_entity_type": "THERAPEUTIC_APPROACH",
  "candidates": [
    {
      "canonical_entity_id": "E:THERAPEUTIC_APPROACH:functional_communication_training",
      "score": 0.87,
      "canonical_name": "functional communication training"
    }
  ],
  "reason": "dense candidate below auto-link threshold"
}
```

### 9.5 输出

```python
list[SegmentEvidencePacket]
```

### 9.6 推荐方法签名

```python
class EvidencePacketAgent:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_index: VectorIndexAdapter | None = None,
        entity_linking_threshold: float = 0.82,
        entity_linking_top_k: int = 10,
    ):
        ...

    def process(
        self,
        summaries: list[dict],
        local_kgs: list[dict] | dict[str, dict],
        unified_kg: dict,
        metadata: dict | None = None,
    ) -> list[SegmentEvidencePacket]:
        ...
```

### 9.7 错误处理

```text
summary_text empty      -> skip or REVIEW
local KG missing        -> keep packet with warning
entity cannot link      -> put into unresolved_entities
duplicate packet_id     -> append stable suffix or raise
embedding model missing -> fallback to deterministic linking and warn
```

---

## 10. Agent 2: ClaimSplitterAgent

### 10.1 作用

把一个 segment summary 拆成多个 atomic claims。

不要把整个 summary 直接当作一个 hyperedge。一个 summary 里经常同时包含 intervention effect、adverse effect、population condition、mechanism、assessment 等多个 claim。

### 10.2 输入

```python
packet: SegmentEvidencePacket
```

### 10.3 处理逻辑

1. 使用 LLM 判断 summary 中包含几个独立 claim。
2. 每个 claim 必须是 atomic。
3. 每个 claim 至少包含 2 个 candidate entities。
4. 每个 claim 给出 `claim_type`。
5. candidate entities 优先来自 packet.local_entities 和 canonical_entity_map。
6. candidate triples 优先来自 packet.local_triples。
7. 如果 summary 过短或无有效知识，返回空列表。

### 10.4 输出

```python
list[AtomicClaim]
```

### 10.5 推荐方法签名

```python
class ClaimSplitterAgent(BaseAgent):
    def __init__(self, client, model_name: str):
        config = get_agent_config("hyperkg_claim_splitter")
        ...

    def process(self, packets: list[SegmentEvidencePacket]) -> list[AtomicClaim]:
        ...

    def split_packet(self, packet: SegmentEvidencePacket) -> list[AtomicClaim]:
        ...
```

### 10.6 Prompt template placeholders

Prompt 内容留空，由 TOML 管理。实现时只需要支持这些 placeholders：

```text
{summary_text}
{local_entities}
{local_triples}
{canonical_entity_map}
{metadata}
```

### 10.7 输出格式要求

LLM 应返回 JSON array。代码需要 parse 成 `AtomicClaim`。

如果 JSON 解析失败：

```text
retry once with same prompt or JSON-repair helper
still fails -> add review item and continue
```

---

## 11. Agent 3: HyperedgeComposerAgent

### 11.1 作用

把 `AtomicClaim` 变成正式的 `EvidenceHyperedge`。

它负责一次性生成：

```text
claim text
participating canonical entities
entity roles
qualifiers
triple projections
source provenance
initial scores
```

### 11.2 输入

```python
claim: AtomicClaim
packet: SegmentEvidencePacket
```

### 11.3 处理逻辑

1. 选择参与 claim 的 canonical entities。
2. 给每个 entity 分配 role。
3. 抽取 qualifiers，例如：
   ```text
   population
   severity
   developmental_stage
   dose
   comparator
   timepoint
   effect_size
   p_value
   sample_size
   outcome_measure
   setting
   ```
4. 生成 triple projections：
   - 必须被 claim_text 或 local triples 支持。
   - 不要生成纯 co-occurrence 关系。
5. 写入 source：
   ```text
   article_id
   segment_id
   summary_id
   section
   ```
6. 生成初始 scores：
   ```text
   entity_linking
   local_kg_agreement
   nary_completeness
   ```
7. 构造 `embedding_text`，但不在 Composer 内部调用 embedding model。

### 11.4 输出

```python
EvidenceHyperedge
```

### 11.5 推荐方法签名

```python
class HyperedgeComposerAgent(BaseAgent):
    def __init__(self, client, model_name: str):
        config = get_agent_config("hyperkg_hyperedge_composer")
        ...

    def process(
        self,
        claims: list[AtomicClaim],
        packet_map: dict[str, SegmentEvidencePacket],
    ) -> list[EvidenceHyperedge]:
        ...

    def compose_one(
        self,
        claim: AtomicClaim,
        packet: SegmentEvidencePacket,
    ) -> EvidenceHyperedge:
        ...
```

### 11.6 Prompt template placeholders

Prompt 内容留空，由 TOML 管理。实现时只需要支持这些 placeholders：

```text
{claim_json}
{packet_json}
{summary_text}
{local_entities}
{local_triples}
{canonical_entity_map}
```

### 11.7 最低质量规则

Composer 生成后做 deterministic validation：

```text
hyperedge must have at least 2 entities
each entity must have entity_id, mention, role
claim_type cannot be empty
source.article_id / source.segment_id / source.summary_id cannot be empty
triple projection cannot reference unknown entity_id
```

不满足规则的 hyperedge 标记为 `REVIEW` 或丢入 review queue。

---

## 12. Agent 4: HyperedgeCriticAgent

### 12.1 作用

检查 evidence hyperedge 是否可靠，并决定是否进入 HyperKG。

### 12.2 输入

```python
hyperedge: EvidenceHyperedge
packet: SegmentEvidencePacket
```

### 12.3 处理逻辑

检查五个维度：

```text
faithfulness:
  claim 是否被 summary 支持

local_kg_agreement:
  hyperedge 的 entities / projections 是否和 local KG 一致

entity_linking:
  entities 是否正确链接到 unified KG canonical entity

scope_completeness:
  是否保留 population, severity, dose, comparator, outcome, timepoint 等关键信息

relation_correctness:
  是否把 ASSOCIATED_WITH 错写成 CAUSAL_OF
  是否把 negation / no significant effect 写反
```

### 12.4 输出

更新后的 `EvidenceHyperedge`，包含：

```python
hyperedge.scores = {
    "faithfulness": 0.0-1.0,
    "local_kg_agreement": 0.0-1.0,
    "entity_linking": 0.0-1.0,
    "scope_completeness": 0.0-1.0,
    "relation_correctness": 0.0-1.0,
    "integration_score": 0.0-1.0
}

hyperedge.decision = "ACCEPT" | "REVIEW" | "REJECT"
```

### 12.5 推荐 score aggregation

```python
integration_score = (
    0.30 * faithfulness
    + 0.20 * entity_linking
    + 0.20 * local_kg_agreement
    + 0.15 * scope_completeness
    + 0.15 * relation_correctness
)
```

### 12.6 推荐 thresholds

```python
ACCEPT if integration_score >= 0.75
REVIEW if 0.55 <= integration_score < 0.75
REJECT if integration_score < 0.55
```

### 12.7 推荐方法签名

```python
class HyperedgeCriticAgent(BaseAgent):
    def __init__(
        self,
        client,
        model_name: str,
        accept_threshold: float = 0.75,
        review_threshold: float = 0.55,
    ):
        config = get_agent_config("hyperkg_hyperedge_critic")
        ...

    def process(
        self,
        hyperedges: list[EvidenceHyperedge],
        packet_map: dict[str, SegmentEvidencePacket],
    ) -> tuple[list[EvidenceHyperedge], list[dict]]:
        ...
```

### 12.8 Prompt template placeholders

Prompt 内容留空，由 TOML 管理。实现时只需要支持这些 placeholders：

```text
{hyperedge_json}
{summary_text}
{local_entities}
{local_triples}
{canonical_entity_map}
```

---

## 13. Agent 5: HyperedgeMergerAgent

### 13.1 作用

把多个 accepted evidence hyperedges 合并成 canonical hyperedges。

前面生成的是 segment-level evidence hyperedges；这个 agent 负责做 global-level integration。

### 13.2 输入

```python
accepted_hyperedges: list[EvidenceHyperedge]
existing_canonical_hyperedges: list[CanonicalHyperedge] | None
embedding_service: EmbeddingService | None
vector_index: VectorIndexAdapter | None
```

### 13.3 处理逻辑

1. 为每个 evidence hyperedge 构造 blocking key。
2. 使用 dense embedding 找出 candidate canonical hyperedges。
3. 对 candidates 做 deterministic filters。
4. 对 uncertain candidate pairs 使用 LLM 判断 merge / new / related / conflict。
5. 合并相同 claim。
6. 保留所有 source-level evidence，不删除冲突 evidence。
7. 生成 support_count、scope_summary、qualifier_summary、confidence_summary。

### 13.4 推荐 blocking key

先做 deterministic blocking，避免全量两两比较：

```python
blocking_key = (
    claim_type,
    sorted(core_entity_ids),
    sorted(core_projection_relations)
)
```

core roles 可以包括：

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

### 13.5 Dense embedding merge logic

推荐流程：

```text
for each evidence hyperedge:
  1. build evidence_hyperedge_embedding_text
  2. search canonical_hyperedge namespace top_k
  3. filter by claim_type
  4. filter by core role/entity compatibility
  5. if similarity >= auto_merge_similarity and filters pass:
       merge deterministically
  6. elif similarity >= llm_merge_similarity:
       call LLM merger prompt
  7. else:
       create new canonical hyperedge
```

推荐阈值：

```python
auto_merge_similarity: float = 0.90
llm_merge_similarity: float = 0.78
max_merge_candidates: int = 10
```

注意：similarity threshold 需要在你的数据集上校准。第一版可以把自动合并阈值设高一点，把不确定样本放入 review 或交给 LLM。

### 13.6 Merge 判断标准

建议同时考虑：

```text
canonical entity overlap
claim_type match
core role pattern match
triple projection overlap
qualifier compatibility
claim_text semantic similarity
```

### 13.7 不要直接合并的情况

```text
same entities but different outcome:
  Risperidone reduces irritability
  Risperidone causes weight gain

same intervention but different population:
  children with ASD
  adults with severe ID

same outcome but different direction:
  intervention reduces aggression
  intervention increases aggression

same relation but negated:
  significant effect
  no significant effect
```

这些应标记为：

```text
RELATED_TO
DIFFERENT_SCOPE
CONTRADICTS
```

### 13.8 输出

```python
list[CanonicalHyperedge]
```

### 13.9 推荐方法签名

```python
class HyperedgeMergerAgent(BaseAgent):
    def __init__(
        self,
        client,
        model_name: str,
        embedding_service: EmbeddingService | None = None,
        vector_index: VectorIndexAdapter | None = None,
        auto_merge_similarity: float = 0.90,
        llm_merge_similarity: float = 0.78,
        max_merge_candidates: int = 10,
    ):
        config = get_agent_config("hyperkg_hyperedge_merger")
        ...

    def process(
        self,
        evidence_hyperedges: list[EvidenceHyperedge],
        existing_canonical_hyperedges: list[CanonicalHyperedge] | None = None,
    ) -> tuple[list[CanonicalHyperedge], list[dict]]:
        ...
```

### 13.10 Prompt template placeholders

Prompt 内容留空，由 TOML 管理。实现时只需要支持这些 placeholders：

```text
{evidence_hyperedge_json}
{candidate_canonical_hyperedges_json}
{merge_criteria}
```

如果没有合适 candidate，可以不调用 LLM，直接创建新的 canonical hyperedge。

---

## 14. Agent 6: HyperKGWriterAgent

### 14.1 作用

把最终 HyperKG 写入本地文件、graph store 和 vector store。

第一版建议先实现 JSONL writer + local vector index writer；Neo4j 可以作为 adapter 后续加入。

### 14.2 输入

```python
evidence_hyperedges: list[EvidenceHyperedge]
canonical_hyperedges: list[CanonicalHyperedge]
packets: list[SegmentEvidencePacket]
review_items: list[dict]
embedding_service: EmbeddingService | None
vector_index: VectorIndexAdapter | None
```

### 14.3 输出文件

推荐输出目录结构：

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
    summary.index
    evidence_hyperedge.index
    canonical_hyperedge.index
    manifest.json
```

### 14.4 JSONL 文件定义

#### evidence_hyperedges.jsonl

每行一个 `EvidenceHyperedge`。不要在 JSONL 中存储大向量。

#### canonical_hyperedges.jsonl

每行一个 `CanonicalHyperedge`。

#### incidence_edges.jsonl

entity-hyperedge incidence edges：

```json
{
  "entity_id": "E:MEDICATION:risperidone",
  "evidence_hyperedge_id": "EH:article123:seg045:01",
  "role": "intervention",
  "mention": "Risperidone",
  "linking_confidence": 0.96
}
```

#### triple_projections.jsonl

```json
{
  "evidence_hyperedge_id": "EH:article123:seg045:01",
  "head_entity_id": "E:MEDICATION:risperidone",
  "relation": "REDUCES",
  "tail_entity_id": "E:SYMPTOM_BEHAVIOR:irritability",
  "confidence": 0.88
}
```

#### summary_links.jsonl

```json
{
  "evidence_hyperedge_id": "EH:article123:seg045:01",
  "article_id": "article123",
  "segment_id": "seg045",
  "summary_id": "SUM:article123:seg045"
}
```

### 14.5 Neo4j schema

如果实现 Neo4j writer，使用 bipartite graph schema：

```text
(:CanonicalEntity)
(:EvidenceHyperedge)
(:CanonicalHyperedge)
(:SegmentSummary)
(:Article)
(:TripleProjection)
```

关系：

```text
(:CanonicalEntity)-[:PARTICIPATES_IN {
    role,
    mention,
    linking_confidence
}]->(:EvidenceHyperedge)

(:EvidenceHyperedge)-[:SUPPORTED_BY]->(:SegmentSummary)

(:SegmentSummary)-[:FROM_ARTICLE]->(:Article)

(:EvidenceHyperedge)-[:MEMBER_OF]->(:CanonicalHyperedge)

(:EvidenceHyperedge)-[:PROJECTS_TO]->(:TripleProjection)

(:CanonicalHyperedge)-[:RELATED_TO]->(:CanonicalHyperedge)

(:CanonicalHyperedge)-[:CONTRADICTS]->(:CanonicalHyperedge)

(:CanonicalHyperedge)-[:DIFFERENT_SCOPE]->(:CanonicalHyperedge)
```

### 14.6 Vector indexes

推荐为以下对象建立 vector index：

```text
canonical_entity_vector_index
evidence_hyperedge_vector_index
canonical_hyperedge_vector_index
summary_vector_index
```

Embedding 文本建议：

```text
entity:
  canonical name + aliases + type + optional description

evidence hyperedge:
  claim_text + claim_type + roles + qualifiers

canonical hyperedge:
  canonical_claim + claim_type + scope_summary + qualifier_summary

summary:
  summary_text + section + article metadata
```

### 14.7 推荐方法签名

```python
class HyperKGWriterAgent:
    def __init__(
        self,
        output_dir: str,
        embedding_service: EmbeddingService | None = None,
        vector_index: VectorIndexAdapter | None = None,
        graph_adapter: object | None = None,
    ):
        ...

    def process(
        self,
        evidence_hyperedges: list[EvidenceHyperedge],
        canonical_hyperedges: list[CanonicalHyperedge],
        packets: list[SegmentEvidencePacket],
        review_items: list[dict],
    ) -> dict:
        ...
```

---

## 15. Pipeline Controller

`HyperKGPipeline` 负责串联 6 个 agents。它不是 agent，不需要 prompt。

### 15.1 推荐方法签名

```python
class HyperKGPipeline:
    def __init__(
        self,
        client,
        model_name: str,
        output_dir: str,
        config: HyperKGConfig | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_index: VectorIndexAdapter | None = None,
    ):
        self.config = config or HyperKGConfig()
        self.embedding_service = embedding_service or build_embedding_service(self.config)
        self.vector_index = vector_index or build_vector_index(self.config)

        self.evidence_packet_agent = EvidencePacketAgent(
            embedding_service=self.embedding_service,
            vector_index=self.vector_index,
            entity_linking_threshold=self.config.entity_linking_threshold,
            entity_linking_top_k=self.config.entity_linking_top_k,
        )
        self.claim_splitter_agent = ClaimSplitterAgent(client, model_name)
        self.hyperedge_composer_agent = HyperedgeComposerAgent(client, model_name)
        self.hyperedge_critic_agent = HyperedgeCriticAgent(
            client,
            model_name,
            accept_threshold=self.config.accept_threshold,
            review_threshold=self.config.review_threshold,
        )
        self.hyperedge_merger_agent = HyperedgeMergerAgent(
            client,
            model_name,
            embedding_service=self.embedding_service,
            vector_index=self.vector_index,
            auto_merge_similarity=self.config.auto_merge_similarity,
            llm_merge_similarity=self.config.llm_merge_similarity,
            max_merge_candidates=self.config.max_merge_candidates,
        )
        self.hyperkg_writer_agent = HyperKGWriterAgent(
            output_dir=output_dir,
            embedding_service=self.embedding_service,
            vector_index=self.vector_index,
        )

    def run(
        self,
        summaries: list[dict],
        local_kgs: list[dict] | dict[str, dict],
        unified_kg: dict,
        metadata: dict | None = None,
        existing_canonical_hyperedges: list[CanonicalHyperedge] | None = None,
    ) -> dict:
        ...
```

### 15.2 推荐运行逻辑

```python
packets = evidence_packet_agent.process(
    summaries=summaries,
    local_kgs=local_kgs,
    unified_kg=unified_kg,
    metadata=metadata,
)

claims = claim_splitter_agent.process(packets)

evidence_hyperedges = hyperedge_composer_agent.process(
    claims=claims,
    packet_map={p.packet_id: p for p in packets},
)

accepted_or_reviewed, review_items = hyperedge_critic_agent.process(
    hyperedges=evidence_hyperedges,
    packet_map={p.packet_id: p for p in packets},
)

accepted = [h for h in accepted_or_reviewed if h.decision == "ACCEPT"]
review = [h for h in accepted_or_reviewed if h.decision == "REVIEW"]

canonical_hyperedges, merge_review_items = hyperedge_merger_agent.process(
    evidence_hyperedges=accepted,
    existing_canonical_hyperedges=existing_canonical_hyperedges,
)

run_stats = hyperkg_writer_agent.process(
    evidence_hyperedges=accepted_or_reviewed,
    canonical_hyperedges=canonical_hyperedges,
    packets=packets,
    review_items=review_items + merge_review_items,
)
```

---

## 16. ID 生成规范

ID 需要稳定、可复现。不要用随机 UUID 作为默认 ID，除非输入缺少必要字段。

### 16.1 Packet ID

```text
P:{article_id}:{segment_id}
```

### 16.2 Claim ID

```text
C:{article_id}:{segment_id}:{claim_index}
```

### 16.3 Evidence Hyperedge ID

```text
EH:{article_id}:{segment_id}:{claim_index}
```

### 16.4 Canonical Hyperedge ID

建议使用 hash：

```text
CH:{claim_type}:{hash(core_entity_ids + core_projection_relations)}
```

### 16.5 Triple Projection ID

```text
TP:{hash(evidence_hyperedge_id + head + relation + tail)}
```

### 16.6 Vector ID

```text
VEC:{namespace}:{object_id}:{embedding_model_hash}
```

---

## 17. Review Queue 规则

以下情况进入 review queue：

```text
entity cannot be linked to unified KG
entity linking confidence low
claim has fewer than 2 linked entities
LLM JSON parsing failed
faithfulness score low
relation correctness score low
integration_score in REVIEW range
claim has high clinical impact but low support
possible contradiction detected
scope difference cannot be resolved
embedding similarity high but deterministic filters disagree
embedding similarity low but LLM suggests merge
```

Review item 格式：

```json
{
  "review_item_id": "REV:000001",
  "type": "LOW_FAITHFULNESS",
  "object_id": "EH:article123:seg045:01",
  "article_id": "article123",
  "segment_id": "seg045",
  "summary_text": "...",
  "object_json": {},
  "reason": "Claim appears stronger than summary evidence.",
  "suggested_action": "Check claim direction and relation type."
}
```

---

## 18. 配置参数

推荐在 pipeline 或 config 文件中支持：

```python
@dataclass
class HyperKGConfig:
    accept_threshold: float = 0.75
    review_threshold: float = 0.55

    min_entities_per_hyperedge: int = 2
    max_claims_per_summary: int = 8

    batch_size: int = 8
    max_llm_retries: int = 1

    enable_neo4j_writer: bool = False
    enable_vector_index: bool = True

    merge_similarity_threshold: float = 0.85
    auto_merge_similarity: float = 0.90
    llm_merge_similarity: float = 0.78
    max_merge_candidates: int = 10

    entity_linking_threshold: float = 0.82
    entity_linking_top_k: int = 10

    embedding_enabled: bool = True
    embedding_model_name: str = ""
    embedding_model_path: str | None = None
    embedding_device: str = "cuda"
    embedding_batch_size: int = 64
    embedding_normalize: bool = True
    embedding_use_fp16: bool = True
    embedding_max_length: int = 512
```

---

## 19. Implementation Phases

### Phase 1: JSONL MVP

实现：

```text
data structures
EvidencePacketAgent
ClaimSplitterAgent
HyperedgeComposerAgent
HyperKGWriterAgent
HyperKGPipeline
```

目标：

```text
summary + local KG + unified KG -> evidence_hyperedges.jsonl
```

可以先不启用 dense embedding，只用 deterministic entity linking 和 JSONL writer。

### Phase 2: Dense Embedding Integration

新增：

```text
EmbeddingService
VectorIndexAdapter
canonical_entity vector index
summary vector index
entity linking by dense retrieval
evidence_hyperedge embedding_text generation
```

目标：

```text
improve entity linking recall
prepare retrieval-ready indexes
reduce future LLM merge calls
```

### Phase 3: Quality Control

新增：

```text
HyperedgeCriticAgent
review_queue.jsonl
score aggregation
threshold filtering
```

目标：

```text
accepted / review / rejected hyperedges
```

### Phase 4: Canonical Hyperedge Layer

新增：

```text
HyperedgeMergerAgent
canonical_hyperedges.jsonl
dense candidate retrieval for merge
scope_summary
conflict / related links
canonical_hyperedge vector index
```

目标：

```text
evidence hyperedges -> canonical hyperedges
```

### Phase 5: Storage and Retrieval Readiness

新增：

```text
Neo4j adapter
FAISS / Qdrant / Milvus / Neo4j vector adapter
incidence edge storage
triple projection storage
```

目标：

```text
retrieval-ready HyperKG
```

---

## 20. Minimal Tests

### 20.1 Unit tests

```text
test_packet_creation_with_full_local_kg
test_packet_creation_with_missing_local_kg
test_entity_linking_exact_match
test_entity_linking_dense_candidate
test_embedding_service_batch_encode
test_vector_index_add_and_search
test_claim_splitter_parses_json_array
test_composer_requires_two_entities
test_composer_rejects_unknown_projection_entity
test_critic_score_aggregation
test_merger_blocks_by_claim_type_and_core_entities
test_merger_uses_embedding_candidates
test_writer_outputs_all_jsonl_files
test_writer_outputs_vector_manifest
```

### 20.2 Integration test

Create three packets:

```text
Packet 1:
  Risperidone reduced irritability in children with ASD.

Packet 2:
  Risperidone decreased irritability among children with autism.

Packet 3:
  Risperidone caused weight gain in children with ASD.
```

Expected behavior:

```text
Packet 1 and Packet 2 -> same canonical hyperedge
Packet 3 -> separate canonical hyperedge
Packet 3 related to Packet 1/2 but not merged
Dense embedding retrieval should retrieve Packet 1 as candidate for Packet 2.
Dense embedding retrieval may retrieve Packet 3 as related, but deterministic filters prevent direct merge.
```

### 20.3 Failure tests

```text
summary empty -> skipped or review
local KG missing -> packet warning, continue
LLM malformed JSON -> retry once, then review
entity unresolved -> provisional / review
projection references unknown entity -> review
embedding model unavailable -> fallback to deterministic mode and warn
vector index unavailable -> write JSONL and skip vector output with warning
```

---

## 21. Acceptance Criteria

Implementation is acceptable when:

```text
1. HyperKG workflow runs without invoking old ingestion / reader / summarizer / entity extraction / relationship extraction.
2. All LLM prompts are loaded from prompts.toml.
3. No long prompt text is hardcoded in agent.py files.
4. Pipeline can process a list of summary/local-KG/unified-KG inputs.
5. Evidence hyperedges have claim_text, claim_type, entities, roles, qualifiers, source.
6. Hyperedges preserve source provenance to article_id, segment_id, summary_id.
7. Triple projections only reference known canonical entity IDs.
8. Critic produces ACCEPT / REVIEW / REJECT decisions.
9. Merger produces canonical hyperedges and does not merge adverse-effect claims with intervention-outcome claims.
10. Writer generates JSONL outputs and run_stats.json.
11. EmbeddingService is optional but supported through config.
12. When embedding is enabled, canonical entities, evidence hyperedges, canonical hyperedges, and summaries can be indexed.
13. Merger uses dense retrieval for candidate generation before calling LLM.
14. Dense embeddings are not stored directly inside large JSONL records.
```

---

## 22. Practical Notes for Codex

Keep the first implementation simple:

```text
Use dataclasses first.
Use JSONL writer first.
Use deterministic entity linking first.
Use EmbeddingService as a shared infrastructure object, not as an agent.
Use dense embedding for candidate retrieval and merge pre-ranking.
Use deterministic merge blocking first.
Use LLM only for claim splitting, composing, critic, and uncertain merge decisions.
Do not implement graph neural networks.
Do not implement hypergraph transformer.
Do not retrain embeddings.
Do not re-read raw PDFs.
```

Preferred implementation style:

```text
small classes
clear process() methods
typed dataclasses
robust JSON parsing
explicit review queue
shared embedding service
batch embedding calls
no silent failures
```

Non-goals for this implementation:

```text
new summary generation
new local KG extraction
new unified KG construction
end-user QA generation
full HyperGraphRAG retrieval
hypergraph neural network learning
supervised embedding training
```

---

## 23. Final HyperKG Semantics

The final HyperKG should answer the following questions for every hyperedge:

```text
What is the claim?
Which canonical entities participate in it?
What role does each entity play?
What qualifiers or scope constraints apply?
Which summary / segment / article supports it?
Which binary triples can it project to?
Is it accepted, rejected, or queued for review?
Which canonical hyperedge cluster does it belong to?
Does it support, contradict, or differ in scope from another claim?
Can it be retrieved by semantic similarity through a vector index?
```

This is the core value of the HyperKG layer: it converts already extracted KG and summary artifacts into a source-grounded, textual-rich, n-ary claim graph, while using dense embeddings to make linking, merging, and retrieval scalable.
