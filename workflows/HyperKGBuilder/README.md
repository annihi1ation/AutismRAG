# HyperKGBuilder

Embedding-routed HyperKG construction workflow over **already-collected**
claims and an **already-built** unified KG. This workflow does **not**
re-run ingestion, reader, summarizer, entity / relationship extraction, or
unified-KG construction.

## Inputs

- Claims JSONL: `output/hyperkg_gpt54mini_full_gpu6/claim_top1/results.jsonl`
- Unified KG: `KARMA/output/unified_kg/unified_kg.json`
- OpenRouter API key: `.env` → `OPENROUTER_API_KEY`

## Pipeline (4 phases)

1. **ClaimKGEmbeddingIndexAgent** — builds vector indexes for claims,
   unified-KG entities, unified-KG triples, and segment summaries.
   No LLM calls.
2. **CandidatePackRouterAgent** — clusters near-duplicate claims, retrieves
   bounded candidates from the indexes, and routes each pack to either
   `auto` (deterministic) or `online_llm`.
3. **OnlineLLMHyperedgeAgent** — the only agent that calls the Online LLM
   (default: `deepseek/deepseek-v4-pro` via OpenRouter). Operates on bounded
   work units only; never receives the full unified KG.
4. **HyperKGWriterIndexerAgent** — aggregates evidence hyperedges into
   canonical hyperedges, writes JSONL outputs, and saves vector indexes.

## CLI

```
python -m workflows.HyperKGBuilder \
  --claims-path output/hyperkg_gpt54mini_full_gpu6/claim_top1/results.jsonl \
  --unified-kg-path KARMA/output/unified_kg/unified_kg.json \
  --output-dir output/hyperkg_builder/run_001 \
  --embedding-model-name <hf-id-or-local-path> \
  --embedding-device cuda:6 \
  --model-name deepseek/deepseek-v4-pro
```

Add `--dry-run` to stop after routing (no LLM calls). Add `--no-resume` to
ignore existing checkpoints.

## Outputs

Under `--output-dir`:

- `candidate_packs.jsonl`
- `evidence_hyperedges.jsonl`
- `canonical_hyperedges.jsonl`
- `triple_projections.jsonl`
- `incidence_edges.jsonl`
- `summary_links.jsonl`
- `vector_indexes/manifest.json` (+ `.metadata.jsonl` / `.vectors.npy`)
- `hyperkg_run_report.json`
- `llm_usage_report.json`
- `review_queue.json`

## Tests

```
python -m unittest discover workflows/HyperKGBuilder/tests
```
