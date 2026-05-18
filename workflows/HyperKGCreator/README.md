# HyperKGCreator

A compact pipeline that builds an **intellectual-disability treatment-response
Hyper Knowledge Graph** from raw journal-article PDFs.

Each final HyperEdge (`type = "ID_TREATMENT_RESPONSE"`) captures one
observation: **patient context + treatment/intervention + outcome/result**.
The pipeline first builds an article-level Patient–Treatment–Outcome (PTO)
graph, then converts PTO events into HyperEdges.

## The section-level span rule (important)

The evidence span unit is a **SECTION**. `ArticleSpan == SectionSpan`.

- Every `evidence_span_id` is a section id. There are no paragraph, sentence,
  or token-chunk evidence ids.
- Tables are attached to the section they belong to; if a table cannot be
  confidently assigned it goes to the nearest previous section, or to a
  synthetic `__UNASSIGNED_TABLES__` section.
- Long sections may be split internally only to satisfy embedding-model length
  limits. Those internal pieces are pooled into one section vector and are
  **never** exposed as spans, PTO/HyperEdge sources, incidence records, or
  verifier inputs.
- The conditional verifier may receive optional section-local **quotes** as
  advisory context. Quotes are never evidence ids and the schema validator
  rejects any attempt to use one as an id.

## Architecture

```
Raw PDFs → PDF/Layout/Table Parser → SectionSpan store
  → offline section embedding index → embedding semantic router
  → Agent 1: Article PTO Builder
  → offline entity candidate retrieval
  → Agent 2: HyperEdge Builder + Entity Selector
  → schema validator
  → offline support scorer + risk gate
  → Agent 3: Conditional Verifier (only if triggered)
  → JSONL HyperKG writer + vector indexes
```

Offline embeddings own routing, entity candidate retrieval, support scoring and
risk gating. Exactly three LLM stages exist: `PTOBuilderAgent`,
`HyperEdgeBuilderEntitySelectorAgent`, `ConditionalVerifierAgent`.

## Install

The package lives at `workflows/HyperKGCreator/` and is run as a module from the
repo root. Most dependencies are already in the repo root `requirements.txt`;
extras are listed in `workflows/HyperKGCreator/requirements.txt`:

```bash
pip install -r requirements.txt                       # repo root
pip install -r workflows/HyperKGCreator/requirements.txt
```

`faiss`, `scikit-learn`, `spacy` and `PyYAML` are optional — the code falls
back to numpy cosine / phrase-window / JSON automatically.

## User-written prompts

Prompt files are **placeholders** for the project owner. They contain only:

```
TODO: project owner will write this prompt.
```

Files (the owner edits these in place):

- `workflows/HyperKGCreator/prompts/pto_builder.txt`
- `workflows/HyperKGCreator/prompts/hyperedge_builder_entity_selector.txt`
- `workflows/HyperKGCreator/prompts/conditional_verifier.txt`

A real LLM client meeting an empty placeholder raises a clear error. Owner
prompts should use `{{variable}}` placeholders (double braces) so JSON braces
inside a prompt are never touched. No prompt content is generated in code.

## Offline embedding model

Set the model path/name and device in config or via env:

```bash
export HKG_EMBEDDING_MODEL=/path/to/local/sentence-transformers-model
export HKG_EMBEDDING_DEVICE=cpu          # or cuda
export HKG_FAKE_EMBEDDINGS=1             # deterministic hashing backend (tests / no model)
```

Default model name: `sentence-transformers/all-MiniLM-L6-v2` (must be locally
available; no internet is assumed).

## Entity catalog

A **starter/demo** catalog is shipped at
`workflows/HyperKGCreator/data/entity_catalog.jsonl` (derived from
`KARMA/entity_type_grouping.md`). It is for out-of-the-box runs and tests only;
loading it for a build logs a warning. Provide your curated catalog (JSONL or
CSV with `entity_id, canonical_name, entity_type, synonyms, description`) via
`--entity-catalog`. `entity_type` must be one of the KARMA ontology types.

## Commands

```bash
# PDFs → ArticleDocument JSON (no LLM)
python -m workflows.HyperKGCreator.cli parse \
    --pdf-dir testbase --out-dir output/hkg

# Build entity vector index (no LLM)
python -m workflows.HyperKGCreator.cli index-entities \
    --entity-catalog workflows/HyperKGCreator/data/entity_catalog.jsonl \
    --out-dir output/hkg

# Build hyperedge vector index (no LLM)
python -m workflows.HyperKGCreator.cli index-hyperedges \
    --hyperedges-jsonl output/hkg/hyperedges.jsonl --out-dir output/hkg

# Validate artifacts (no LLM)
python -m workflows.HyperKGCreator.cli validate \
    --article-json output/hkg/article_documents/<id>.json

# Full single-article build (requires an LLM client + owner prompts)
python -m workflows.HyperKGCreator.cli build-article \
    --pdf testbase/<file>.pdf \
    --entity-catalog workflows/HyperKGCreator/data/entity_catalog.jsonl \
    --out-dir output/hkg

# Full dataset build, optionally driven by CSVs
python -m workflows.HyperKGCreator.cli build-dataset \
    --pdf-dir testbase \
    --entity-catalog workflows/HyperKGCreator/data/entity_catalog.jsonl \
    --out-dir output/hkg \
    --metadata-csv data/metadata/CT.csv \
    --selected-csv workflows/filtering/CT_selected.csv
```

`parse`, `index-entities`, `index-hyperedges` and `validate` never call an LLM.

## Generated artifacts (under `--out-dir`)

```
article_documents/{article_id}.json
pto_graphs/{article_id}.json
hyperedges.jsonl
entities.jsonl
incidence_edges.jsonl          # entity_id --role--> hyperedge_id
validation_reports.jsonl
verification_results.jsonl
section_embeddings.index
entity_embeddings.index
hyperedge_embeddings.index
```

## Clinical safety

This is HKG construction, not medical advice. The build pipeline emits no
treatment recommendations. Any retrieval helper must enforce source side =
PATIENT + TREATMENT and target side = OUTCOME; OUTCOME entities are never used
as recommendation-time seed nodes.

## Tests

```bash
cd <repo root>
HKG_FAKE_EMBEDDINGS=1 python -m pytest workflows/HyperKGCreator/tests -q
```

Tests are offline and deterministic (FakeLLMClient + hashing embeddings); no
real LLM call and no model download.
