# KARMA Workspace Overview

This document explains how the KARMA pipeline is configured in this workspace, what it extracts, which model is being used, and how many papers have already been processed.

It is based on the current code and files in this repository, not just the high-level marketing description in the main README.

## What KARMA Does

KARMA is a multi-stage pipeline that reads scientific documents and turns them into structured knowledge graph outputs.

In the current workspace, the prompt set is tailored to literature about intellectual disability, challenging behavior, interventions, care systems, and related psychosocial or clinical outcomes.

At a high level, each paper goes through these steps:

1. Extract text from PDF or read plain text.
2. Extract document metadata.
3. Split the text into segments and score segment relevance.
4. Summarize the relevant segments.
5. Extract entities from the summaries.
6. Extract relationships between those entities.
7. Normalize entity types and relation labels.
8. Resolve obvious contradictions and keep only triples that pass final scoring.

## Model Used

There is more than one default in this codebase, so it is important to distinguish between the library defaults and the script used for the current batch outputs.

### API backend

- The pipeline uses the OpenAI Python client.
- By default, it points that client to the OpenRouter API base URL: `https://openrouter.ai/api/v1`.

### Model defaults in code

- Core pipeline default: `deepseek/deepseek-v3.2`
- CLI default in `karma/cli.py`: `deepseek/deepseek-v3.2`
- `main.py` default: `deepseek/deepseek-v3.2`

### Model used by the current workspace batch script

- `run.py` explicitly sets the model to `google/gemini-3-flash-preview`
- `karma.sh` also defaults to `google/gemini-3-flash-preview`

### Practical interpretation

If you run the library or CLI without overriding the model, KARMA uses `deepseek/deepseek-v3.2` through OpenRouter.

If you run the current workspace batch script in `run.py`, it uses `google/gemini-3-flash-preview` through OpenRouter.

## Current Paper Counts And Output Counts

The numbers below reflect the current workspace snapshot and the files currently present in `testbase/` and the `KARMA/output/` subfolders.

### Source corpus

- Source PDF folder used by `run.py`: `/data2/leyizhao/CommTool/testbase`
- PDF count in that folder: `406`

### Generated output files

- Per-paper summary files in `KARMA/output/summaries/`: `406`
- Per-paper knowledge graph files in `KARMA/output/local_kg/`: `406`
- Aggregate knowledge graph file in `KARMA/output/unified_kg/`: `1` (`unified_kg.json`)
- Total `*_kg.json` files across `KARMA/output/local_kg/` and `KARMA/output/unified_kg/`: `407`

### Output content totals

- Documents with at least one integrated triple: `403` of `406`
- Total triples across all per-paper KG files: `24,072`
- Total entity entries across all per-paper KG files: `43,743`
- Unified graph entity count: `3,801`
- Unified graph triple count: `15,644`
- Unified graph unique relation labels: `20`

### Important note about these totals

The per-paper totals count repeated entities and repeated triples across different papers. The unified graph is smaller because it merges repeated entities and relations into one aggregate graph.

## What Is Being Extracted

KARMA does not only extract one final graph. It extracts several layers of information.

### Metadata

For each paper, the ingestion stage tries to extract:

- title
- authors
- journal
- publication date
- DOI
- PMID
- document type

### Entities

The prompt configuration is specialized for intellectual disability and challenging behavior literature. The entity extractor is instructed to pull domain-relevant mentions such as:

- disorders and syndromes
- symptom and behavior terms
- emotional states
- functional abilities
- assessment tools
- medications
- therapeutic approaches
- educational approaches
- person roles such as caregivers, clinicians, and teachers
- social or care environment factors
- healthcare access or system factors

The full entity ontology in `karma/agents/prompts.toml` is broader than the old README examples and is explicitly tuned to this intellectual-disability-focused corpus.

### Relationships

The relationship prompts are also domain-specific. They target relation types such as:

- `CAUSAL_OF`
- `RISK_FACTOR_FOR`
- `ASSOCIATED_WITH`
- `EXACERBATES`
- `HAS_DIAGNOSIS`
- `EXHIBITS_SYMPTOM`
- `HAS_SEVERITY_LEVEL`
- `REDUCES`
- `INCREASES`
- `IMPROVES_FUNCTION`
- `PREVENTS`
- `ADVERSE_EFFECT`

One implementation detail matters here: the prompt vocabulary is rich, but the current schema-alignment step only performs light rule-based normalization. That means final relation labels may include both normalized forms like `associated_with` and more direct raw relation phrases depending on what the extractor returned.

## How Extraction Works

This section describes the actual implemented pipeline.

### 1. Document loading

`KARMAPipeline.process_document()` accepts either a path or raw text.

- PDFs are read through `PDFReader`
- Extraction priority is:
  - PyMuPDF (`fitz`)
  - PyPDF2 fallback
- Post-processing removes common PDF artifacts such as broken line wraps and hyphenation across line breaks

### 2. Ingestion agent

The Ingestion Agent:

- uses the first 5,000 characters to extract metadata with an LLM call
- falls back to regex-based DOI, PMID, title, and author extraction if needed
- normalizes whitespace
- applies some OCR cleanup and Unicode normalization

Examples of normalization include ligature repair, replacing Greek characters such as `alpha` and `beta`, and preserving paragraph breaks for later segmentation.

### 3. Reader agent

The Reader Agent:

- splits normalized text on blank lines into paragraph-like segments
- labels segments with rough section types such as `abstract`, `methods`, `results`, `discussion`, `references`, and `content`
- scores segments for extraction relevance in batches of five using the LLM
- keeps only segments whose score is greater than or equal to the relevance threshold

Default relevance threshold in most entry points is `0.2`.

### 4. Summarizer agent

The Summarizer Agent turns each relevant segment into a short summary.

Its instructions emphasize preserving:

- domain terms
- participant details
- numbers and statistics
- p-values and measurements
- intervention and outcome language

If summarization fails or the content is too weak, it falls back to a sentence-ranking approach and returns the most informative sentences under a word budget.

### 5. Entity extraction agent

The Entity Extraction Agent:

- runs on the summaries rather than the full paper text
- asks the LLM to return a JSON array of entities
- stores each entity with a mention, type, normalized identifier, and aliases
- deduplicates entities case-insensitively across all summaries from the same paper

If the LLM response fails, the code falls back to regex patterns for a smaller set of biomedical-style entities.

### 6. Relationship extraction agent

The Relationship Extraction Agent:

- takes the summaries plus the extracted entity list
- asks the LLM for direct relationships between those entities
- expects JSON output with `head`, `relation` or `raw_code`, `tail`, and optionally `confidence`
- keeps only triples whose head and tail can be matched back to known entities

The code then assigns:

- `confidence` from the extracted response
- `clarity` from simple heuristics about specificity and relation wording
- `relevance` from simple heuristics about clinical or biomedical importance

### 7. Schema alignment agent

The Schema Alignment Agent is currently lightweight and mostly rule-based.

It:

- fills in unknown entity types using suffix or pattern heuristics
- normalizes some common relation synonyms

Examples:

- `inhibit` -> `inhibits`
- `treat` -> `treats`
- `cause` -> `causes`
- `associated with` -> `associated_with`

### 8. Conflict resolution agent

The Conflict Resolution Agent compares candidate triples against the knowledge graph already accumulated in memory during the run.

The current logic is intentionally simple:

- it only checks a small set of contradictory relation pairs
- it only compares triples that share the same head and tail
- if two relations contradict, it keeps the higher-confidence triple

### 9. Evaluator agent

The Evaluator Agent decides whether a triple is integrated into the final output.

It computes:

`integration_score = 0.5 * confidence + 0.25 * clarity + 0.25 * relevance`

Only triples at or above the configured integration threshold are kept.

Current threshold behavior depends on the entry point:

- `run.py`: uses the config default integration threshold of `0.6`
- `main.py`: defaults to `0.5`
- `karma.sh`: defaults to `0.5`

## Files Produced

### Per-paper files

When running `process_batch()` with the current workspace setup, KARMA writes:

- `summaries/<paper_stem>_summaries.json`
- `local_kg/<paper_stem>_kg.json`

Each per-paper KG file contains:

- `entities`
- `triples`
- `metadata`
- `statistics`

### Optional intermediate files

Some entry points such as `main.py` or the CLI can also save intermediate pipeline state, for example:

- all intermediate stage outputs
- CSV exports of extracted relationships
- a separately exported aggregate graph

### Aggregate graph

The `unified_kg/unified_kg.json` file under `KARMA/output/` is an aggregate graph artifact, not an extra paper.

## Important Implementation Notes

### The code currently wires eight agents

The public README mentions nine specialized agents, but the implemented pipeline in `karma/core/pipeline.py` currently instantiates these eight agents:

1. Ingestion
2. Reader
3. Summarizer
4. Entity Extraction
5. Relationship Extraction
6. Schema Alignment
7. Conflict Resolution
8. Evaluator

This document follows the code, not the older description.

### The prompts are ID-focused, not generic biomedical prompts

Some older documentation still frames KARMA as a general biomedical KG system. In this workspace, the active prompt templates are much more specific to:

- intellectual disability
- challenging behavior
- interventions
- care systems
- psychosocial and clinical context

That is why the extracted entities and relations are better interpreted as an intellectual-disability knowledge graph rather than a generic biomedical graph.

### Batch processing accumulates knowledge across papers

`process_batch()` processes papers one by one and keeps an in-memory graph during the run. That means later papers can be checked against triples collected from earlier papers during the conflict-resolution step.

## Main Files To Read If You Want To Extend This

- `run.py`: current workspace batch runner
- `main.py`: single-document runner with richer console output
- `karma/core/pipeline.py`: orchestration logic
- `karma/config/settings.py`: model and threshold defaults
- `karma/agents/prompts.toml`: domain-specific prompt instructions
- `karma/utils/pdf_reader.py`: PDF extraction backend

## Short Summary

In this workspace, KARMA is processing `406` papers from `testbase/` into per-paper summaries and knowledge graphs. The current batch runner uses `google/gemini-3-flash-preview` through OpenRouter, while the reusable library defaults to `deepseek/deepseek-v3.2`. The extraction flow is PDF or text ingestion, metadata extraction, relevance-based segmentation, segment summarization, entity extraction, relationship extraction, schema alignment, contradiction filtering, and final weighted integration into the graph.