# HyperKGConstruction

This workflow builds a textual-rich HyperKG from already-generated
segment summaries, local KGs, and a unified KG. It does not run PDF
ingestion, summarization, entity extraction, relation extraction, or
unified KG construction.

Prompts are intentionally blank in `prompts.toml`. Fill the relevant TOML
sections before running the LLM-backed stages:

- `hyperkg_claim_splitter`
- `hyperkg_hyperedge_composer`
- `hyperkg_hyperedge_critic`
- `hyperkg_hyperedge_merger`

Run from KARMA artifacts:

```bash
python -m workflows.HyperKGConstruction.run run-karma-output \
  --karma-output-dir KARMA/output \
  --output-dir output/hyperkg \
  --workers 8
```

The OpenRouter API key is loaded from `.env` by default:

```bash
OPENROUTER_API_KEY=sk-or-...
```

You can also pass `--env-file path/to/.env` or override with
`--api-key`. The default API base URL is
`https://openrouter.ai/api/v1`; set `OPENROUTER_BASE_URL` or pass
`--api-base-url` to override it.

`run-karma-output` will automatically read per-paper summaries from
`KARMA/output/summaries/` and per-paper KGs from `KARMA/output/local_kg/`,
with fallback to the older flat layout if those subfolders are absent.
It also defaults to `KARMA/output/unified_kg/unified_kg.json` for the
unified KG when `--unified-kg` is omitted.

The current KARMA local KGs are article-level while summaries are stored
as per-article lists. The packet builder therefore filters each article
KG down to summary-relevant entities/triples before passing it to LLM
agents. Tune the prompt footprint with `--max-local-entities` and
`--max-local-triples`.

Main outputs:

- `packets.jsonl`
- `claims.jsonl`
- `evidence_hyperedges.jsonl`
- `canonical_hyperedges.jsonl`
- `incidence_edges.jsonl`
- `triple_projections.jsonl`
- `summary_links.jsonl`
- `review_queue.jsonl`
- `run_stats.json`
- `vector_indexes/manifest.json` when embeddings are enabled

Long runs write `<output-dir>/run.log` and checkpoint files under
`<output-dir>/checkpoints/` by default. If a run is interrupted, rerun the
same command to resume completed items. Use `--no-resume` to ignore existing
checkpoint files, `--no-checkpoint` to disable intermediate saves, and
`--no-progress` to disable terminal progress bars.
