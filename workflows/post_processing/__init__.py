"""
Post-processing workflow for aggregating per-article Knowledge Graphs
into a unified KG for RAG-based retrieval.

Pipeline stages:
  1. Loader       – load all per-article _kg.json files with provenance
  2. EntityTyper  – recover entity types via LLM (Gemini Flash)
  3. EntityResolver – embedding-based cross-article entity resolution (BGE-large on GPU)
  4. TripleMerger – remap + deduplicate + aggregate triples
  5. ConflictResolver – LLM-based contradiction arbitration
  6. Exporter     – write unified_kg.json
"""

__version__ = "0.1.0"
