"""End-to-end article build with a fake LLM and hashing embeddings."""

from __future__ import annotations

import json

from workflows.HyperKGCreator.config import load_default_config
from workflows.HyperKGCreator.pipeline.build_article import build_article_hkg
from workflows.HyperKGCreator.storage.jsonl_store import read_jsonl


def test_end_to_end_article_build(tmp_path, mini_article, fake_llm, hashing_model, catalog_path):
    out_dir = tmp_path / "out"
    res = build_article_hkg(
        pdf_path=None,
        entity_catalog_path=catalog_path,
        output_dir=out_dir,
        config=load_default_config(),
        llm_client=fake_llm,
        embedding_model=hashing_model,
        article=mini_article,
    )

    assert res.article_id == "ART1"
    assert res.n_sections == 3
    assert res.n_pto_events == 1
    assert res.n_hyperedges >= 1

    # PTO graph + hyperedges written.
    pto_path = out_dir / "pto_graphs" / "ART1.json"
    assert pto_path.exists()
    pto = json.loads(pto_path.read_text())
    assert pto["article_id"] == "ART1"

    hyperedges = read_jsonl(out_dir / "hyperedges.jsonl")
    assert hyperedges
    section_ids = {s.section_id for s in mini_article.sections}
    for he in hyperedges:
        assert he["type"] == "ID_TREATMENT_RESPONSE"
        for sid in he["source"]["evidence_span_ids"]:
            assert sid in section_ids, f"{sid} is not a section id"

    # Incidence + validation + verification artifacts present.
    incidence = read_jsonl(out_dir / "incidence_edges.jsonl")
    assert incidence
    assert {r["role"] for r in incidence} <= {"PATIENT", "TREATMENT", "OUTCOME"}

    reports = read_jsonl(out_dir / "validation_reports.jsonl")
    assert reports
    verifications = read_jsonl(out_dir / "verification_results.jsonl")
    assert verifications and verifications[0]["status"] == "accepted"

    # Vector indexes saved.
    assert (out_dir / "section_embeddings.index").exists()
    assert (out_dir / "entity_embeddings.index").exists()
