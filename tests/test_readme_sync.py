"""The README's backend-dependent blocks are generated from reports, never hand-edited."""
import json
import shutil

import pytest

from scripts.readme_sync import replace_block, sync
from tests.conftest import ROOT


def _outputs(tmp_path, backend):
    out = tmp_path / "outputs"
    out.mkdir()
    sem = {"model": "sentence-transformers/all-MiniLM-L6-v2", "gold_justification_recall@5": 1.0,
           "gold_justification_recall@1": 1.0, "route_specific_recall@5": 1.0, "route_specific_mrr": 0.95}
    tf = {"model": "tfidf", "gold_justification_recall@5": 1.0, "gold_justification_recall@1": 0.67,
          "route_specific_recall@5": 1.0, "route_specific_mrr": 0.818}
    backends = {"tfidf": tf} if backend == "tfidf" else {"sentence-transformers": sem, "tfidf": tf}
    (out / "evaluation_report.json").write_text(json.dumps(
        {"retrieval": {"top_k_used_in_pipeline": 5, "backends": backends}}))
    (out / "run_metadata.json").write_text(json.dumps({"retrieval": {"backend": backend, "model": "M"}}))
    (out / "cost_report.json").write_text(json.dumps(
        {"embeddings": {"notes_embedded": 10, "queries_embedded": 19, "tokens_embedded": 1234}}))
    readme = tmp_path / "README.md"
    shutil.copy(ROOT / "README.md", readme)
    return readme, out


def test_semantic_results_replace_placeholder(tmp_path):
    readme, out = _outputs(tmp_path, "sentence-transformers")
    text = sync(readme, out, {"semantic k=5 vs semantic k=all notes": True, "semantic vs TF-IDF": True})
    assert "not measured yet" not in text and "| 0.950 |" in text
    assert "1234 tokens" in text and "huggingface.co was blocked" not in text
    assert "semantic vs TF-IDF: identical" in text


def test_tfidf_keeps_honest_placeholder(tmp_path):
    readme, out = _outputs(tmp_path, "tfidf")
    text = sync(readme, out)
    assert "not measured yet" in text and "produced with the TF-IDF retriever" in text


def test_sync_is_idempotent(tmp_path):
    readme, out = _outputs(tmp_path, "tfidf")
    assert sync(readme, out) == sync(readme, out)


def test_missing_markers_raise():
    with pytest.raises(ValueError):
        replace_block("no markers here", "RETRIEVAL", "x")
