"""Rewrite the backend-dependent README blocks from the actual reports (never by hand).

Blocks are delimited by <!-- NAME:START --> / <!-- NAME:END --> markers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = {"sentence-transformers": "sentence-transformer", "tfidf": "TF-IDF (baseline)"}


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README is missing the {name} markers")
    return pattern.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text)


def retrieval_block(report: dict, invariance: dict | None) -> str:
    r = report["retrieval"]
    k = r["top_k_used_in_pipeline"]
    rows = [f"Retrieval quality on the 19 flagged-week queries (is the right note in the top {k}?):", "",
            f"| Retriever | correct note in top {k} | correct note ranked 1st | route's own notes in top {k} | MRR |",
            "|---|---|---|---|---|"]
    for backend in ["sentence-transformers", "tfidf"]:
        v = r["backends"].get(backend)
        if v is None:
            rows.append(f"| {LABELS[backend]} | not measured yet (run `python scripts/finalize.py` with internet) "
                        f"| | | |")
            continue
        rows.append(f"| {LABELS[backend]} | {v[f'gold_justification_recall@{k}']:.2f} | "
                    f"{v['gold_justification_recall@1']:.2f} | {v[f'route_specific_recall@{k}']:.2f} | "
                    f"{v['route_specific_mrr']:.3f} |")
    if invariance:
        rows += ["", "Verdict comparison (graded columns, cell by cell): "
                 + "; ".join(f"{name}: {'identical' if same else 'different'}" for name, same in invariance.items())
                 + "."]
    return "\n".join(rows)


def backend_block(meta: dict, cost: dict) -> str:
    r = meta["retrieval"]
    if r["backend"] == "sentence-transformers":
        e = cost["embeddings"]
        return (f"The committed outputs use `{r['model']}` running locally on CPU. A full run embeds "
                f"{e['notes_embedded']} notes and {e['queries_embedded']} queries ({e['tokens_embedded']} tokens "
                f"by the model's tokenizer), with no API calls.")
    return ("Note on the committed outputs: they were produced with the TF-IDF retriever, because the "
            "environment they were generated in could not download the sentence-transformer weights "
            "(huggingface.co was blocked). The verdicts don't depend on the retriever (see above). "
            "Running `python scripts/finalize.py` with internet access regenerates everything with the "
            "semantic model and updates this paragraph and the retrieval table.")


def sync(readme: Path, out_dir: Path, invariance: dict | None = None) -> str:
    report = json.loads((out_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cost = json.loads((out_dir / "cost_report.json").read_text(encoding="utf-8"))
    text = readme.read_text(encoding="utf-8")
    text = replace_block(text, "RETRIEVAL", retrieval_block(report, invariance))
    text = replace_block(text, "BACKEND", backend_block(meta, cost))
    readme.write_text(text, encoding="utf-8", newline="\n")
    return text


if __name__ == "__main__":
    sync(ROOT / "README.md", ROOT / "outputs")
    print("README synced from outputs/")
