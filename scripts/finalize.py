"""One command to produce the final, semantic-backend deliverables (needs internet on first run).

    python scripts/finalize.py

Steps, each must succeed or the script stops:
  1. pipeline with the sentence-transformer model, strict (no silent TF-IDF fallback)
  2. evaluation report (semantic + TF-IDF baseline)
  3. reproducibility check: 3 fresh-process runs, strict semantic
  4. verdict invariance: semantic k=5 vs semantic k=all vs TF-IDF k=5, graded columns cell by cell
  5. test suite
  6. README retrieval table and backend note rewritten from the reports
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GRADED = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]


def step(title: str, cmd: list[str]) -> None:
    print(f"\n=== {title}\n$ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"\nFAILED at: {title}. Nothing was written to the README.")


def main() -> int:
    py = sys.executable
    step("1/6 pipeline (strict semantic)", [py, "run.py", "--strict-semantic"])
    step("2/6 evaluation", [py, "scripts/evaluate.py", "--strict-semantic"])
    step("3/6 reproducibility (3 runs)", [py, "scripts/reproducibility_check.py", "--strict-semantic"])

    print("\n=== 4/6 verdict invariance", flush=True)
    from src.config import DEFAULT_CONFIG
    from src.pipeline import run_pipeline
    from src.retrieval import get_embedder
    emb = get_embedder("semantic", DEFAULT_CONFIG.embedding_model, allow_fallback=False)
    sem5 = run_pipeline(embedder=emb).output[GRADED]
    sem_all = run_pipeline(embedder=emb, cfg=replace(DEFAULT_CONFIG, retrieval_top_k=10_000)).output[GRADED]
    tfidf5 = run_pipeline(retriever="tfidf").output[GRADED]
    invariance = {"semantic k=5 vs semantic k=all notes": sem5.equals(sem_all),
                  "semantic vs TF-IDF": sem5.equals(tfidf5)}
    for name, same in invariance.items():
        print(f"  {name}: {'identical' if same else 'DIFFERENT'}")
        if not same:
            other = sem_all if "k=all" in name else tfidf5
            diff = sem5.compare(other)
            print(diff.head(20).to_string())

    step("5/6 tests", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider"])

    print("\n=== 6/6 README sync", flush=True)
    from scripts.readme_sync import sync
    sync(ROOT / "README.md", ROOT / "outputs", invariance)
    print("README updated. Review `git diff`, then commit outputs/ and README.md.")
    if not all(invariance.values()):
        print("\nWARNING: semantic verdicts differ from another configuration - inspect the diff above "
              "before submitting; the README records it.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
