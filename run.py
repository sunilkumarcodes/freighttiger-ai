"""Run the full pipeline and write all deliverables to outputs/.

    python run.py                      # semantic retrieval (falls back to TF-IDF if model unavailable)
    python run.py --retriever tfidf    # lexical baseline
    python run.py --strict-semantic    # fail instead of falling back
    python run.py --out-dir outputs/repro/run_1 --quiet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG, OUTPUT_DIR
from src.pipeline import run_pipeline, write_submission_csv
from src.retrieval import EmbeddingModelUnavailable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_report(result, elapsed: float) -> dict:
    r = result.retrieval_info
    semantic = r["backend"] == "sentence-transformers"
    return {
        "llm": {
            "model": None,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "price_per_token_usd": None,
            "estimated_cost_usd": 0.0,
            "why_zero": ("No generative LLM is called. Explanations are deterministic templates filled with "
                         "computed numbers and verbatim note text; verdicts come from deterministic validation."),
        },
        "embeddings": {
            "backend": r["backend"],
            "model": r["model"],
            "computed_locally": True,
            "api_calls": 0,
            "texts_embedded": r["texts_embedded"],
            "notes_embedded": r["texts_embedded"] - r["queries"],
            "queries_embedded": r["queries"],
            "tokens_embedded": r["tokens_embedded"] if semantic else None,
            "tokens_note": ("Counted with the model's own tokenizer (includes special tokens)." if semantic else
                            "Not applicable: TF-IDF is a sparse lexical model with no subword tokenizer."),
            "cost_usd": 0.0,
        },
        "call_volume_design": ("One retrieval query per numerically anomalous route-week (not per shipment); "
                               "all queries embedded in a single batch."),
        "runtime_seconds": round(elapsed, 2),
        "measurement_caveats": "runtime_seconds is wall-clock on the machine that produced this file.",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retriever", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--strict-semantic", action="store_true", help="error if the embedding model can't load")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    t0 = time.perf_counter()
    try:
        result = run_pipeline(cfg=DEFAULT_CONFIG, retriever=args.retriever,
                              allow_fallback=not args.strict_semantic)
    except (FileNotFoundError, ValueError, EmbeddingModelUnavailable) as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1
    elapsed = time.perf_counter() - t0

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final_output.csv"
    write_submission_csv(result.output, final)
    write_submission_csv(result.output[result.output["flagged"] != "No"], out / "flagged_only.csv")
    result.audit.to_csv(out / "route_week_audit.csv", index=False, lineterminator="\n")

    o = result.output
    summary = {
        "rows": len(o),
        "flagged_yes": int((o["flagged"] == "Yes").sum()),
        "justified": int((o["flagged"] == "No (justified)").sum()),
        "not_flagged": int((o["flagged"] == "No").sum()),
        "numeric_anomalies": int(result.audit["numeric_anomaly"].sum()),
        "matched_note_ids": sorted(set(o["matched_note_id"]) - {""}),
    }
    meta = {
        "config": asdict(DEFAULT_CONFIG),
        "retrieval": result.retrieval_info,
        "data_quality": result.data_quality.to_dict(),
        "summary": summary,
        "runtime_seconds": round(elapsed, 2),
        "final_output_sha256": sha256(final),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (out / "cost_report.json").write_text(json.dumps(cost_report(result, elapsed), indent=2) + "\n")
    if not args.quiet:
        print(json.dumps({k: meta[k] for k in ["retrieval", "summary", "final_output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
