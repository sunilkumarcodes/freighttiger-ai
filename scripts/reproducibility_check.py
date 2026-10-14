"""Run the full pipeline 3 times in fresh processes and prove the outputs are identical.

    python scripts/reproducibility_check.py [--retriever semantic|tfidf]

Writes outputs/repro/run_{1,2,3}/ and outputs/reproducibility_report.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "outputs" / "repro"
FILES = ["final_output.csv", "route_week_audit.csv"]
KEY = ["route", "week_of"]
GRADED = ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_sha256(p: Path) -> str:
    """Hash with CRLF normalised to LF, so a Windows/git checkout of the same data hashes the same."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["semantic", "tfidf"], default="semantic")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--strict-semantic", action="store_true", help="error if the embedding model can't load")
    args = ap.parse_args(argv)

    dirs = []
    for i in range(1, args.runs + 1):
        d = REPRO / f"run_{i}"
        cmd = [sys.executable, str(ROOT / "run.py"), "--retriever", args.retriever, "--out-dir", str(d), "--quiet"]
        if args.strict_semantic:
            cmd.append("--strict-semantic")
        subprocess.run(cmd, check=True, cwd=ROOT)
        dirs.append(d)

    lines = [f"Reproducibility check - {args.runs} independent runs (separate processes), unchanged inputs",
             f"generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", ""]
    backends = {json.loads((d / "run_metadata.json").read_text())["retrieval"]["backend"] for d in dirs}
    lines.append(f"retrieval backend(s) used: {sorted(backends)}")
    inputs = [ROOT / "data" / "shipment_records.csv", ROOT / "data" / "context_notes.csv"]
    lines += [f"input  {p.name}: {content_sha256(p)}  (content hash, line endings normalised)" for p in inputs]
    lines.append("")

    all_ok = True
    for f in FILES:
        hashes = [sha256(d / f) for d in dirs]
        same = len(set(hashes)) == 1
        all_ok &= same
        lines.append(f"{f}:")
        lines += [f"  run_{i}  {h}" for i, h in enumerate(hashes, start=1)]
        lines.append(f"  byte-identical across runs: {same}")

    frames = [pd.read_csv(d / "final_output.csv", dtype=str, keep_default_na=False) for d in dirs]
    base = frames[0].set_index(KEY)
    lines.append("")
    lines.append("Field-level comparison vs run_1 (final_output.csv):")
    for i, df in enumerate(frames[1:], start=2):
        other = df.set_index(KEY)
        same_keys = base.index.equals(other.index)
        diffs = {c: int((base[c] != other[c]).sum()) for c in base.columns} if same_keys else {}
        all_ok &= same_keys and not any(diffs.values())
        lines.append(f"  run_{i}: same route-week rows: {same_keys}; differing cells per column: {diffs}")

    summary = frames[0]["flagged"].value_counts().to_dict()
    lines += ["", f"rows: {len(frames[0])}; verdicts: {summary}",
              f"graded fields checked: {GRADED + ['reason']}", "",
              f"RESULT: {'IDENTICAL' if all_ok else 'MISMATCH'}"]
    report = "\n".join(lines) + "\n"
    (ROOT / "outputs" / "reproducibility_report.txt").write_text(report)
    print(report)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
