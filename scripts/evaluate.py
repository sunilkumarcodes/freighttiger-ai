"""Score the system against hand-labelled sets and write outputs/evaluation_report.{json,md}.

    python scripts/evaluate.py                     # semantic retriever (TF-IDF fallback if unavailable)
    python scripts/evaluate.py --retriever tfidf

Labelled sets live in eval/. They were written by the author from a manual reading of the
notes; see README "Evaluation" for what that does and does not prove.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly_detection import classify_numeric  # noqa: E402
from src.config import DEFAULT_CONFIG, NOTES_PATH, OUTPUT_DIR  # noqa: E402
from src.evidence_validation import route_endpoints, validate  # noqa: E402
from src.ingestion import load_notes  # noqa: E402
from src.note_parser import build_note_card, build_note_cards, classify_impact  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.retrieval import NoteIndex, TfidfEmbedder, anomaly_query, get_embedder  # noqa: E402

EVAL = ROOT / "eval"
CFG = DEFAULT_CONFIG


def binary_scores(gold: list[bool], pred: list[bool]) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / len(gold) if gold else None, "n": len(gold)}


def eval_pair_validator(cards) -> dict:
    df = pd.read_csv(EVAL / "labeled_note_pairs.csv")
    gold, pred, failures, by_cat = [], [], [], Counter()
    for r in df.itertuples():
        res = validate(cards[r.note_id], r.route, pd.Timestamp(r.week_of), r.own_pct, r.peer_pct, CFG)
        g, p = r.label == "valid", res.valid
        gold.append(g); pred.append(p)
        by_cat[(r.category, g == p)] += 1
        if g != p:
            failures.append({"route": r.route, "week_of": r.week_of, "note_id": r.note_id, "gold": r.label,
                             "predicted": "valid" if p else "invalid", "why": res.failure_text()})
    cats = sorted({c for c, _ in by_cat})
    return {
        "description": "Does the validator accept exactly the (route-week, note) pairs a careful reader would?",
        "positive_class": "valid (note justifies the route-week)",
        **binary_scores(gold, pred),
        "by_category": {c: {"correct": by_cat[(c, True)], "wrong": by_cat[(c, False)]} for c in cats},
        "errors": failures,
    }


def eval_note_parser(cards) -> dict:
    gold = pd.read_csv(EVAL / "supplied_note_labels.csv", dtype=str, keep_default_na=False)
    rows, correct = [], Counter()
    for g in gold.itertuples():
        c = cards[g.note_id]
        mag = f"{c.magnitude_pct[0]:g}-{c.magnitude_pct[1]:g}" if c.magnitude_pct else ""
        fields = {"scope": (c.scope, g.gold_scope), "impact": (c.impact, g.gold_impact),
                  "start": (f"{c.start:%Y-%m-%d}", g.gold_start), "end": (f"{c.end:%Y-%m-%d}", g.gold_end),
                  "magnitude": (mag, g.gold_magnitude)}
        for f, (p, t) in fields.items():
            correct[f] += p == t
        rows.append({"note_id": g.note_id, **{f: {"parsed": p, "gold": t, "ok": p == t}
                                             for f, (p, t) in fields.items()}})
    n = len(gold)
    return {"description": "Structured fields parsed from the 10 supplied notes vs manual labels "
                           "(in-sample: the parser was written with these notes in view).",
            "field_accuracy": {f: f"{k}/{n}" for f, k in correct.items()},
            "notes": rows}


def eval_impact_heldout() -> dict:
    df = pd.read_csv(EVAL / "heldout_impact_notes.csv", keep_default_na=False)
    labels = ["increase", "no_increase", "unknown"]
    matrix = {g: {p: 0 for p in labels} for g in labels}
    errors = []
    for r in df.itertuples():
        pred, cue = classify_impact(r.note)
        matrix[r.gold_impact][pred] += 1
        if pred != r.gold_impact:
            errors.append({"note_id": r.note_id, "note": r.note, "gold": r.gold_impact,
                           "predicted": pred, "matched_cue": cue,
                           "consequence": ("UNSAFE: would let this note justify a flag"
                                           if pred == "increase" else "safe: note rejected (missed justification)"
                                           if r.gold_impact == "increase" else "safe: still rejected")})
    gold_inc = [g == "increase" for g in df["gold_impact"]]
    pred_inc = [classify_impact(n)[0] == "increase" for n in df["note"]]
    return {"description": "Impact classifier on 24 notes NOT in the dataset, written to probe phrasing the "
                           "supplied notes do not use. Same author as the rules, so this is a robustness probe, "
                           "not an unbiased accuracy estimate.",
            "confusion_matrix_gold_by_pred": matrix,
            "increase_vs_rest": binary_scores(gold_inc, pred_inc),
            "errors": errors}


def eval_retrieval(notes, run_output, backends: list) -> dict:
    gold = pd.read_csv(EVAL / "gold_verdicts.csv", keep_default_na=False)
    cards = build_note_cards(notes)
    result = {}
    for embedder in backends:
        index = NoteIndex(notes, embedder)
        just, route_specific = [], []
        for g in gold.itertuples():
            hits = index.search(anomaly_query(g.route, pd.Timestamp(g.week_of)), len(notes))
            ranks = {h.note_id: h.rank for h in hits}
            if g.gold_note_id:
                just.append(ranks[g.gold_note_id])
            # Notes written about this exact corridor are the topically relevant ones.
            for nid, c in cards.items():
                if c.scope == "route" and route_endpoints(g.route) in c.routes:
                    route_specific.append(ranks[nid])
        k = CFG.retrieval_top_k
        result[embedder.backend] = {
            "model": embedder.model_name,
            "gold_justification_ranks": just,
            f"gold_justification_recall@{k}": sum(r <= k for r in just) / len(just),
            "gold_justification_recall@1": sum(r == 1 for r in just) / len(just),
            "route_specific_note_pairs": len(route_specific),
            f"route_specific_recall@{k}": sum(r <= k for r in route_specific) / len(route_specific),
            "route_specific_mrr": sum(1 / r for r in route_specific) / len(route_specific),
        }
    return {"description": "Do the right notes reach the validator? Queries = the 19 numeric anomalies.",
            "top_k_used_in_pipeline": CFG.retrieval_top_k, "backends": result}


def eval_end_to_end(run) -> dict:
    gold = pd.read_csv(EVAL / "gold_verdicts.csv", keep_default_na=False)
    out = run.output.set_index(["route", "week_of"])
    rows, ok_flag, ok_note = [], 0, 0
    for g in gold.itertuples():
        mine = out.loc[(g.route, g.week_of)]
        f_ok, n_ok = mine["flagged"] == g.gold_flagged, mine["matched_note_id"] == g.gold_note_id
        ok_flag += f_ok; ok_note += n_ok
        rows.append({"route": g.route, "week_of": g.week_of, "gold": g.gold_flagged, "pred": mine["flagged"],
                     "gold_note": g.gold_note_id, "pred_note": mine["matched_note_id"], "ok": bool(f_ok and n_ok)})
    gold_keys = set(zip(gold.route, gold.week_of))
    o = run.output
    extra = o[(o["flagged"] != "No") & ~o.set_index(["route", "week_of"]).index.isin(gold_keys)]
    justified = o[o["flagged"] == "No (justified)"]
    gold_just = {(r.route, r.week_of): r.gold_note_id for r in gold.itertuples() if r.gold_note_id}
    fabricated = [(r.route, r.week_of, r.matched_note_id) for r in justified.itertuples()
                  if gold_just.get((r.route, r.week_of)) != r.matched_note_id]
    return {"description": "Final verdicts vs gold for all numeric anomalies.",
            "anomalies_scored": len(gold), "flag_correct": ok_flag, "note_id_correct": ok_note,
            "rows_flagged_or_justified_outside_gold": len(extra),
            "justified_rows": len(justified), "fabricated_justifications": fabricated, "rows": rows}


def threshold_sensitivity(run) -> dict:
    a = run.audit
    grid = []
    for own in [10, 15, 20, 25]:
        for peer in [15, 20, 25, 30]:
            cfg = replace(CFG, own_history_threshold_pct=own, peer_threshold_pct=peer)
            flags = [classify_numeric(o, p, cfg)[0] for o, p in zip(a["own_history_pct"], a["peer_pct"])]
            sel = a[flags]
            grid.append({"own_pct": own, "peer_pct": peer, "numeric_anomalies": int(sum(flags)),
                         "by_route": sel["route"].value_counts().sort_index().to_dict()})
    dh, dp = a["own_history_pct"].dropna(), a["peer_pct"].dropna()
    q = [0.5, 0.9, 0.95, 0.99]
    return {"description": "How many route-weeks trip the numeric rule under other thresholds.",
            "distribution": {"own_history_pct_quantiles": {str(k): round(dh.quantile(k), 2) for k in q},
                             "peer_pct_quantiles": {str(k): round(dp.quantile(k), 2) for k in q},
                             "own_history_pct_std": round(dh.std(), 2)},
            "grid": grid}


def to_markdown(rep: dict) -> str:
    pv, ih, e2e = rep["pair_validator"], rep["impact_heldout"], rep["end_to_end"]
    L = ["# Evaluation report", "", f"Retrieval backend for end-to-end run: `{rep['pipeline_backend']}`", "",
         "## 1. Evidence validator on labelled (route-week, note) pairs",
         f"n = {pv['n']} | TP {pv['tp']} FP {pv['fp']} FN {pv['fn']} TN {pv['tn']} | "
         f"precision {pv['precision']} | recall {pv['recall']}", "",
         "| category | correct | wrong |", "|---|---|---|"]
    L += [f"| {c} | {v['correct']} | {v['wrong']} |" for c, v in pv["by_category"].items()]
    L += ["", f"Errors: {pv['errors'] or 'none'}", "",
          "## 2. Note parser on the 10 supplied notes (in-sample)",
          ", ".join(f"{k}: {v}" for k, v in rep["note_parser"]["field_accuracy"].items()), "",
          "## 3. Impact classifier on 24 held-out notes", "",
          "| gold \\ predicted | increase | no_increase | unknown |", "|---|---|---|---|"]
    L += [f"| {g} | " + " | ".join(str(v) for v in row.values()) + " |"
          for g, row in ih["confusion_matrix_gold_by_pred"].items()]
    s = ih["increase_vs_rest"]
    fp_ = rep.get("impact_heldout_first_pass", {}).get("increase_vs_rest", {})
    L += ["", f"'increase' vs rest (current lexicon): precision {s['precision']}, recall {s['recall']}, "
              f"accuracy {s['accuracy']:.3f}",
          "",
          f"**First pass, before the lexicon was revised (the only genuinely held-out result):** "
          f"precision {fp_.get('precision')}, recall {fp_.get('recall')}, 3-class errors "
          f"{len(rep.get('impact_heldout_first_pass', {}).get('errors', []))}/24, all failing safe "
          f"(note rejected). The current numbers are in-sample for these 24 notes.", "", "Errors:"]
    L += [f"- {x['note_id']} \"{x['note']}\" gold={x['gold']} pred={x['predicted']} -> {x['consequence']}"
          for x in ih["errors"]] or ["- none"]
    L += ["", "## 4. Retrieval", "", "| backend | gold note recall@k | gold recall@1 | route-specific recall@k | MRR |",
          "|---|---|---|---|---|"]
    k = rep["retrieval"]["top_k_used_in_pipeline"]
    for b, v in rep["retrieval"]["backends"].items():
        L.append(f"| {b} | {v[f'gold_justification_recall@{k}']:.2f} | {v['gold_justification_recall@1']:.2f} | "
                 f"{v[f'route_specific_recall@{k}']:.2f} | {v['route_specific_mrr']:.3f} |")
    L += ["", "## 5. End-to-end verdicts",
          f"{e2e['flag_correct']}/{e2e['anomalies_scored']} verdicts correct, "
          f"{e2e['note_id_correct']}/{e2e['anomalies_scored']} matched_note_id correct, "
          f"{e2e['rows_flagged_or_justified_outside_gold']} unexpected flags outside the gold set, "
          f"fabricated justifications: {e2e['fabricated_justifications'] or 'none'}", "",
          "## 6. Threshold sensitivity (numeric anomalies out of 728 route-weeks)", "",
          "| own \\ peer | " + " | ".join(str(p) for p in [15, 20, 25, 30]) + " |", "|---|---|---|---|---|"]
    grid = rep["threshold_sensitivity"]["grid"]
    for own in [10, 15, 20, 25]:
        L.append(f"| {own} | " + " | ".join(str(g["numeric_anomalies"]) for g in grid if g["own_pct"] == own) + " |")
    L += ["", f"Distribution: {json.dumps(rep['threshold_sensitivity']['distribution'])}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--strict-semantic", action="store_true", help="error if the embedding model can't load")
    args = p.parse_args(argv)

    notes = load_notes(NOTES_PATH)
    cards = build_note_cards(notes, CFG.open_ended_note_weeks)
    primary = get_embedder(args.retriever, CFG.embedding_model, allow_fallback=not args.strict_semantic)
    run = run_pipeline(cfg=CFG, embedder=primary)
    backends = [primary] if primary.backend == "tfidf" else [primary, TfidfEmbedder()]

    rep = {
        "pipeline_backend": run.retrieval_info["backend"],
        "pair_validator": eval_pair_validator(cards),
        "note_parser": eval_note_parser(cards),
        "impact_heldout": eval_impact_heldout(),
        "impact_heldout_first_pass": json.loads((EVAL / "heldout_first_pass_result.json").read_text()),
        "retrieval": eval_retrieval(notes, run.output, backends),
        "end_to_end": eval_end_to_end(run),
        "threshold_sensitivity": threshold_sensitivity(run),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "evaluation_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    (OUTPUT_DIR / "evaluation_report.md").write_text(to_markdown(rep))
    print(to_markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
