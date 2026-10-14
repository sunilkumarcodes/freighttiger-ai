"""End-to-end pipeline: ingest -> weekly metrics -> numeric anomalies -> retrieve -> validate -> explain.

Three separate outcomes are kept per route-week (and all three are in the audit file):
  numeric_anomaly   did the numbers trip the rule?
  evidence          was a note retrieved AND accepted by every validation check?
  flagged           final verdict: Yes / No / No (justified)
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import explanations as ex
from .anomaly_detection import detect_anomalies
from .config import (DEFAULT_CONFIG, FLAG_JUSTIFIED, FLAG_NO, FLAG_YES, NOTES_PATH, OUTPUT_COLUMNS,
                     SHIPMENTS_PATH, Config)
from .evidence_validation import validate
from .ingestion import DataQualityReport, load_notes, load_shipments
from .metrics import compute_route_week_metrics
from .note_parser import NoteCard, build_note_cards
from .retrieval import NoteIndex, anomaly_query, get_embedder

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    output: pd.DataFrame            # exact submission schema
    audit: pd.DataFrame             # full-precision numbers + evidence trail
    cards: dict[str, NoteCard]
    data_quality: DataQualityReport
    retrieval_info: dict = field(default_factory=dict)


def resolve_evidence(row, hits, cards: dict[str, NoteCard], cfg: Config):
    """Validate retrieved candidates in rank order. Returns (accepted, validations, closest_rejected)."""
    validations = [(cards[h.note_id], validate(cards[h.note_id], row.route, row.week_of,
                                               row.own_history_pct, row.peer_pct, cfg), h) for h in hits]
    accepted = next(((c, v) for c, v, _ in validations if v.valid), None)
    # Closest rejected candidate shown to the reviewer: best-ranked note that at least covers this route.
    closest = next(((c, v) for c, v, _ in validations
                    if not v.valid and v.checks[0].passed), None)
    return accepted, validations, closest


def run_pipeline(shipments_path: Path | str = SHIPMENTS_PATH, notes_path: Path | str = NOTES_PATH,
                 cfg: Config = DEFAULT_CONFIG, retriever: str = "semantic", allow_fallback: bool = True,
                 embedder=None) -> RunResult:
    shipments, dq = load_shipments(shipments_path)
    notes = load_notes(notes_path)
    cards = build_note_cards(notes, cfg.open_ended_note_weeks)

    metrics = compute_route_week_metrics(shipments, cfg.history_weeks)
    rows = detect_anomalies(metrics, cfg)

    embedder = embedder or get_embedder(retriever, cfg.embedding_model, allow_fallback)
    index = NoteIndex(notes, embedder)
    anomalous = rows[rows["numeric_anomaly"]]
    queries = [anomaly_query(r.route, r.week_of) for r in anomalous.itertuples()]
    hits_by_row = dict(zip(anomalous.index, index.search_many(queries, cfg.retrieval_top_k)))

    out_records, audit_records = [], []
    for row in rows.itertuples():
        matched, flagged, evidence = "", FLAG_NO, "not_needed"
        retrieved, checks = [], []
        if not row.numeric_anomaly:
            reason = ex.reason_not_flagged(row, cfg)
        else:
            hits = hits_by_row[row.Index]
            accepted, validations, closest = resolve_evidence(row, hits, cards, cfg)
            retrieved = [{"note_id": h.note_id, "score": h.score, "rank": h.rank} for h in hits]
            checks = [{"note_id": v.note_id, "valid": v.valid,
                       "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in v.checks}}
                      for _, v, _ in validations]
            if accepted:
                card, _ = accepted
                matched, flagged, evidence = card.note_id, FLAG_JUSTIFIED, "accepted"
                reason = ex.reason_justified(row, card)
            else:
                flagged, evidence = FLAG_YES, "none_qualified"
                reason = ex.reason_unexplained(row, closest)

        out_records.append({
            "route": row.route,
            "week_of": row.week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": f"{row.cost_per_tonne_km:.2f}",
            "vs_own_history": ex.own_history_text(row.own_history_pct, row.n_prior_weeks),
            "vs_similar_routes": ex.peer_text(row.peer_pct, row.route_type),
            "flagged": flagged,
            "matched_note_id": matched,
            "reason": reason,
        })
        audit_records.append({
            "route": row.route, "route_type": row.route_type, "week_of": row.week_of.strftime("%Y-%m-%d"),
            "n_shipments": row.n_shipments, "total_cost_inr": row.total_cost,
            "total_tonne_km": round(row.total_tonne_km, 6),
            "cost_per_tonne_km": round(row.cost_per_tonne_km, 10),
            "own_history_avg": round(row.own_history_avg, 10), "n_prior_weeks": row.n_prior_weeks,
            "own_history_pct": round(row.own_history_pct, 10),
            "peer_avg": round(row.peer_avg, 10), "n_peers": row.n_peers,
            "peer_pct": round(row.peer_pct, 10),
            "numeric_anomaly": row.numeric_anomaly, "numeric_status": row.numeric_status,
            "anomaly_triggers": row.anomaly_triggers, "evidence": evidence,
            "flagged": flagged, "matched_note_id": matched,
            "retrieved": json.dumps(retrieved), "validations": json.dumps(checks),
        })

    output = pd.DataFrame(out_records, columns=OUTPUT_COLUMNS).sort_values(["route", "week_of"], kind="stable")
    audit = pd.DataFrame(audit_records).sort_values(["route", "week_of"], kind="stable")
    info = {
        "backend": embedder.backend, "model": embedder.model_name, "top_k": cfg.retrieval_top_k,
        "queries": len(queries), "texts_embedded": embedder.texts_embedded,
        "tokens_embedded": embedder.tokens_embedded,
    }
    return RunResult(output.reset_index(drop=True), audit.reset_index(drop=True), cards, dq, info)


def submission_csv_bytes(df: pd.DataFrame) -> bytes:
    """RFC-4180 CSV: fields with commas/quotes are quoted, embedded quotes doubled, LF line ends."""
    assert list(df.columns) == OUTPUT_COLUMNS, f"schema drift: {list(df.columns)}"
    return df.to_csv(index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n").encode("utf-8")


def write_submission_csv(df: pd.DataFrame, path: Path | str) -> None:
    data = submission_csv_bytes(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
