"""Deterministic, template-based explanations.

Every factual phrase is either a computed number or a verbatim quote of the note, so a
reason can't say anything the data and notes don't. No LLM is involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly_detection import TRIGGER_OWN, TRIGGER_PEER
from .config import Config
from .evidence_validation import Validation
from .note_parser import NoteCard


def fmt_pct(v: float) -> str:
    v = round(float(v), 1)
    return f"{v + 0.0:+.1f}%"  # + 0.0 turns -0.0 into 0.0


def own_history_text(own_pct: float, n_prior: int) -> str:
    if not np.isfinite(own_pct):
        return "n/a (no prior weeks for this route)"
    return f"{fmt_pct(own_pct)} vs this route's past average"


def peer_text(peer_pct: float, route_type: str) -> str:
    if not np.isfinite(peer_pct):
        return f"n/a (no other {route_type} routes this week)"
    return f"{fmt_pct(peer_pct)} vs similar-length routes this week"


def _numbers(row) -> str:
    parts = []
    if np.isfinite(row.own_history_pct):
        n = row.n_prior_weeks
        basis = "" if n >= 8 else f" (only {n} prior week{'s' if n != 1 else ''} available)"
        parts.append(f"{fmt_pct(row.own_history_pct)} vs its past average{basis}")
    if np.isfinite(row.peer_pct):
        parts.append(f"{fmt_pct(row.peer_pct)} vs similar-length routes")
    return " and ".join(parts)


def reason_not_flagged(row, cfg: Config) -> str:
    if row.numeric_status == "no_history":
        return "Not flagged: first week on record for this route, so there is no past average to compare against."
    if row.numeric_status == "not_rising":
        return f"Not flagged: cost is not rising ({fmt_pct(row.own_history_pct)} vs this route's past average)."
    return (f"Not flagged: cost is {_numbers(row)}, within the review thresholds "
            f"({cfg.own_history_threshold_pct:g}% vs own history, {cfg.peer_threshold_pct:g}% vs similar routes).")


def reason_justified(row, card: NoteCard) -> str:
    window = f"{card.start:%Y-%m-%d} to {card.end:%Y-%m-%d}"
    return (f"Justified by {card.note_id} ({card.date:%Y-%m-%d}, {card.applies_to}): \"{card.summary}\" "
            f"This week falls within the note's window ({window}), which explains the rise "
            f"({_numbers(row)}).")


def reason_unexplained(row, closest: tuple[NoteCard, Validation] | None) -> str:
    trig = {TRIGGER_OWN: "own history", TRIGGER_PEER: "similar-length routes"}
    triggered = " and ".join(trig[t] for t in row.anomaly_triggers.split("+") if t)
    text = f"Unexplained: cost is {_numbers(row)} (out of the ordinary vs {triggered}). "
    if closest is None:
        return text + "No note applies to this route and week. Flagged for human review."
    card, val = closest
    return text + (f"Closest note {card.note_id} ({card.date:%Y-%m-%d}) does not qualify: {val.failure_text()}. "
                   f"Flagged for human review.")
