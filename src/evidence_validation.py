"""Deterministic gate between a retrieved note and a "justified" verdict.

Retrieval only *proposes* notes. A note justifies a route-week only if every check
below passes. All checks are always evaluated so the audit trail shows every reason a
note was rejected, not just the first.

  1. scope      note applies to this route's corridor, or nationwide; never "out of scope"
  2. window     note's effective interval overlaps the Monday-Sunday week
  3. causal     note states a cost increase (not "unaffected", not "normal conditions")
  4. magnitude  if the note quantifies its effect (e.g. 5-7%), the observed rise vs own
                history must not exceed upper bound + tolerance: a small real effect does
                not excuse a large unrelated spike
  5. peer_gap   a nationwide note hits peer routes too, so it cannot explain a gap vs
                peers; reject if the row is anomalous vs similar routes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .note_parser import IMPACT_INCREASE, IMPACT_NONE, NoteCard


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str       # full audit text
    short: str = ""   # reviewer-facing phrase used in the CSV reason when the check fails


@dataclass(frozen=True)
class Validation:
    note_id: str
    checks: tuple[Check, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def failure_text(self) -> str:
        return "; ".join(c.short or c.detail for c in self.failures)


def route_endpoints(route: str) -> frozenset[str]:
    return frozenset(p.strip().lower() for p in route.split("-"))


def covers_route(card: NoteCard, route: str) -> bool:
    return card.scope == "all" or (card.scope == "route" and route_endpoints(route) in card.routes)


def _fmt(d: pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def check_scope(card: NoteCard, route: str) -> Check:
    if card.scope == "none":
        return Check("scope", False, f"{card.note_id} says its affected routes are not part of this dataset",
                     "it says its affected routes are not in this dataset")
    if card.scope == "all":
        return Check("scope", True, f"{card.note_id} applies to all routes")
    # Corridor match is direction-agnostic: a flood on Chennai-Bangalore affects both directions.
    if covers_route(card, route):
        return Check("scope", True, f"{card.note_id} applies to {card.applies_to}")
    return Check("scope", False, f"{card.note_id} applies to {card.applies_to}, not {route}",
                 f"it covers {card.applies_to}, not this route")


def check_window(card: NoteCard, week_of: pd.Timestamp) -> Check:
    week_end = week_of + pd.Timedelta(days=6)
    overlaps = card.start <= week_end and card.end >= week_of
    span = f"{_fmt(card.start)} to {_fmt(card.end)}"
    if overlaps:
        return Check("window", True, f"{card.note_id} window {span} covers week {_fmt(week_of)}")
    return Check("window", False,
                 f"{card.note_id} window {span} ({card.window_basis}) does not cover week "
                 f"{_fmt(week_of)} to {_fmt(week_end)}",
                 f"its window ({span}) does not cover this week")


def check_causal(card: NoteCard) -> Check:
    if card.impact == IMPACT_INCREASE:
        return Check("causal", True, f"{card.note_id} states a cost increase ('{card.impact_evidence}')")
    if card.impact == IMPACT_NONE:
        return Check("causal", False,
                     f"{card.note_id} does not describe a cost increase ('{card.impact_evidence}')",
                     f"it does not describe a cost increase ('{card.impact_evidence}')")
    return Check("causal", False, f"{card.note_id} gives no clear evidence of a cost increase",
                 "it gives no clear evidence of a cost increase")


def check_magnitude(card: NoteCard, own_pct: float, cfg: Config) -> Check:
    if card.magnitude_pct is None:
        return Check("magnitude", True, f"{card.note_id} does not quantify its effect")
    lo, hi = card.magnitude_pct
    limit = hi + cfg.magnitude_tolerance_pp
    stated = f"{lo:g}-{hi:g}%" if lo != hi else f"{hi:g}%"
    if np.isfinite(own_pct) and own_pct <= limit:
        return Check("magnitude", True, f"rise of {own_pct:+.1f}% is consistent with the stated ~{stated}")
    return Check("magnitude", False,
                 f"{card.note_id} quantifies its effect at ~{stated}, but cost rose {own_pct:+.1f}% vs "
                 f"own history (limit {limit:g}%)",
                 f"it accounts for ~{stated}, not {own_pct:+.1f}%")


def check_peer_gap(card: NoteCard, peer_pct: float, cfg: Config) -> Check:
    if card.scope != "all":
        return Check("peer_gap", True, "route-specific note; peer gap is expected")
    if np.isfinite(peer_pct) and peer_pct >= cfg.peer_threshold_pct:
        return Check("peer_gap", False,
                     f"{card.note_id} is nationwide, so it also raises similar routes; it cannot explain "
                     f"this route being {peer_pct:+.1f}% above them",
                     "a nationwide change cannot explain a gap vs similar routes")
    return Check("peer_gap", True, "no route-specific gap vs peers left to explain")


def validate(card: NoteCard, route: str, week_of: pd.Timestamp, own_pct: float, peer_pct: float,
             cfg: Config) -> Validation:
    return Validation(card.note_id, (
        check_scope(card, route),
        check_window(card, pd.Timestamp(week_of)),
        check_causal(card),
        check_magnitude(card, own_pct, cfg),
        check_peer_gap(card, peer_pct, cfg),
    ))
