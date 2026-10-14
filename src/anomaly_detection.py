"""Deterministic numeric anomaly rule.

A route-week is a numeric anomaly when
    ΔH > 0                       (cost is rising vs its own history; brief: "rising")
    and (ΔH >= own_threshold     (out of the ordinary vs own history)
         or ΔP >= peer_threshold)  (out of the ordinary vs similar routes)

Decisions use unrounded values. A row with no prior history (ΔH undefined) cannot be
"rising" and is never flagged; a row with no peers that week can still be flagged on ΔH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TRIGGER_OWN = "own_history"
TRIGGER_PEER = "similar_routes"


def classify_numeric(own_pct: float, peer_pct: float, cfg: Config) -> tuple[bool, list[str], str]:
    """Return (is_anomaly, triggers, status) for one route-week."""
    if own_pct is None or not np.isfinite(own_pct):
        return False, [], "no_history"
    if cfg.require_rising and own_pct <= 0:
        return False, [], "not_rising"
    triggers = []
    if own_pct >= cfg.own_history_threshold_pct:
        triggers.append(TRIGGER_OWN)
    if peer_pct is not None and np.isfinite(peer_pct) and peer_pct >= cfg.peer_threshold_pct:
        triggers.append(TRIGGER_PEER)
    return (bool(triggers), triggers, "anomaly" if triggers else "within_thresholds")


def detect_anomalies(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = metrics.copy()
    results = [classify_numeric(o, p, cfg) for o, p in zip(out["own_history_pct"], out["peer_pct"])]
    out["numeric_anomaly"] = [r[0] for r in results]
    out["anomaly_triggers"] = ["+".join(r[1]) for r in results]
    out["numeric_status"] = [r[2] for r in results]
    return out
