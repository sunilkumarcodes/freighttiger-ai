"""Central configuration. Every tunable policy lives here so it is visible and testable.

Nothing in this file is an official requirement of the brief unless marked [BRIEF].
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SHIPMENTS_PATH = DATA_DIR / "shipment_records.csv"
NOTES_PATH = DATA_DIR / "context_notes.csv"
SAMPLE_OUTPUT_PATH = DATA_DIR / "sample_output_format_v2.csv"

# [BRIEF] Output contract: exact names and order (taken from sample_output_format_v2.csv).
OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

FLAG_YES = "Yes"
FLAG_NO = "No"
FLAG_JUSTIFIED = "No (justified)"


@dataclass(frozen=True)
class Config:
    # [BRIEF] trailing window length, strictly-prior weeks only.
    history_weeks: int = 8

    # Flagging policy (ours, not the brief's). A route-week is a numeric anomaly when
    # it is rising vs its own history AND out of the ordinary on at least one baseline.
    own_history_threshold_pct: float = 15.0
    peer_threshold_pct: float = 20.0
    require_rising: bool = True  # ΔH must be > 0

    # Evidence policy.
    retrieval_top_k: int = 5
    # An open-ended note ("starting this week") is honoured for this many weeks:
    # after `history_weeks` the shift is inside the own-history baseline anyway.
    open_ended_note_weeks: int = 8
    # A note that quantifies its impact (e.g. "roughly 5-7%") cannot excuse an increase
    # larger than its upper bound plus this tolerance, in percentage points.
    magnitude_tolerance_pp: float = 3.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


DEFAULT_CONFIG = Config()
