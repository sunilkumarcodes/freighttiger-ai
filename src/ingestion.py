"""Load and validate the two input files.

Invalid shipment rows are dropped, never repaired, and every drop is recorded in a
data-quality report so the numbers can be audited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SHIPMENT_COLUMNS = [
    "shipment_id", "origin", "destination", "route_type", "material",
    "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date", "transporter",
]
NOTE_COLUMNS = ["note_id", "date", "applies_to", "note"]
VALID_ROUTE_TYPES = {"Short", "Medium", "Long"}


class SchemaError(ValueError):
    """Raised when an input file is missing required columns."""


@dataclass
class DataQualityReport:
    rows_read: int = 0
    rows_kept: int = 0
    dropped: dict[str, list[str]] = field(default_factory=dict)  # reason -> shipment_ids
    warnings: list[str] = field(default_factory=list)

    def drop(self, reason: str, ids) -> None:
        ids = [str(i) for i in ids]
        if ids:
            self.dropped.setdefault(reason, []).extend(ids)

    def to_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_dropped": sum(len(v) for v in self.dropped.values()),
            "dropped_by_reason": {k: {"count": len(v), "shipment_ids": v[:50]} for k, v in self.dropped.items()},
            "warnings": self.warnings,
        }


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing required columns: {missing}")


def load_shipments(path: Path | str) -> tuple[pd.DataFrame, DataQualityReport]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    return clean_shipments(raw)


def clean_shipments(raw: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Type-cast and validate shipment rows. Accepts an all-string frame."""
    _require_columns(raw, SHIPMENT_COLUMNS, "shipment_records")
    report = DataQualityReport(rows_read=len(raw))
    df = raw[SHIPMENT_COLUMNS].copy()

    for col in ["shipment_id", "origin", "destination", "route_type", "material", "transporter"]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    # ISO dates, optionally with a time component ("2024-01-01 10:30:00"); time is dropped.
    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"].astype(str).str.strip(), format="ISO8601", errors="coerce"
    ).dt.normalize()

    checks = [
        ("missing_origin_or_destination", (df["origin"] == "") | (df["destination"] == "")),
        ("invalid_route_type", ~df["route_type"].isin(VALID_ROUTE_TYPES)),
        ("invalid_shipment_date", df["shipment_date"].isna()),
        ("non_numeric_or_missing_value",
         df[["quantity_tonnes", "distance_km", "freight_cost_inr"]].isna().any(axis=1)),
        # tonne-km is the denominator; it must be strictly positive.
        ("non_positive_quantity_or_distance", (df["quantity_tonnes"] <= 0) | (df["distance_km"] <= 0)),
        # A zero or negative freight bill is a data error, not a cheap shipment.
        ("non_positive_freight_cost", df["freight_cost_inr"] <= 0),
    ]
    bad = pd.Series(False, index=df.index)
    for reason, mask in checks:
        mask = mask.fillna(True) & ~bad  # attribute each row to its first failing check
        report.drop(reason, df.loc[mask, "shipment_id"])
        bad |= mask

    dup = df["shipment_id"].duplicated(keep="first") & ~bad
    report.drop("duplicate_shipment_id", df.loc[dup, "shipment_id"])
    bad |= dup

    df = df.loc[~bad].copy()
    if df.empty:
        raise ValueError(f"No valid shipment rows after validation: {report.to_dict()['dropped_by_reason']}")
    df["route"] = df["origin"] + "-" + df["destination"]

    types_per_route = df.groupby("route")["route_type"].nunique()
    for route in types_per_route[types_per_route > 1].index:
        report.warnings.append(
            f"Route {route} appears with several route_types; each (route, route_type) is tracked separately."
        )

    report.rows_kept = len(df)
    for reason, ids in report.dropped.items():
        log.warning("Dropped %d shipment rows: %s", len(ids), reason)
    return df.reset_index(drop=True), report


def load_notes(path: Path | str) -> pd.DataFrame:
    notes = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(notes, NOTE_COLUMNS, "context_notes")
    notes = notes[NOTE_COLUMNS].apply(lambda s: s.str.strip())
    notes["date"] = pd.to_datetime(notes["date"], format="%Y-%m-%d", errors="raise")
    if notes["note_id"].duplicated().any():
        raise SchemaError("context_notes has duplicate note_id values")
    return notes.sort_values("note_id").reset_index(drop=True)
