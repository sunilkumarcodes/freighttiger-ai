"""Output contract, CSV escaping, and end-to-end verdicts on the supplied data."""
import csv
import io
from dataclasses import replace

import pandas as pd
import pytest

from src.config import (DEFAULT_CONFIG, FLAG_JUSTIFIED, FLAG_NO, FLAG_YES, OUTPUT_COLUMNS,
                        SAMPLE_OUTPUT_PATH)
from src.pipeline import run_pipeline, write_submission_csv

# Gold verdicts for every numeric anomaly, written from a manual reading of the notes before
# the validator was implemented (see eval/gold_verdicts.csv for the rationale per row).
EXPECTED_JUSTIFIED = {
    ("Ahmedabad-Mumbai", "2025-01-20"): "N002",
    ("Chennai-Bangalore", "2025-02-24"): "N001",
    ("Chennai-Bangalore", "2025-03-03"): "N001",
}


# 13. Exact columns and order ----------------------------------------------------------
def test_columns_match_sample_header(tfidf_run):
    with open(SAMPLE_OUTPUT_PATH, newline="") as f:
        header = next(csv.reader(f))
    assert header == OUTPUT_COLUMNS
    assert list(tfidf_run.output.columns) == header


def test_one_row_per_route_week(tfidf_run):
    o = tfidf_run.output
    assert len(o) == 728
    assert not o.duplicated(["route", "week_of"]).any()
    assert set(o["flagged"]) <= {FLAG_YES, FLAG_NO, FLAG_JUSTIFIED}
    assert (pd.to_datetime(o["week_of"]).dt.weekday == 0).all()


# 14. matched_note_id discipline -------------------------------------------------------
def test_matched_note_id_blank_unless_justified(tfidf_run):
    o = tfidf_run.output
    assert (o.loc[o["flagged"] != FLAG_JUSTIFIED, "matched_note_id"] == "").all()
    just = o[o["flagged"] == FLAG_JUSTIFIED]
    assert (just["matched_note_id"] != "").all()
    for r in just.itertuples():
        assert r.matched_note_id in r.reason  # cited in prose too


def test_justified_rows_equal_gold(tfidf_run):
    o = tfidf_run.output
    got = {(r.route, r.week_of): r.matched_note_id for r in o[o["flagged"] == FLAG_JUSTIFIED].itertuples()}
    assert got == EXPECTED_JUSTIFIED


def test_flood_aftermath_weeks_are_not_excused(tfidf_run):
    o = tfidf_run.output.set_index(["route", "week_of"])
    for week in ["2025-03-10", "2025-03-17"]:
        assert o.loc[("Chennai-Bangalore", week), "flagged"] == FLAG_YES
        assert o.loc[("Chennai-Bangalore", week), "matched_note_id"] == ""


def test_verdicts_do_not_depend_on_retrieval_depth():
    """With k = all notes, validation sees every candidate; verdicts must be unchanged."""
    a = run_pipeline(retriever="tfidf").output
    b = run_pipeline(retriever="tfidf", cfg=replace(DEFAULT_CONFIG, retrieval_top_k=10)).output
    cols = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
            "flagged", "matched_note_id"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


# 15. CSV quoting and re-parsing ---------------------------------------------------------
def test_csv_round_trip(tfidf_run, tmp_path):
    path = tmp_path / "out.csv"
    write_submission_csv(tfidf_run.output, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == OUTPUT_COLUMNS
    assert all(len(r) == 8 for r in rows)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(back, tfidf_run.output.reset_index(drop=True))


def test_csv_escapes_commas_quotes_and_newlines(tmp_path):
    nasty = 'The closest note (N006, 2025-09-22) says "stable"\nsecond line'
    df = pd.DataFrame([["A-B", "2025-01-06", "1.00", "+1.0% x", "+1.0% y", "Yes", "", nasty]],
                      columns=OUTPUT_COLUMNS)
    path = tmp_path / "x.csv"
    write_submission_csv(df, path)
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert rows[1][7] == nasty and len(rows[1]) == 8


def test_writer_refuses_schema_drift(tmp_path):
    df = pd.DataFrame([[1, 2]], columns=["route", "week_of"])
    with pytest.raises(AssertionError):
        write_submission_csv(df, tmp_path / "x.csv")


def test_no_notes_means_every_anomaly_unexplained(tmp_path):
    from src.config import SHIPMENTS_PATH
    empty = tmp_path / "notes.csv"
    empty.write_text("note_id,date,applies_to,note\n")
    o = run_pipeline(SHIPMENTS_PATH, empty, retriever="tfidf").output
    assert (o["flagged"] != FLAG_JUSTIFIED).all() and (o["matched_note_id"] == "").all()
    assert (o["flagged"] == FLAG_YES).sum() == 19
