"""Metric tests against hand-calculated values (expected numbers are written out, not recomputed)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.ingestion import SchemaError, clean_shipments
from src.metrics import add_own_history, add_peer_baseline, compute_route_week_metrics, week_start, weekly_costs
from tests.conftest import make_shipments


# 1. Monday-based weeks ------------------------------------------------------------------
@pytest.mark.parametrize("date, monday", [
    ("2024-01-01", "2024-01-01"),  # Monday maps to itself
    ("2024-01-07", "2024-01-01"),  # Sunday belongs to the preceding Monday
    ("2024-01-08", "2024-01-08"),
    ("2024-02-29", "2024-02-26"),  # leap day (Thursday)
    ("2025-01-01", "2024-12-30"),  # week straddles the year boundary
    ("2025-03-09", "2025-03-03"),  # Sunday
])
def test_week_start_is_monday(date, monday):
    assert week_start(pd.Series([pd.Timestamp(date)]))[0] == pd.Timestamp(monday)


# 2. Weighted aggregation -----------------------------------------------------------------
def test_weekly_cost_is_ratio_of_sums_not_mean_of_ratios():
    # s1: 1000 / (10 t * 10 km)  = 10.0 per t-km ; s2: 3000 / (5 t * 100 km) = 6.0 per t-km
    # weighted = 4000 / (100 + 500) = 6.6667 ; naive mean of ratios would be 8.0
    df = make_shipments([
        ("A-B", "Short", "2024-01-01", 10, 10, 1000),
        ("A-B", "Short", "2024-01-07", 5, 100, 3000),  # Sunday: same week
    ])
    w = weekly_costs(df)
    assert len(w) == 1
    assert w.loc[0, "cost_per_tonne_km"] == pytest.approx(4000 / 600)
    assert w.loc[0, "cost_per_tonne_km"] != pytest.approx(8.0)
    assert w.loc[0, "n_shipments"] == 2


def _ten_week_route(values, start="2024-01-01", skip=()):
    rows = []
    for k, v in enumerate(values):
        if k in skip:
            continue
        date = (pd.Timestamp(start) + pd.Timedelta(weeks=k)).strftime("%Y-%m-%d")
        rows.append(("A-B", "Short", date, 1, 1, v))  # 1 t * 1 km, so C == cost
    return add_own_history(weekly_costs(make_shipments(rows)), 8)


# 3 & 4. Strictly-prior 8-week history ---------------------------------------------------
def test_history_uses_strictly_prior_eight_weeks():
    h = _ten_week_route([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).set_index("week_of")
    w = lambda k: pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=k)
    assert math.isnan(h.loc[w(0), "own_history_avg"])          # no prior weeks
    assert h.loc[w(0), "n_prior_weeks"] == 0
    assert h.loc[w(2), "own_history_avg"] == pytest.approx(1.5)  # mean(1,2): fewer than 8 available
    assert h.loc[w(2), "n_prior_weeks"] == 2
    assert h.loc[w(8), "own_history_avg"] == pytest.approx(4.5)  # mean(1..8), excludes current (9)
    assert h.loc[w(9), "own_history_avg"] == pytest.approx(5.5)  # mean(2..9), week 1 has rolled off
    assert h.loc[w(9), "n_prior_weeks"] == 8
    # ΔH for the last week: 100 * (10 - 5.5) / 5.5
    assert h.loc[w(9), "own_history_pct"] == pytest.approx(100 * 4.5 / 5.5)


def test_history_has_no_look_ahead():
    base = _ten_week_route([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    shocked = _ten_week_route([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])  # change only the last week
    pd.testing.assert_series_equal(base["own_history_avg"], shocked["own_history_avg"])


def test_missing_weeks_are_skipped_not_imputed():
    # Week index 4 (value 5) has no shipments. For week 9 the calendar window is weeks 1..8;
    # observed values there are 2,3,4,6,7,8,9 -> mean 39/7 over 7 weeks (no zero, no fill).
    h = _ten_week_route([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], skip={4}).set_index("week_of")
    last = h.loc[pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=9)]
    assert last["own_history_avg"] == pytest.approx(39 / 7)
    assert last["n_prior_weeks"] == 7


# 5. Peer baseline -------------------------------------------------------------------------
def test_peer_average_excludes_self_and_other_types():
    df = make_shipments([
        ("A-B", "Medium", "2024-01-01", 1, 1, 2),
        ("C-D", "Medium", "2024-01-02", 1, 1, 4),
        ("E-F", "Medium", "2024-01-03", 1, 1, 9),
        ("G-H", "Short", "2024-01-03", 1, 1, 100),  # different type: must not leak in
        ("I-J", "Long", "2024-01-03", 1, 1, 5),     # only Long route: no peers
    ])
    p = add_peer_baseline(weekly_costs(df)).set_index("route")
    assert p.loc["A-B", "peer_avg"] == pytest.approx(6.5)       # (4 + 9) / 2
    assert p.loc["C-D", "peer_avg"] == pytest.approx(5.5)       # (2 + 9) / 2
    assert p.loc["E-F", "peer_avg"] == pytest.approx(3.0)       # (2 + 4) / 2
    assert p.loc["A-B", "peer_pct"] == pytest.approx(100 * (2 / 6.5 - 1))
    assert p.loc["A-B", "n_peers"] == 2
    assert math.isnan(p.loc["I-J", "peer_avg"]) and p.loc["I-J", "n_peers"] == 0
    assert math.isnan(p.loc["I-J", "peer_pct"])


def test_peer_average_is_unweighted_mean_of_route_costs():
    # Route C-D ships 100x more tonne-km than E-F; peer mean for A-B must still be (4 + 9) / 2.
    df = make_shipments([
        ("A-B", "Medium", "2024-01-01", 1, 1, 2),
        ("C-D", "Medium", "2024-01-01", 100, 1, 400),
        ("E-F", "Medium", "2024-01-01", 1, 1, 9),
    ])
    p = add_peer_baseline(weekly_costs(df)).set_index("route")
    assert p.loc["A-B", "peer_avg"] == pytest.approx(6.5)


# 6. Invalid input values -----------------------------------------------------------------
def test_invalid_rows_are_dropped_and_reported():
    good = {"origin": "A", "destination": "B", "route_type": "Short", "material": "x",
            "quantity_tonnes": "10", "distance_km": "100", "freight_cost_inr": "5000",
            "shipment_date": "2024-01-01", "transporter": "T"}
    variants = {
        "ok1": {}, "ok2": {"freight_cost_inr": " 6000 "},
        "bad_qty_text": {"quantity_tonnes": "ten"},
        "bad_qty_zero": {"quantity_tonnes": "0"},
        "bad_km_neg": {"distance_km": "-5"},
        "bad_cost_zero": {"freight_cost_inr": "0"},
        "bad_cost_missing": {"freight_cost_inr": ""},
        "bad_date": {"shipment_date": "2024-13-45"},
        "bad_type": {"route_type": "Extra Long"},
        "bad_origin": {"origin": " "},
    }
    rows = [{"shipment_id": sid, **good, **v} for sid, v in variants.items()]
    rows.append({"shipment_id": "ok1", **good})  # duplicate id
    df, rep = clean_shipments(pd.DataFrame(rows))
    assert sorted(df["shipment_id"]) == ["ok1", "ok2"]
    d = rep.to_dict()
    assert d["rows_read"] == 11 and d["rows_kept"] == 2 and d["rows_dropped"] == 9
    by = {k: v["shipment_ids"] for k, v in d["dropped_by_reason"].items()}
    assert set(by["non_numeric_or_missing_value"]) == {"bad_qty_text", "bad_cost_missing"}
    assert set(by["non_positive_quantity_or_distance"]) == {"bad_qty_zero", "bad_km_neg"}
    assert by["non_positive_freight_cost"] == ["bad_cost_zero"]
    assert by["invalid_shipment_date"] == ["bad_date"]
    assert by["invalid_route_type"] == ["bad_type"]
    assert by["missing_origin_or_destination"] == ["bad_origin"]
    assert by["duplicate_shipment_id"] == ["ok1"]


def test_route_with_two_route_types_gets_unique_labels():
    df = make_shipments([
        ("A-B", "Short", "2024-01-01", 1, 1, 2),
        ("A-B", "Long", "2024-01-01", 1, 1, 5),
        ("C-D", "Short", "2024-01-01", 1, 1, 3),
    ])
    assert sorted(df["route"].unique()) == ["A-B (Long)", "A-B (Short)", "C-D"]
    w = compute_route_week_metrics(df)
    assert not w.duplicated(["route", "week_of"]).any()


def test_implausible_shipments_are_kept_flagged_and_counted():
    rows = [("A-B", "Short", "2024-01-0%d" % d, 10, 100, 1000) for d in range(1, 6)]
    rows.append(("A-B", "Short", "2024-01-06", 10, 10, 1000))    # distance typo: 10 km instead of 100
    rows.append(("A-B", "Short", "2024-01-07", 10, 100, 9000))   # 9x the route's usual rate
    df = make_shipments(rows)
    assert len(df) == 7  # nothing dropped
    assert sorted(df.loc[df["data_flag"] != "", "shipment_id"]) == ["S00005", "S00006"]
    assert weekly_costs(df).loc[0, "n_suspect"] == 2


def test_missing_column_raises_schema_error():
    with pytest.raises(SchemaError):
        clean_shipments(pd.DataFrame({"shipment_id": ["x"]}))


# Oracle on the real data --------------------------------------------------------------
def test_matches_sample_rows_from_brief(tfidf_run):
    """The organisers' sample rows are an external oracle for the graded numeric fields."""
    import csv
    from src.config import SAMPLE_OUTPUT_PATH

    out = tfidf_run.output.set_index(["route", "week_of"])
    with open(SAMPLE_OUTPUT_PATH, newline="") as f:
        sample = list(csv.reader(f))[1:]
    for r in sample:
        mine = out.loc[(r[0], r[1])]
        assert mine["cost_per_tonne_km"] == r[2]
        assert mine["vs_own_history"] == r[3]
        assert mine["vs_similar_routes"] == r[4]
        assert mine["flagged"] == r[5]
        assert mine["matched_note_id"] == r[6]


def test_independent_pure_python_oracle(tfidf_run):
    """Re-derive C, H, P for all 728 route-weeks with csv + datetime only (no pandas)."""
    import csv
    import datetime as dt
    from collections import defaultdict
    from src.config import SHIPMENTS_PATH

    cost, tk, rtype = defaultdict(float), defaultdict(float), {}
    with open(SHIPMENTS_PATH, newline="") as f:
        for r in csv.DictReader(f):
            d = dt.date.fromisoformat(r["shipment_date"])
            monday = d - dt.timedelta(days=d.weekday())
            key = (f"{r['origin']}-{r['destination']}", monday)
            cost[key] += float(r["freight_cost_inr"])
            tk[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
            rtype[key[0]] = r["route_type"]
    C = {k: cost[k] / tk[k] for k in cost}

    audit = tfidf_run.audit
    assert len(audit) == len(C) == 728
    for a in audit.itertuples():
        wk = dt.date.fromisoformat(a.week_of)
        c = C[(a.route, wk)]
        prior = [C[(a.route, wk - dt.timedelta(weeks=j))] for j in range(1, 9)
                 if (a.route, wk - dt.timedelta(weeks=j)) in C]
        peers = [C[(r, wk)] for r in rtype if r != a.route and rtype[r] == a.route_type and (r, wk) in C]
        assert a.cost_per_tonne_km == pytest.approx(c, rel=1e-9)
        if prior:
            h = sum(prior) / len(prior)
            assert a.own_history_avg == pytest.approx(h, rel=1e-9)
            assert a.own_history_pct == pytest.approx(100 * (c - h) / h, rel=1e-7, abs=1e-9)
        else:
            assert np.isnan(a.own_history_avg)
        p = sum(peers) / len(peers)
        assert a.peer_avg == pytest.approx(p, rel=1e-9)
        assert a.peer_pct == pytest.approx(100 * (c - p) / p, rel=1e-7, abs=1e-9)


def test_dates_with_time_component_are_accepted():
    df, rep = clean_shipments(pd.DataFrame([{
        "shipment_id": "a", "origin": "A", "destination": "B", "route_type": "Short", "material": "x",
        "quantity_tonnes": "1", "distance_km": "1", "freight_cost_inr": "5",
        "shipment_date": "2024-01-07 23:59:00", "transporter": "T"}]))
    assert rep.rows_kept == 1 and df.loc[0, "shipment_date"] == pd.Timestamp("2024-01-07")
    assert week_start(df["shipment_date"])[0] == pd.Timestamp("2024-01-01")


def test_all_rows_invalid_raises_clear_error():
    with pytest.raises(ValueError, match="No valid shipment rows"):
        clean_shipments(pd.DataFrame([{
            "shipment_id": "a", "origin": "A", "destination": "B", "route_type": "Short", "material": "x",
            "quantity_tonnes": "0", "distance_km": "1", "freight_cost_inr": "5",
            "shipment_date": "2024-01-01", "transporter": "T"}]))
