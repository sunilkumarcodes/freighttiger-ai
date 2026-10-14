"""Weekly cost per tonne-km and the two graded baselines.

    C[r,t] = sum_i cost_i / sum_i (tonnes_i * km_i)          (weighted, not mean of ratios)
    H[r,t] = mean of C[r,s] for observed weeks s in [t-8w, t-1w]  (strictly prior)
    P[r,t] = mean of C[j,t] over routes j != r with type(j) == type(r)  (unweighted)

Weeks with no shipments are absent from a route's series: they are not imputed as zero
and not carried forward. The history window is 8 *calendar* weeks, so if a route had a
gap, fewer than 8 values are averaged. In the supplied data every route has every week,
so this is the same as "the last 8 observations".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROUTE_KEY = ["route", "origin", "destination", "route_type"]


def week_start(dates: pd.Series) -> pd.Series:
    """Monday of the ISO week (weeks run Monday-Sunday)."""
    d = pd.to_datetime(dates).dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    df = shipments.copy()
    df["week_of"] = week_start(df["shipment_date"])
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    # Shipments flagged as implausible at ingestion (kept, but worth a look before acting on a flag).
    df["is_suspect"] = (df["data_flag"] != "") if "data_flag" in df else False
    weekly = (
        df.groupby(ROUTE_KEY + ["week_of"], as_index=False)
        .agg(total_cost=("freight_cost_inr", "sum"),
             total_tonne_km=("tonne_km", "sum"),
             n_shipments=("shipment_id", "size"),
             n_suspect=("is_suspect", "sum"))
    )
    weekly["n_suspect"] = weekly["n_suspect"].astype(int)
    weekly["cost_per_tonne_km"] = safe_div(weekly["total_cost"], weekly["total_tonne_km"])
    return weekly.sort_values(["route", "route_type", "week_of"]).reset_index(drop=True)


def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=(den > 0) & np.isfinite(den))
    return out


def pct_change(current, baseline):
    return 100.0 * (safe_div(current, baseline) - 1.0)


def add_own_history(weekly: pd.DataFrame, history_weeks: int = 8) -> pd.DataFrame:
    # Time-based window [t - 8 weeks, t): closed="left" excludes the current week, so there is no look-ahead,
    # and a calendar window (not "last 8 rows") means weeks with no shipments are skipped, never imputed.
    out = weekly.reset_index(drop=True)
    key = ["route", "route_type"]
    rolling = (out.sort_values(key + ["week_of"]).set_index("week_of")
               .groupby(key)["cost_per_tonne_km"]
               .rolling(f"{7 * history_weeks}D", closed="left"))
    stats = pd.DataFrame({"own_history_avg": rolling.mean(), "n_prior_weeks": rolling.count()})
    out = out.join(stats, on=key + ["week_of"])
    out["n_prior_weeks"] = out["n_prior_weeks"].fillna(0).astype(int)
    out["own_history_pct"] = pct_change(out["cost_per_tonne_km"], out["own_history_avg"])
    return out


def add_peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    # Leave-one-out mean within (route_type, week): (sum - self) / (count - 1).
    grp = out.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    total = grp.transform("sum")
    count = grp.transform("count")
    n_peers = (count - 1).astype(int)
    out["peer_avg"] = safe_div(total - out["cost_per_tonne_km"], n_peers)
    out["n_peers"] = n_peers
    out["peer_pct"] = pct_change(out["cost_per_tonne_km"], out["peer_avg"])
    return out


def compute_route_week_metrics(shipments: pd.DataFrame, history_weeks: int = 8) -> pd.DataFrame:
    weekly = weekly_costs(shipments)
    weekly = add_own_history(weekly, history_weeks)
    weekly = add_peer_baseline(weekly)
    return weekly
