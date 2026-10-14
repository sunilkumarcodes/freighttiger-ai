"""Ask-the-assistant: deterministic question answering over the pipeline's own results.

No LLM. A question is parsed for a route (or city), a month/year and an intent; the answer
is assembled only from computed route-weeks and validated notes. If the question can't be
grounded, the assistant says so instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .config import FLAG_JUSTIFIED, FLAG_YES

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


@dataclass
class Answer:
    text: str
    rows: pd.DataFrame  # the evidence rows the answer is built from


def _find_routes(q: str, routes: list[str]) -> list[str]:
    ql = q.lower().replace("–", "-").replace("—", "-")
    exact = [r for r in routes if r.lower() in ql or "-".join(reversed(r.lower().split("-"))) in ql]
    if exact:
        return exact
    tokens = set(re.findall(r"[a-z]+", ql))
    scored = [(sum(c in tokens for c in r.lower().split("-")), r) for r in routes]
    best = max((s for s, _ in scored), default=0)
    return [r for s, r in scored if s == best and s > 0]


def _find_period(q: str) -> tuple[int | None, int | None]:
    ql = q.lower()
    year = re.search(r"\b(20\d\d)\b", ql)
    month = next((i for i, m in enumerate(MONTHS, start=1) if re.search(rf"\b{m}\b|\b{m[:3]}\b", ql)), None)
    return month, int(year.group(1)) if year else None


def _filter_period(df: pd.DataFrame, month, year) -> pd.DataFrame:
    w = pd.to_datetime(df["week_of"])
    # A week belongs to a month if any of its 7 days fall in it.
    ends = w + pd.Timedelta(days=6)
    mask = pd.Series(True, index=df.index)
    if month:
        mask &= (w.dt.month == month) | (ends.dt.month == month)
    if year:
        mask &= (w.dt.year == year) | (ends.dt.year == year)
    return df[mask]


def _line(r) -> str:
    note = f" [{r.matched_note_id}]" if r.matched_note_id else ""
    return f"- {r.route}, week of {r.week_of}: {r.flagged}{note}. {r.reason}"


def answer(question: str, output: pd.DataFrame) -> Answer:
    q = question.strip()
    ql = q.lower()
    routes = sorted(output["route"].unique())
    found = _find_routes(q, routes)
    month, year = _find_period(q)
    notable = output[output["flagged"] != "No"]

    if re.search(r"\b(highest|largest|biggest|top|worst)\b", ql):
        df = output.copy()
        df["_own"] = df["vs_own_history"].str.extract(r"^([+-]?\d+\.\d)").astype(float)
        df["_peer"] = df["vs_similar_routes"].str.extract(r"^([+-]?\d+\.\d)").astype(float)
        df = df[df["flagged"] != "No"]
        if found:
            df = df[df["route"].isin(found)]
        top = df.sort_values(["_own", "_peer"], ascending=False).head(5)
        lines = [f"- {r.route}, week of {r.week_of}: {r.vs_own_history}, {r.vs_similar_routes} -> {r.flagged}"
                 + (f" [{r.matched_note_id}]" if r.matched_note_id else "") for r in top.itertuples()]
        return Answer("Largest cost anomalies vs own history (flagged or justified):\n" + "\n".join(lines),
                      top.drop(columns=["_own", "_peer"]))

    if re.search(r"unexplained|unjustified|flagged|need(s)? review|suspicious", ql) and not month:
        df = output[output["flagged"] == FLAG_YES]
        if found:
            df = df[df["route"].isin(found)]
        if df.empty:
            return Answer("No unexplained increases" + (f" on {', '.join(found)}." if found else "."), df)
        summary = df.groupby("route")["week_of"].agg(["count", "min", "max"])
        lines = [f"- {r}: {c} week(s) flagged, {mn} to {mx}" for r, (c, mn, mx) in summary.iterrows()]
        return Answer("Routes with unexplained cost increases (flagged for review):\n" + "\n".join(lines), df)

    if found or month or year:
        df = notable
        if found:
            df = df[df["route"].isin(found)]
        df = _filter_period(df, month, year)
        scope = (", ".join(found) if found else "all routes") + (
            f" in {MONTHS[month - 1].title() if month else ''} {year or ''}".rstrip() if (month or year) else "")
        if df.empty:
            return Answer(f"No flagged or justified cost increase found for {scope}. "
                          f"Costs there stayed within the review thresholds.", df)
        just, unexp = df[df["flagged"] == FLAG_JUSTIFIED], df[df["flagged"] == FLAG_YES]
        head = f"For {scope}: {len(just)} justified and {len(unexp)} unexplained week(s)."
        return Answer(head + "\n" + "\n".join(_line(r) for r in df.itertuples()), df)

    return Answer("I can only answer from this analysis. Try naming a route (e.g. Chennai-Bangalore), "
                  "a month, or ask 'which routes have unexplained increases?' / "
                  "'which routes have the highest cost anomalies?'.", output.head(0))
