"""Ask-the-assistant: deterministic question answering over the pipeline's own results.

No LLM and no API. A question is parsed for routes (or cities, or a route type), a period (month,
year, a date, "latest week") and an intent; the answer is assembled only from the computed
route-weeks, the validated notes and the configured method. If the question can't be grounded,
the assistant says so and lists what it can answer instead of guessing.

Intents, checked in this order (first match wins):
    note lookup        "What does N003 say?"
    method             "How is cost per tonne-km calculated?", "What are the thresholds?"
    counts             "How many weeks were flagged?"
    route list         "Which routes are monitored?"
    notes list         "What notes are there?"
    comparison         "Compare Mumbai-Pune and Delhi-Jaipur"
    ranking            "highest/lowest anomalies", "cheapest/most expensive route"
    justified weeks    "Which weeks were justified by a note?"
    unexplained weeks  "Which routes have unexplained increases?", "Why is Mumbai-Pune flagged?"
    cost lookup        "What was the cost on Mumbai-Delhi in the last week of 2025?"
    route / period     "Why did Chennai-Bangalore get pricier in March 2025?"
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .config import DEFAULT_CONFIG, FLAG_JUSTIFIED, FLAG_NO, FLAG_YES, Config

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
ROUTE_TYPES = ["Short", "Medium", "Long"]

_ANOMALY_WORDS = r"\b(anomal\w*|flag\w*|increase\w*|rise|rises|rising|spike\w*|jump\w*|unexplained|justified)\b"
_DESC = r"\b(highest|largest|biggest|top|worst|most expensive|costliest|priciest)\b"
_ASC = r"\b(lowest|smallest|least|cheapest|minimum)\b"


@dataclass
class Answer:
    text: str
    rows: pd.DataFrame  # the evidence rows the answer is built from (submission columns)
    matched: bool = True  # False when no intent matched and the answer is the help text


@dataclass
class _Query:
    text: str
    lower: str
    routes: list[str]          # routes the question names (by route, city or route type)
    month: int | None
    year: int | None
    date: pd.Timestamp | None  # a specific day named in the question (its week is used)
    latest: bool               # "latest/last/most recent week"
    output: pd.DataFrame
    audit: pd.DataFrame | None
    cards: dict | None
    cfg: Config

    @property
    def has_period(self) -> bool:
        return bool(self.month or self.year or self.date is not None or self.latest)

    def scope(self) -> str:
        where = ", ".join(self.routes) if self.routes else "all routes"
        if self.date is not None:
            return f"{where}, week of {_monday(self.date):%Y-%m-%d}"
        when = f"{MONTHS[self.month - 1].title() if self.month else ''} {self.year or ''}".strip()
        return f"{where} in {when}" if when else where


# ---------------------------------------------------------------- parsing
def _find_routes(q: str, routes: list[str]) -> list[str]:
    ql = q.lower().replace("–", "-").replace("—", "-")
    exact = [r for r in routes if r.lower() in ql or "-".join(reversed(r.lower().split("-"))) in ql]
    if exact:
        return exact
    tokens = set(re.findall(r"[a-z]+", ql))
    scored = [(sum(c in tokens for c in r.lower().split("-")), r) for r in routes]
    best = max((s for s, _ in scored), default=0)
    return [r for s, r in scored if s == best and s > 0]


def _routes_of_type(q: str, audit: pd.DataFrame | None) -> list[str]:
    m = re.search(r"\b(short|medium|long)\b(?:[- ](?:haul|distance|length))?[- ]?routes?\b", q.lower())
    if not m or audit is None:
        return []
    return sorted(audit.loc[audit["route_type"] == m.group(1).title(), "route"].unique())


def _month_pattern(m: str) -> str:
    if m == "may":  # also a verb ("why may costs rise?"): only a month next to a date word or a year
        return r"\b(?:in|during|of|since|by|until|from|early|mid|late|end of)\s+may\b|\bmay\s+20\d\d\b"
    return rf"\b{m}\b|\b{m[:3]}\b"


def _find_period(q: str) -> tuple[int | None, int | None]:
    ql = q.lower()
    year = re.search(r"\b(20\d\d)\b", ql)
    month = next((i for i, m in enumerate(MONTHS, start=1) if re.search(_month_pattern(m), ql)), None)
    return month, int(year.group(1)) if year else None


def _find_date(q: str) -> pd.Timestamp | None:
    m = re.search(r"\b(20\d\d-\d\d-\d\d)\b", q)
    if not m:
        return None
    try:
        return pd.Timestamp(m.group(1))
    except ValueError:
        return None


def _monday(d: pd.Timestamp) -> pd.Timestamp:
    return d - pd.Timedelta(days=d.weekday())


def _filter_period(df: pd.DataFrame, month, year) -> pd.DataFrame:
    w = pd.to_datetime(df["week_of"])
    # A week belongs to a period if any of its 7 days falls in it. Month and year are checked on the same
    # day, so the week of 2024-12-30 is in December 2024 and January 2025, not December 2025.
    mask = pd.Series(False, index=df.index)
    for offset in range(7):
        d = w + pd.Timedelta(days=offset)
        mask |= ((d.dt.month == month) if month else True) & ((d.dt.year == year) if year else True)
    return df[mask]


def _scoped(q: _Query, df: pd.DataFrame) -> pd.DataFrame:
    """Rows of `df` for the question's routes and period."""
    if q.routes:
        df = df[df["route"].isin(q.routes)]
    if q.date is not None:
        return df[df["week_of"] == f"{_monday(q.date):%Y-%m-%d}"]
    df = _filter_period(df, q.month, q.year) if (q.month or q.year) else df
    if q.latest and not df.empty:
        df = df[df["route"].map(df.groupby("route")["week_of"].max()) == df["week_of"]]
    return df


def _numeric(q: _Query) -> pd.DataFrame:
    """Submission rows plus float columns _cost, _own, _peer (full precision when the audit is available)."""
    df = q.output.copy()
    if q.audit is not None:
        exact = q.audit.set_index(["route", "week_of"])[["cost_per_tonne_km", "own_history_pct", "peer_pct"]]
        exact.columns = ["_cost", "_own", "_peer"]
        return df.join(exact, on=["route", "week_of"])
    df["_cost"] = df["cost_per_tonne_km"].astype(float)
    df["_own"] = df["vs_own_history"].str.extract(r"^([+-]?\d+\.\d)", expand=False).astype(float)
    df["_peer"] = df["vs_similar_routes"].str.extract(r"^([+-]?\d+\.\d)", expand=False).astype(float)
    return df


def _rows(df: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    return df[list(output.columns)]


def _line(r) -> str:
    note = f" [{r.matched_note_id}]" if r.matched_note_id else ""
    return f"- {r.route}, week of {r.week_of}: {r.flagged}{note}. {r.reason}"


def _week_line(r) -> str:
    note = f" [{r.matched_note_id}]" if r.matched_note_id else ""
    return (f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km} INR per tonne-km "
            f"({r.vs_own_history}; {r.vs_similar_routes}) -> {r.flagged}{note}")


# ---------------------------------------------------------------- intents
def _note_lookup(q: _Query) -> Answer | None:
    ids = sorted({f"N{int(n):03d}" for n in re.findall(r"\bn0*(\d{1,3})\b", q.lower)})
    if not ids:
        return None
    out, lines = q.output, []
    for nid in ids:
        card = (q.cards or {}).get(nid)
        if card is None and q.cards is not None:
            lines.append(f"{nid}: there is no note with this id.")
            continue
        if card is not None:
            lines.append(f"{nid} ({card.date:%Y-%m-%d}, {card.applies_to}): \"{card.text}\"")
            applies = {"all": "all routes", "none": "no route in this dataset"}.get(card.scope, card.coverage)
            lines.append(f"  Read as: applies to {applies}; "
                         f"window {card.start:%Y-%m-%d} to {card.end:%Y-%m-%d} ({card.window_basis}); "
                         f"cost impact: {card.impact.replace('_', ' ')}"
                         + (f" ('{card.impact_evidence}')" if card.impact_evidence else "") + ".")
        used = out[out["matched_note_id"] == nid]
        lines.append(f"  It justifies {len(used)} week(s): " + ", ".join(f"{r.route} {r.week_of}" for r in used.itertuples())
                     if len(used) else "  It does not justify any flagged week.")
    rows = out[out["matched_note_id"].isin(ids)]
    return Answer("\n".join(lines), rows)


def _method(q: _Query) -> Answer | None:
    how = re.search(r"\bhow (is|are|do|does|did|was|were)\b.*\b(calculat|comput|measur|decid|determin|flag|work|"
                    r"defin|chosen|set|justif|check|verif)", q.lower)
    terms = re.search(r"\b(formula|thresholds?|baselines?|methodology|method)\b", q.lower)
    what_is = (re.search(r"\bwhat (is|are|does|do)\b.*\b(cost per tonne|own history|similar routes|"
                         r"no \(justified\)|mean|means)\b", q.lower) and not q.routes and not q.has_period)
    if not (how or terms or what_is):
        return None
    c = q.cfg
    text = "\n".join([
        "How the numbers and verdicts work:",
        "- Weeks run Monday to Sunday; week_of is the Monday.",
        "- Weekly cost per tonne-km = total freight cost of the route's shipments that week divided by "
        "their total tonnes x km.",
        f"- vs own history: compared with the average of the route's previous {c.history_weeks} weeks (earlier "
        f"weeks only; at the start of the data, all prior weeks available).",
        "- vs similar routes: compared with the average of the other routes of the same type "
        "(Short / Medium / Long) in the same week.",
        f"- Flagged when the cost is rising vs its own history and is at least {c.own_history_threshold_pct:g}% "
        f"above it or at least {c.peer_threshold_pct:g}% above similar routes.",
        "- A flagged week becomes 'No (justified)' only if a context note covers that route and week, states "
        "a cost increase, and fits the size of the rise. Otherwise it stays 'Yes' for human review.",
    ])
    return Answer(text, q.output.head(0))


def _counts(q: _Query) -> Answer | None:
    if not re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q.lower):
        return None
    df = _scoped(q, q.output)
    n_yes, n_just = int((df["flagged"] == FLAG_YES).sum()), int((df["flagged"] == FLAG_JUSTIFIED).sum())
    text = (f"For {q.scope()}: {df['route'].nunique()} route(s), {df['week_of'].nunique()} week(s), "
            f"{len(df)} route-weeks. {n_yes + n_just} were out of the ordinary: {n_yes} unexplained (flagged) "
            f"and {n_just} justified by a note.")
    return Answer(text, df[df["flagged"] != FLAG_NO])


def _route_list(q: _Query) -> Answer | None:
    if not (re.search(r"\b(which|what|list|show)\b.*\broutes\b", q.lower)
            and re.search(r"\b(monitor\w*|track\w*|cover\w*|exist|there|available|analy[sz]\w*|all routes)\b",
                          q.lower)):
        return None
    types = q.audit.groupby("route")["route_type"].first() if q.audit is not None else {}
    weeks = q.output.groupby("route")["week_of"].agg(["min", "max", "count"])
    lines = [f"- {r}" + (f" ({types[r]})" if r in types else "") + f": {n} weeks, {mn} to {mx}"
             for r, (mn, mx, n) in weeks.iterrows()]
    return Answer(f"{len(weeks)} routes are monitored:\n" + "\n".join(lines), q.output.head(0))


def _notes_list(q: _Query) -> Answer | None:
    if q.cards is None or not re.search(r"\b(which|what|list|show|all)\b.*\bnotes\b", q.lower):
        return None
    if re.search(r"justif|used|matched", q.lower):
        return None  # "which notes were used" is about justified weeks
    used = q.output["matched_note_id"].value_counts()
    lines = [f"- {c.note_id} ({c.date:%Y-%m-%d}, {c.applies_to}): \"{c.summary}\" -> "
             + (f"justifies {used[c.note_id]} week(s)" if c.note_id in used else "justifies no week")
             for c in sorted(q.cards.values(), key=lambda c: c.note_id)]
    return Answer(f"{len(lines)} context notes:\n" + "\n".join(lines), q.output.head(0))


def _comparison(q: _Query) -> Answer | None:
    if len(q.routes) < 2 or not re.search(r"\bcompare\w*|\bversus\b|\bvs\.?\b|\bdifference\b|\bbetween\b", q.lower):
        return None
    df = _scoped(q, _numeric(q))
    lines = []
    for route, g in df.groupby("route"):
        g = g.sort_values("week_of")
        last = g.iloc[-1]
        lines.append(f"- {route}: average {g['_cost'].mean():.2f} INR per tonne-km over {len(g)} week(s) "
                     f"(range {g['_cost'].min():.2f} to {g['_cost'].max():.2f}); latest {last['_cost']:.2f} "
                     f"(week of {last['week_of']}); {int((g['flagged'] == FLAG_YES).sum())} unexplained, "
                     f"{int((g['flagged'] == FLAG_JUSTIFIED).sum())} justified week(s).")
    note = "Shorter routes normally cost more per tonne-km, so compare a route with its own type first."
    return Answer(f"Comparison for {q.scope()}:\n" + "\n".join(lines) + f"\n{note}",
                  _rows(df[df["flagged"] != FLAG_NO], q.output))


def _ranking(q: _Query) -> Answer | None:
    desc, asc = re.search(_DESC, q.lower), re.search(_ASC, q.lower)
    if not (desc or asc):
        return None
    ascending = bool(asc) and not desc
    df = _numeric(q)
    if re.search(_ANOMALY_WORDS, q.lower):
        df = _scoped(q, df[df["flagged"] != FLAG_NO])
        top = df.sort_values(["_own", "_peer"], ascending=ascending).head(5)
        word = "Smallest" if ascending else "Largest"
        lines = [f"- {r.route}, week of {r.week_of}: {r.vs_own_history}, {r.vs_similar_routes} -> {r.flagged}"
                 + (f" [{r.matched_note_id}]" if r.matched_note_id else "") for r in top.itertuples()]
        if top.empty:
            return Answer(f"No flagged or justified weeks for {q.scope()}.", _rows(top, q.output))
        return Answer(f"{word} cost anomalies vs own history (flagged or justified), {q.scope()}:\n"
                      + "\n".join(lines), _rows(top, q.output))
    # No anomaly words: rank routes by their average cost per tonne-km.
    df = _scoped(q, df)
    avg = df.groupby("route")["_cost"].agg(["mean", "count"]).sort_values("mean", ascending=ascending)
    word = "Cheapest" if ascending else "Most expensive"
    lines = [f"- {r}: {m:.2f} INR per tonne-km on average over {int(n)} week(s)" for r, (m, n) in avg.iterrows()]
    note = "Shorter routes normally cost more per tonne-km, so this mostly reflects route length."
    return Answer(f"{word} first, {q.scope()}:\n" + "\n".join(lines) + f"\n{note}", q.output.head(0))


def _justified(q: _Query) -> Answer | None:
    if not re.search(r"\bjustif\w*|\bexplained by\b|\bbacked by\b|\bnotes? (were |was )?(used|matched)\b", q.lower):
        return None
    if re.search(r"\bunjustif|\bunexplained", q.lower):
        return None
    df = _scoped(q, q.output[q.output["flagged"] == FLAG_JUSTIFIED])
    if df.empty:
        return Answer(f"No week was justified by a note for {q.scope()}.", df)
    return Answer(f"Weeks justified by a note, {q.scope()}:\n" + "\n".join(_line(r) for r in df.itertuples()), df)


def _unexplained(q: _Query) -> Answer | None:
    if not re.search(r"unexplained|unjustified|flagged|need(s)? review|suspicious", q.lower) or q.month:
        return None
    df = _scoped(q, q.output[q.output["flagged"] == FLAG_YES])
    if df.empty:
        return Answer("No unexplained increases" + (f" on {', '.join(q.routes)}." if q.routes else "."), df)
    summary = df.groupby("route")["week_of"].agg(["count", "min", "max"])
    lines = [f"- {r}: {c} week(s) flagged, {mn} to {mx}" for r, (c, mn, mx) in summary.iterrows()]
    text = "Routes with unexplained cost increases (flagged for review):\n" + "\n".join(lines)
    if q.routes:  # a question about specific routes gets the reason for every flagged week
        text += "\n\nWhy, week by week:\n" + "\n".join(_line(r) for r in df.itertuples())
    return Answer(text, df)


def _cost_lookup(q: _Query) -> Answer | None:
    asks = re.search(r"\b(cost|costs|price|prices|rate|rates|cost per tonne)\b", q.lower) and re.search(
        r"\b(what|how much|show|tell|give|list|trend|average|value)\b", q.lower)
    if not asks or re.search(r"\bwhy\b", q.lower) or not (q.routes or q.has_period):
        return None
    df = _scoped(q, _numeric(q))
    if df.empty:
        return Answer(f"There are no computed weeks for {q.scope()}.", _rows(df, q.output))
    df = df.sort_values(["route", "week_of"])
    if len(df) <= 12:
        lines = [_week_line(r) for r in df.itertuples()]
    else:
        lines = []
        for route, g in df.groupby("route"):
            lo, hi, last = g.loc[g["_cost"].idxmin()], g.loc[g["_cost"].idxmax()], g.iloc[-1]
            lines.append(f"- {route}: average {g['_cost'].mean():.2f} INR per tonne-km over {len(g)} weeks; "
                         f"lowest {lo['_cost']:.2f} (week of {lo['week_of']}), highest {hi['_cost']:.2f} "
                         f"(week of {hi['week_of']}), latest {last['_cost']:.2f} (week of {last['week_of']}); "
                         f"{int((g['flagged'] == FLAG_YES).sum())} unexplained and "
                         f"{int((g['flagged'] == FLAG_JUSTIFIED).sum())} justified week(s).")
    return Answer(f"Cost per tonne-km, {q.scope()}:\n" + "\n".join(lines), _rows(df, q.output))


_COST_TOPIC = (r"\b(cost\w*|pric\w*|expens\w*|rate|rates|increase\w*|rise|rose|rising|flag\w*|odd|unusual|weird|"
               r"strange|anomal\w*|spike\w*|jump\w*|justif\w*|explain\w*|why|happen\w*|going on|wrong|issue\w*|"
               r"problem\w*|change\w*|anything)\b")


def _route_or_period(q: _Query) -> Answer | None:
    # Naming a city is not enough ("what's the weather in Delhi?"): the question must be about costs.
    if not (q.routes or q.has_period) or not re.search(_COST_TOPIC, q.lower):
        return None
    df = _scoped(q, q.output[q.output["flagged"] != FLAG_NO])
    if df.empty:
        return Answer(f"No flagged or justified cost increase found for {q.scope()}. "
                      f"Costs there stayed within the review thresholds.", df)
    just, unexp = df[df["flagged"] == FLAG_JUSTIFIED], df[df["flagged"] == FLAG_YES]
    head = f"For {q.scope()}: {len(just)} justified and {len(unexp)} unexplained week(s)."
    return Answer(head + "\n" + "\n".join(_line(r) for r in df.itertuples()), df)


INTENTS = [_note_lookup, _method, _counts, _route_list, _notes_list, _comparison, _ranking, _justified,
           _unexplained, _cost_lookup, _route_or_period]

FALLBACK = ("I can only answer from this analysis, and I couldn't match that question to it. You can ask, "
            "for example:\n"
            "- Why did Chennai-Bangalore get pricier in March 2025?\n"
            "- Which routes have unexplained increases? / Why is Mumbai-Pune flagged?\n"
            "- Which weeks were justified by a note? / What does N003 say?\n"
            "- Which routes have the highest (or lowest) cost anomalies?\n"
            "- What was the cost on Mumbai-Delhi in the latest week? / Which route is cheapest?\n"
            "- Compare Mumbai-Pune and Delhi-Jaipur / How many weeks were flagged in 2025?\n"
            "- How is cost per tonne-km calculated? / What are the thresholds?")


def answer(question: str, output: pd.DataFrame, audit: pd.DataFrame | None = None, cards: dict | None = None,
           cfg: Config = DEFAULT_CONFIG) -> Answer:
    """`audit` adds full-precision numbers and route types; `cards` (parsed notes) enables note questions."""
    text = question.strip()
    lower = text.lower()
    routes = sorted(output["route"].unique())
    found = _find_routes(text, routes) or _routes_of_type(text, audit)
    month, year = _find_period(text)
    q = _Query(text=text, lower=lower, routes=found, month=month, year=year, date=_find_date(text),
               latest=bool(re.search(r"\b(latest|most recent|last)\s+week\b", lower)),
               output=output, audit=audit, cards=cards, cfg=cfg)
    if not text:
        return Answer(FALLBACK, output.head(0), matched=False)
    for intent in INTENTS:
        result = intent(q)
        if result is not None:
            return result
    return Answer(FALLBACK, output.head(0), matched=False)
