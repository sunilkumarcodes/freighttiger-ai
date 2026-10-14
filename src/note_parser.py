"""Turn a free-text context note into a structured, auditable NoteCard.

This is deterministic text parsing, not a model. Each field records *which phrase* it
was derived from so a reviewer can check it against the note in seconds.

Policies (see README "Temporal applicability"):
  * explicit range ("from Feb 24 to Mar 8")  -> that range
  * open-ended start ("starting this week")  -> start week + `open_ended_note_weeks` weeks
  * anything else (duration unspecified)     -> the week containing the note date only.
    A note never applies indefinitely and never applies before its own date.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _mon(name: str) -> str:
    return rf"(?P<{name}>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


_DAY = r"(?P<{}>\d{{1,2}})(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s*\d{4})?"
_RANGE_SEP = r"\s*(?:to|until|till|through|thru|and|-|–|—)\s*"
# "from Feb 24 to Mar 8", "between Feb 24 and Mar 8", "(Feb 24 - Mar 8)", "Feb 24 to 28"
RANGE_MONTH_FIRST = (rf"{_mon('m1')}\s+{_DAY.format('d1')}{_YEAR}{_RANGE_SEP}"
                     rf"(?:{_mon('m2')}\s+)?{_DAY.format('d2')}\b")
# "from 24 Feb to 8 Mar", "24th February - 8th March"
RANGE_DAY_FIRST = (rf"\b{_DAY.format('d1')}\s+{_mon('m1')}{_YEAR}{_RANGE_SEP}"
                   rf"{_DAY.format('d2')}\s+{_mon('m2')}")
# "until Mar 8" (start = note date)
END_ONLY = rf"\b(?:until|till|through)\s+{_mon('m2')}\s+{_DAY.format('d2')}"

IMPACT_INCREASE = "increase"
IMPACT_NONE = "no_increase"
IMPACT_UNKNOWN = "unknown"

# Precedence: explicit statement that costs did NOT rise  >  explicit cost increase
#             >  description of normal / improved conditions  >  unknown.
_COST_NOUN = r"(?:costs?|rates?|prices?|tariffs?|charges?|fares?)"
COST_NEGATION_CUES = [
    rf"{_COST_NOUN} (?:were|was|are|is|have been|has been) not (?:significantly |materially )?(?:affected|impacted|changed|increased|raised|hiked|revised)",
    r"(?:surcharges?|premiums?|levy|levies|fees?|tolls?) (?:was|were|has been|have been) (?:waived|withdrawn|removed|cancelled|canceled|scrapped|rolled back)",
    rf"{_COST_NOUN} (?:did|do|does) not (?:change|rise|increase|go up)",
    rf"without (?:a|any) (?:rate|price|cost|tariff) (?:change|increase|revision|impact)",
    r"\babsorb(?:ed|s|ing)?\b",
    rf"no (?:rate|price|cost|tariff) (?:change|increase|impact)",
    rf"no (?:change|increase|rise) in (?:the )?(?:freight |transport(?:ation)? )?{_COST_NOUN}",
    rf"{_COST_NOUN} (?:remained|stayed) (?:unchanged|flat|stable)",
    r"without passing (?:it|them|this|the cost) on",
    r"unaffected",
]
# Cost went DOWN: never a justification for a rise.
COST_DECREASE_CUES = [
    r"\bcheaper\b",
    rf"(?:reduced|cut|lowered|slashed) (?:the )?(?:freight |transport(?:ation)? )?{_COST_NOUN}",
    rf"{_COST_NOUN} (?:fell|dropped|declined|decreased|eased|went down)",
]
COST_INCREASE_CUES = [
    r"higher (?:trip |freight |transport(?:ation)? |operating )?(?:costs?|rates?|charges?|fares?)",
    r"surcharges?",
    r"push(?:ed|es|ing)? up",
    r"(?:drove|driving|drives|drive) up",
    r"(?:raised|raising|raise[sd]?|increased|increasing|hiked|hiking) (?:the )?(?:freight |transport(?:ation)? |trip )?(?:rates?|prices?|costs?|charges?|fares?)",
    rf"{_COST_NOUN} (?:rose|rise|rising|increased|went up|jumped|surged|spiked|climbed)",
    rf"{_COST_NOUN}(?: on [\w\s-]+?)? (?:were|was|have been|has been) (?:hiked|raised|increased|revised upward)",
    r"(?:price|cost|rate|tariff|toll) (?:hike|increase)",
    r"new toll",
    r"more expensive",
    r"\bcostlier\b",
    r"(?:charged|levied|applied|imposed|added) (?:a |an )?(?:\d+(?:\.\d+)? ?% )?(?:premium|surcharge|levy|fee)",
]
# An increase cue does not count if its clause negates it ("no surcharge was applied",
# "costs were not increased") or if it is cancelled right after ("surcharge was waived").
_NEGATION_BEFORE = re.compile(r"\b(?:no|not|never|without|nor|neither)\b|n't\b", re.IGNORECASE)
# Speculation is not evidence: "rates could rise", "if diesel prices rise", "expected to push up".
_HEDGE_BEFORE = re.compile(r"\b(?:might|could|would|if|expected to|likely to|forecast|projected|planned|proposed)\b",
                           re.IGNORECASE)
_CANCELLED_AFTER = re.compile(
    r"^\W*(?:\w+\W+){0,3}?(?:was|were|has been|have been|is|are)?\s*(?:waived|withdrawn|removed|cancell?ed|"
    r"scrapped|rolled back|not applied|not levied|not charged|not passed)", re.IGNORECASE)
NORMAL_CONDITION_CUES = [
    r"returned to normal",
    r"remained (?:normal|stable)",
    r"no (?:major |significant )?disruptions?",
    r"improved road conditions",
    r"resurfacing work was completed",
    r"\bstable\b",
]
OUT_OF_SCOPE_CUES = [
    r"not (?:a )?part of this dataset",
    r"routes? (?:are|is) not (?:covered|included)",
]
OPEN_ENDED_CUES = [r"starting this week", r"from this week", r"onwards", r"with effect from", r"starting\b"]


@dataclass(frozen=True)
class NoteCard:
    note_id: str
    date: pd.Timestamp
    applies_to: str
    text: str
    scope: str                      # "route" | "all" | "none"
    routes: tuple[frozenset[str], ...]  # route scope: e.g. ({"chennai", "bangalore"},); direction-agnostic
    start: pd.Timestamp
    end: pd.Timestamp               # inclusive
    window_basis: str
    impact: str
    impact_evidence: str
    magnitude_pct: tuple[float, float] | None

    @property
    def summary(self) -> str:
        """First sentence of the note, verbatim (used in explanations; nothing paraphrased)."""
        m = re.match(r"(.+?[.!?])(\s|$)", self.text)
        return (m.group(1) if m else self.text).strip()


def _find(cues: list[str], text: str) -> str | None:
    for cue in cues:
        m = re.search(cue, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _increase_cue(text: str) -> tuple[str | None, str | None, str | None]:
    """First increase cue not negated or hedged in its clause. Returns (cue, negated_cue, hedged_cue)."""
    negated = hedged = None
    for cue in COST_INCREASE_CUES:
        for m in re.finditer(cue, text, flags=re.IGNORECASE):
            clause_start = max(text.rfind(c, 0, m.start()) for c in ".;:!?") + 1
            before = text[clause_start:m.start()]
            if _NEGATION_BEFORE.search(before) or _CANCELLED_AFTER.search(text[m.end():m.end() + 40]):
                negated = negated or m.group(0)
                continue
            if _HEDGE_BEFORE.search(before):
                hedged = hedged or m.group(0)
                continue
            return m.group(0), None, None
    return None, negated, hedged


def classify_impact(text: str) -> tuple[str, str]:
    for cues in (COST_NEGATION_CUES, COST_DECREASE_CUES):
        hit = _find(cues, text)
        if hit:
            return IMPACT_NONE, hit
    cue, negated, hedged = _increase_cue(text)
    if cue:
        return IMPACT_INCREASE, cue
    if negated:
        return IMPACT_NONE, f"negated: {negated}"
    if hedged:
        return IMPACT_UNKNOWN, f"speculative: {hedged}"
    hit = _find(NORMAL_CONDITION_CUES, text)
    if hit:
        return IMPACT_NONE, hit
    return IMPACT_UNKNOWN, ""


def parse_magnitude(text: str) -> tuple[float, float] | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*%", text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        v = float(m.group(1))
        return (v, v)
    return None


def _resolve(month: str, day: str, ref: pd.Timestamp) -> pd.Timestamp:
    """Month/day mentioned in a note -> date in the note's year (or the next, if it wraps)."""
    d = pd.Timestamp(year=ref.year, month=MONTHS[month[:3].lower()], day=int(day))
    if d < ref - pd.Timedelta(days=31):  # e.g. a December note mentioning "Jan 5"
        d = d.replace(year=ref.year + 1)
    return d


def parse_window(text: str, note_date: pd.Timestamp, open_ended_weeks: int) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    week0 = note_date - pd.Timedelta(days=note_date.weekday())
    for pattern in (RANGE_MONTH_FIRST, RANGE_DAY_FIRST):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                start = _resolve(m["m1"], m["d1"], note_date)
                end = _resolve(m["m2"] or m["m1"], m["d2"], start)
            except ValueError:  # e.g. "Feb 30"
                continue
            if end >= start:
                return start, end, f"explicit range stated in note ({m.group(0).strip()})"
    m = re.search(END_ONLY, text, re.IGNORECASE)
    if m:
        try:
            end = _resolve(m["m2"], m["d2"], note_date)
        except ValueError:
            end = None
        if end is not None and end >= note_date:
            return note_date, end, f"from note date until date stated in note ({m.group(0).strip()})"
    if _find(OPEN_ENDED_CUES, text):
        end = week0 + pd.Timedelta(weeks=open_ended_weeks) - pd.Timedelta(days=1)
        return note_date, end, (f"open-ended from {note_date:%Y-%m-%d}; honoured for {open_ended_weeks} weeks "
                                f"(after that the change is inside the own-history baseline)")
    end = week0 + pd.Timedelta(days=6)
    return note_date, end, "no duration stated; applies to the week of the note date only"


ALL_SCOPE = re.compile(r"^(?:all(?:\s+(?:routes?|corridors?|lanes?))?(?:\s+(?:nationwide|across india))?"
                       r"|nationwide|pan[- ]india|all[- ]india)$", re.IGNORECASE)


def parse_route(text: str) -> frozenset[str] | None:
    """'Chennai-Bangalore', 'Chennai – Bangalore', 'Chennai to Bangalore' -> {'chennai', 'bangalore'}."""
    t = re.sub(r"\s+to\s+", "-", text.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*[-–—]\s*", "-", t)
    ends = [p.strip().lower() for p in t.split("-") if p.strip()]
    return frozenset(ends) if len(ends) == 2 and len(set(ends)) == 2 else None


def parse_scope(applies_to: str, text: str) -> tuple[str, tuple[frozenset[str], ...]]:
    if _find(OUT_OF_SCOPE_CUES, text):
        return "none", ()
    a = re.sub(r"\s+", " ", applies_to.strip())
    if ALL_SCOPE.match(a):
        return "all", ()
    parts = re.split(r"\s*(?:,|;|/|&|\band\b)\s*", a, flags=re.IGNORECASE)
    routes = {r for r in (parse_route(p) for p in parts if p) if r}
    # Unparseable scope -> no routes -> never matches (fails safe).
    return "route", tuple(sorted(routes, key=sorted))


def build_note_card(row, open_ended_weeks: int = 8) -> NoteCard:
    date = pd.Timestamp(row["date"]).normalize()
    text = str(row["note"])
    scope, routes = parse_scope(str(row["applies_to"]), text)
    start, end, basis = parse_window(text, date, open_ended_weeks)
    impact, evidence = classify_impact(text)
    return NoteCard(
        note_id=str(row["note_id"]), date=date, applies_to=str(row["applies_to"]), text=text,
        scope=scope, routes=routes, start=start, end=end, window_basis=basis,
        impact=impact, impact_evidence=evidence, magnitude_pct=parse_magnitude(text),
    )


def build_note_cards(notes: pd.DataFrame, open_ended_weeks: int = 8) -> dict[str, NoteCard]:
    return {r["note_id"]: build_note_card(r, open_ended_weeks) for _, r in notes.iterrows()}
