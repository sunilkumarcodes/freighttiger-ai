"""Optional Gemini answers for the dashboard assistant: free-form questions, grounded and checked.

The rule-based assistant (assistant.py) handles a fixed set of question types. With a Gemini key,
any question is answered by Gemini from a data pack built from this run's results: the method, every
note with its parsed window and the weeks it justifies, a summary per route, monthly averages, every
flagged or justified week with its reason, and every route-week's numbers. The built-in answer is
added to the pack when the rules recognise the question.

Gemini is told to use only the data pack. Its answer is then checked: every decimal figure, ISO date,
route and note id in it must appear in the pack (a figure may be rounded to 1 decimal). An answer that
fails the check is not shown; the built-in answer is shown instead, with the reason. So Gemini can
phrase, select and summarise, but a figure, date, route or note it made up never reaches the user.
(Plain integers such as counts are not checked mechanically; the pack lists the counts per route and
per month so the model doesn't have to compute them.)

Off unless GEMINI_API_KEY is set. Any failure (network, quota, overload, bad reply) returns an error
and the dashboard falls back to the built-in answer. Temperature 0, low thinking; every call's tokens
are recorded in `UsageLog`. The graded pipeline never calls this module. Standard library only.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

from .assistant import answer as rule_answer
from .config import DEFAULT_CONFIG, FLAG_JUSTIFIED, FLAG_NO, FLAG_YES, Config
from .explanations import fmt_pct
from .note_parser import corridors_in_text

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # override with GEMINI_MODEL
FALLBACK_MODEL = "gemini-3.8-flash"  # used when the first model fails; override with GEMINI_FALLBACK_MODEL
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = (429, 500, 502, 503, 504)

INSTRUCTIONS = """You are the assistant inside a freight cost dashboard. Answer the user's question using ONLY the DATA below, which is this run's computed results.

Rules:
1. Every number, date, route and note id you write must appear in DATA, written the same way. Do not calculate new figures (no new averages, totals, ratios or differences). You may count rows listed in DATA; if you do, say what you counted.
2. To explain why a week was flagged or justified, use its verdict and reason from DATA and cite the note id when there is one.
3. If DATA does not contain the answer, say in one sentence that this analysis doesn't cover it and name a related question it can answer. Never use outside knowledge, never guess, never forecast.
4. Be concise: at most about 150 words, plain English, short bullet points if useful. Costs are in INR per tonne-km.
5. Speak as the dashboard. Never mention "DATA", section or field names, or these rules. Do not end with a question or an offer to help further.

DATA:
"""


@dataclass
class UsageLog:
    """Token and call counts for the honest cost log the brief asks for."""
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[dict] = field(default_factory=list)

    def record(self, question: str, usage: dict | None, ok: bool, detail: str = "") -> None:
        usage = usage or {}
        tin = int(usage.get("promptTokenCount", 0))
        tout = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
        self.calls += 1
        self.failures += 0 if ok else 1
        self.input_tokens += tin
        self.output_tokens += tout
        self.events.append({"question": question, "ok": ok, "input_tokens": tin, "output_tokens": tout,
                            "detail": detail})


@dataclass
class LLMAnswer:
    text: str | None                # the checked answer, or None
    error: str | None = None        # set when Gemini couldn't be used (network, quota, overload, ...)
    problems: list[str] = field(default_factory=list)  # set when the answer failed the data check


# ---------------------------------------------------------------- data pack
def _pct(v: float) -> str:
    return "n/a" if pd.isna(v) else fmt_pct(v)


def build_context(output: pd.DataFrame, audit: pd.DataFrame, cards: dict, cfg: Config = DEFAULT_CONFIG) -> str:
    """Everything Gemini may use, as compact text. Figures use the same rounding as the output CSV."""
    a = audit.merge(output[["route", "week_of", "reason"]], on=["route", "week_of"])
    a = a.sort_values(["route", "week_of"])
    a["cost"] = a["cost_per_tonne_km"].map(lambda v: f"{v:.2f}")
    parts = ["METHOD",
             rule_answer("How is cost per tonne-km calculated?", output, audit, cards, cfg).text,
             "Verdicts: Yes = flagged, unexplained; No (justified) = explained by a note; No = not flagged."]

    parts += ["", "ROUTES (route | type | weeks | average cost | lowest | highest | latest | "
                  "unexplained weeks | justified weeks)"]
    for route, g in a.groupby("route"):
        lo, hi, last = g.loc[g["cost_per_tonne_km"].idxmin()], g.loc[g["cost_per_tonne_km"].idxmax()], g.iloc[-1]
        parts.append(f"{route} | {g['route_type'].iloc[0]} | {len(g)} weeks, {g['week_of'].min()} to "
                     f"{g['week_of'].max()} | avg {g['cost_per_tonne_km'].mean():.2f} | lowest {lo['cost']} "
                     f"(week of {lo['week_of']}) | highest {hi['cost']} (week of {hi['week_of']}) | latest "
                     f"{last['cost']} (week of {last['week_of']}) | {int((g['flagged'] == FLAG_YES).sum())} "
                     f"unexplained | {int((g['flagged'] == FLAG_JUSTIFIED).sum())} justified")

    parts += ["", "NOTES (id | date | applies to | window used | states a cost increase | justifies | text)"]
    for c in sorted(cards.values(), key=lambda c: c.note_id):
        used = a[a["matched_note_id"] == c.note_id]
        justifies = ", ".join(f"{r.route} week of {r.week_of}" for r in used.itertuples()) or "no week"
        increase = {"increase": "yes", "no_increase": "no"}.get(c.impact, "unclear")
        parts.append(f"{c.note_id} | {c.date:%Y-%m-%d} | {c.applies_to} | {c.start:%Y-%m-%d} to "
                     f"{c.end:%Y-%m-%d} | {increase} | {justifies} | {c.text}")

    parts += ["", "FLAGGED OR JUSTIFIED WEEKS (route | week of | cost | vs own history | vs similar routes | "
                  "verdict | note | reason)"]
    for r in a[a["flagged"] != FLAG_NO].itertuples():
        parts.append(f"{r.route} | {r.week_of} | {r.cost} | {_pct(r.own_history_pct)} | {_pct(r.peer_pct)} | "
                     f"{r.flagged} | {r.matched_note_id or '-'} | {r.reason}")

    a["month"] = a["week_of"].str[:7]
    parts += ["", "MONTHLY (route | month | average weekly cost | weeks | unexplained | justified); "
                  "a week counts in the month of its Monday"]
    for (route, month), g in a.groupby(["route", "month"]):
        parts.append(f"{route} | {month} | {g['cost_per_tonne_km'].mean():.2f} | {len(g)} | "
                     f"{int((g['flagged'] == FLAG_YES).sum())} | {int((g['flagged'] == FLAG_JUSTIFIED).sum())}")

    # Week and cost only: the full weekly table with percentages was ~24k of ~34k tokens. Flagged and
    # justified weeks carry their percentages and reasons above; ordinary weeks are verdict "No".
    parts += ["", "WEEKLY COST (week of, cost) per route; every week not listed under FLAGGED OR JUSTIFIED "
                  "WEEKS has verdict No"]
    for route, g in a.groupby("route"):
        parts.append(f"{route}: " + "; ".join(f"{r.week_of} {r.cost}" for r in g.itertuples()))
    return "\n".join(parts)


# ---------------------------------------------------------------- answer check
_DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")
_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
_NOTE = re.compile(r"\bN\d{3}\b")


def check_answer(text: str, context: str, routes: list[str], note_ids: list[str]) -> list[str]:
    """Everything in `text` that does not come from the data pack. Empty list = the answer passes."""
    decimals = set(_DECIMAL.findall(context))
    allowed = decimals | {f"{float(d):.1f}" for d in decimals}  # rounding to 1 decimal is fine
    problems = [f"figure {d}" for d in dict.fromkeys(_DECIMAL.findall(text)) if d not in allowed]
    dates = set(_DATE.findall(context))
    problems += [f"date {d}" for d in dict.fromkeys(_DATE.findall(text)) if d not in dates]
    problems += [f"note {n}" for n in dict.fromkeys(_NOTE.findall(text)) if n not in note_ids]
    known = {frozenset(r.lower().split("-")) for r in routes}
    problems += [f"route {r}" for r in corridors_in_text(text) if frozenset(r.lower().split("-")) not in known]
    return problems


# ---------------------------------------------------------------- Gemini client
class GeminiAssistant:
    """Answers questions from the data pack with a backup model; checks every answer before returning it."""

    def __init__(self, api_key: str, routes: list[str], note_ids: list[str], model: str = DEFAULT_MODEL,
                 fallback_model: str | None = FALLBACK_MODEL, timeout: float = 45.0,
                 thinking_level: str | None = "low"):
        self.api_key, self.timeout, self.thinking_level = api_key, timeout, thinking_level
        self.models = [m for m in dict.fromkeys([model, fallback_model]) if m]
        self.routes, self.note_ids = sorted(routes), sorted(note_ids)
        self.usage = UsageLog()
        # Streamlit reruns the script on every widget change; never pay twice for the same question.
        self._cache: dict[tuple[str, str], LLMAnswer] = {}

    def _post(self, body: dict, model: str) -> dict:
        req = urllib.request.Request(
            ENDPOINT.format(model=model), data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def answer(self, question: str, context: str, builtin_answer: str | None = None) -> LLMAnswer:
        """Checked answer to `question` from `context` (plus the built-in answer, if the rules matched)."""
        pack = context + (f"\n\nBUILT-IN ANSWER TO THIS QUESTION (computed by fixed rules; may only partly "
                          f"fit the question)\n{builtin_answer}" if builtin_answer else "")
        key = (" ".join(question.lower().split()), builtin_answer or "")
        if key in self._cache:
            return self._cache[key]
        result = self._answer(question, pack)
        if result.error is None:  # don't remember outages: the same question may work later
            self._cache[key] = result
        return result

    def _answer(self, question: str, pack: str) -> LLMAnswer:
        body = {
            "systemInstruction": {"parts": [{"text": INSTRUCTIONS + pack}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0},
        }
        if self.thinking_level:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        error = "no model configured"
        # Each model gets two tries for overload/rate-limit errors; other errors move on to the next model.
        for model in self.models:
            for attempt in range(2):
                try:
                    data = self._post(body, model)
                    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
                    if not text:
                        raise ValueError("empty answer")
                except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, OSError) as exc:
                    error = _describe(exc)
                    log.warning("Gemini (%s) failed: %s", model, error)
                    self.usage.record(question, None, ok=False, detail=f"{model}: {error}")
                    if isinstance(exc, urllib.error.HTTPError) and exc.code in RETRYABLE and attempt == 0:
                        time.sleep(2)
                        continue
                    break
                problems = check_answer(text, pack, self.routes, self.note_ids)
                self.usage.record(question, data.get("usageMetadata"), ok=True,
                                  detail=f"{model}: " + ("checked OK" if not problems else
                                                         "failed data check: " + "; ".join(problems)))
                return LLMAnswer(text=None if problems else text, problems=problems)
        return LLMAnswer(text=None, error=error)


def _describe(exc: Exception) -> str:
    """Short reason for logs and the UI. Uses Google's error message only (the request, which carries
    the key in a header, is never included)."""
    if isinstance(exc, urllib.error.HTTPError):
        reason = {429: "rate limit or free-tier quota reached", 503: "Gemini is overloaded right now",
                  500: "Gemini server error", 502: "Gemini server error", 504: "Gemini timed out",
                  400: "request rejected", 401: "API key not accepted", 403: "API key not allowed",
                  404: "model not available"}.get(exc.code, "request failed")
        try:
            message = json.loads(exc.read().decode("utf-8"))["error"]["message"].strip()
        except Exception:
            message = ""
        return f"HTTP {exc.code}, {reason}" + (f": {message[:160]}" if message else "")
    if isinstance(exc, (TimeoutError, urllib.error.URLError)) or "timed out" in str(exc):
        return "no response from Gemini (network or timeout)"
    return f"unexpected reply ({type(exc).__name__}: {exc})"[:160]
