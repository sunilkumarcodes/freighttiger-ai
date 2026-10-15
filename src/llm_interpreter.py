"""Optional Gemini fallback for the assistant: understands wording the rules can't, never answers.

The rule-based assistant (assistant.py) always runs first. Only when it can't match a question is
Gemini asked to classify it into one of the assistant's fixed question types, with routes and note
ids restricted to the real ones (enforced by a JSON response schema and checked again here). That
structure is turned into a canonical question, which the same rule-based assistant answers from the
pipeline's results. So Gemini can misread a question, but it cannot put a fact in an answer.

Off unless GEMINI_API_KEY is set (Streamlit secret or environment variable). Any failure (no key,
network, quota, bad JSON) returns None and the assistant shows its usual help text. Temperature is 0;
every call's token counts are recorded in `UsageLog`. The graded pipeline never calls this module.

Standard library only (urllib), so no extra dependency.
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

from .assistant import MONTHS, ROUTE_TYPES

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # override with GEMINI_MODEL
FALLBACK_MODEL = "gemini-3.8-flash"  # used when the first model is overloaded; GEMINI_FALLBACK_MODEL
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

INTENTS = {
    "why_pricier": "why costs rose on routes and/or in a period (needs routes or a period)",
    "unexplained": "which routes/weeks are flagged or unexplained, or why a route is flagged",
    "justified": "which weeks were justified/explained by a note",
    "anomaly_ranking": "highest or lowest cost anomalies (set direction)",
    "cost_ranking": "most expensive or cheapest routes by cost per tonne-km (set direction)",
    "cost_lookup": "the actual cost per tonne-km for routes and/or a period (needs routes or a period)",
    "compare": "compare two or more routes",
    "counts": "how many weeks/routes were flagged or justified",
    "route_list": "which routes are monitored",
    "note_list": "which context notes exist",
    "note_lookup": "what a specific note says (needs note_ids)",
    "method": "how the numbers, baselines, thresholds or verdicts are calculated",
    "unsupported": "anything that cannot be answered from these freight cost results",
}


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


def response_schema(routes: list[str], note_ids: list[str]) -> dict:
    """Gemini structured-output schema: every field required, enums for anything that names data."""
    return {
        "type": "OBJECT",
        "properties": {
            "intent": {"type": "STRING", "enum": list(INTENTS)},
            "routes": {"type": "ARRAY", "items": {"type": "STRING", "enum": routes}},
            "route_type": {"type": "STRING", "enum": ["None", *ROUTE_TYPES]},  # Gemini rejects empty enum values
            "note_ids": {"type": "ARRAY", "items": {"type": "STRING", "enum": note_ids or ["N000"]}},
            "month": {"type": "INTEGER", "description": "1-12, or 0 if none"},
            "year": {"type": "INTEGER", "description": "e.g. 2025, or 0 if none"},
            "date": {"type": "STRING", "description": "YYYY-MM-DD if a specific day is named, else empty"},
            "latest_week": {"type": "BOOLEAN"},
            "direction": {"type": "STRING", "enum": ["highest", "lowest"]},
        },
        "required": ["intent", "routes", "route_type", "note_ids", "month", "year", "date", "latest_week",
                     "direction"],
    }


def system_prompt(routes: list[str], note_ids: list[str], first_week: str, last_week: str,
                  note_summaries: list[str] | None = None) -> str:
    kinds = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    notes = "\n".join(f"- {n}" for n in note_summaries) if note_summaries else (", ".join(note_ids) or "none")
    return (
        "You classify questions about a freight cost analysis. You never answer them.\n"
        f"Routes in the data (use these exact names; either direction counts): {', '.join(routes)}.\n"
        f"Route types: {', '.join(ROUTE_TYPES)}.\n"
        "Context notes (use them to connect events such as floods, festivals or diesel prices to routes, "
        f"notes and dates):\n{notes}\n"
        f"The data covers the weeks of {first_week} to {last_week}. Read relative dates such as "
        f"'last month' relative to {last_week}.\n"
        "Map city names to the routes that contain them. Leave a field empty, 'None' or 0 when the question "
        "does not give it. Prefer a supported question type whenever the question is about freight costs, "
        "routes, flags or these notes; use 'unsupported' only for questions clearly unrelated to this "
        "analysis (forecasts, weather, general knowledge).\n"
        f"Question types:\n{kinds}"
    )


def canonical_question(parsed: dict, routes: list[str], note_ids: list[str]) -> str | None:
    """Structured interpretation -> a question the rule-based assistant is tested to understand."""
    intent = parsed.get("intent")
    if intent not in INTENTS or intent == "unsupported":
        return None
    named = [r for r in parsed.get("routes", []) if r in routes]
    rtype = parsed.get("route_type") if parsed.get("route_type") in ROUTE_TYPES else ""
    where = " and ".join(named) if named else (f"{rtype.lower()} routes" if rtype else "")
    notes = [n for n in parsed.get("note_ids", []) if n in note_ids]

    month = parsed.get("month") or 0
    year = parsed.get("year") or 0
    date = parsed.get("date") or ""
    if date and not re.fullmatch(r"20\d\d-\d\d-\d\d", date):
        date = ""
    if date:
        try:
            pd.Timestamp(date)
        except ValueError:
            date = ""
    month = month if isinstance(month, int) and 1 <= month <= 12 else 0
    year = year if isinstance(year, int) and 2000 <= year <= 2099 else 0
    if date:
        when = f"on {date}"
    elif parsed.get("latest_week"):
        when = "in the latest week" + (f" of {year}" if year else "")
    elif month or year:
        when = "in " + " ".join(p for p in (MONTHS[month - 1].title() if month else "", str(year) if year else "") if p)
    else:
        when = ""
    on = f" on {where}" if where else ""
    tail = f"{on} {when}".rstrip()
    high = parsed.get("direction") != "lowest"

    if intent == "why_pricier":
        return f"Why did {where or 'costs'} get pricier {when}".rstrip() + "?" if (where or when) else None
    if intent == "unexplained":
        return f"Why is {where} flagged?" if where else "Which routes have unexplained increases?"
    if intent == "justified":
        return f"Which weeks were justified by a note{tail}?"
    if intent == "anomaly_ranking":
        return f"Which routes have the {'highest' if high else 'lowest'} cost anomalies{tail}?"
    if intent == "cost_ranking":
        return f"Which route is the {'most expensive' if high else 'cheapest'}{tail}?"
    if intent == "cost_lookup":
        return f"What was the cost{tail}?" if (where or when) else None
    if intent == "compare":
        return f"Compare {' and '.join(named)} {when}".rstrip() if len(named) >= 2 else None
    if intent == "counts":
        return f"How many weeks were flagged{tail}?"
    if intent == "route_list":
        return "Which routes are monitored?"
    if intent == "note_list":
        return "What notes are there?"
    if intent == "note_lookup":
        return f"What does {' and '.join(notes)} say?" if notes else None
    if intent == "method":
        return "How is cost per tonne-km calculated?"
    return None


class GeminiInterpreter:
    """Calls Gemini with a fallback model. `last_error` says why the latest question got no answer."""

    def __init__(self, api_key: str, routes: list[str], note_ids: list[str], first_week: str, last_week: str,
                 model: str = DEFAULT_MODEL, fallback_model: str | None = FALLBACK_MODEL, timeout: float = 20.0,
                 note_summaries: list[str] | None = None, thinking_level: str | None = "low"):
        # thinking_level "low" keeps answers to a few seconds (default thinking took ~25 s on gemini-3.8-flash).
        self.api_key, self.timeout, self.thinking_level = api_key, timeout, thinking_level
        self.models = [m for m in dict.fromkeys([model, fallback_model]) if m]
        self.routes, self.note_ids = sorted(routes), sorted(note_ids)
        self.system = system_prompt(self.routes, self.note_ids, first_week, last_week, note_summaries)
        self.schema = response_schema(self.routes, self.note_ids)
        self.usage = UsageLog()
        self.last_error: str | None = None
        # Streamlit reruns the script on every widget change; never pay twice for the same question.
        self._cache: dict[str, str | None] = {}

    def _post(self, body: dict, model: str) -> dict:
        req = urllib.request.Request(
            ENDPOINT.format(model=model), data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def interpret(self, question: str) -> str | None:
        """Canonical question for the rule-based assistant, or None. On None, `last_error` is set if the
        call failed (overload, quota, network) and left None if Gemini judged the question unanswerable."""
        key = " ".join(question.lower().split())
        if key in self._cache:
            self.last_error = None
            return self._cache[key]
        result = self._interpret(question)
        if self.last_error is None:  # don't remember outages: the same question may work later
            self._cache[key] = result
        return result

    def _interpret(self, question: str) -> str | None:
        body = {
            "systemInstruction": {"parts": [{"text": self.system}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": self.schema},
        }
        if self.thinking_level:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        # Each model gets two tries for overload/rate-limit errors; other errors move on to the next model.
        for model in self.models:
            for attempt in range(2):
                try:
                    data = self._post(body, model)
                    parsed = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
                except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, OSError) as exc:
                    self.last_error = _describe(exc)
                    log.warning("Gemini (%s) failed: %s", model, self.last_error)
                    self.usage.record(question, None, ok=False, detail=f"{model}: {self.last_error}")
                    if isinstance(exc, urllib.error.HTTPError) and exc.code in RETRYABLE and attempt == 0:
                        time.sleep(2)
                        continue
                    break
                self.last_error = None
                canonical = canonical_question(parsed, self.routes, self.note_ids)
                self.usage.record(question, data.get("usageMetadata"), ok=True,
                                  detail=f"{model}: {parsed.get('intent')} -> {canonical or 'no answerable question'}")
                return canonical
        return None


RETRYABLE = (429, 500, 502, 503, 504)


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
    return f"unexpected reply ({type(exc).__name__})"
