"""Gemini fallback: no network or key needed; the HTTP call is replaced with canned responses."""
import json
import urllib.error

import pytest

from src.assistant import answer
from src.llm_interpreter import GeminiInterpreter, canonical_question, response_schema

ROUTES = ["Ahmedabad-Mumbai", "Chennai-Bangalore", "Delhi-Chennai", "Delhi-Jaipur", "Kolkata-Bhubaneswar",
          "Mumbai-Delhi", "Mumbai-Pune"]
NOTES = [f"N{i:03d}" for i in range(1, 11)]
BLANK = {"intent": "unsupported", "routes": [], "route_type": "None", "note_ids": [], "month": 0, "year": 0,
         "date": "", "latest_week": False, "direction": "highest"}


def parsed(**kw):
    return {**BLANK, **kw}


# Every structured interpretation must become a question the rule-based assistant really answers.
@pytest.mark.parametrize("p, expect_in_text", [
    (parsed(intent="why_pricier", routes=["Chennai-Bangalore"], month=3, year=2025), "N001"),
    (parsed(intent="why_pricier", month=11, year=2024), "Delhi-Jaipur"),
    (parsed(intent="unexplained"), "Routes with unexplained"),
    (parsed(intent="unexplained", routes=["Mumbai-Pune"]), "week by week"),
    (parsed(intent="justified"), "N002"),
    (parsed(intent="anomaly_ranking", direction="lowest"), "Smallest"),
    (parsed(intent="anomaly_ranking", direction="highest", year=2024), "Largest"),
    (parsed(intent="cost_ranking", direction="lowest"), "Cheapest"),
    (parsed(intent="cost_ranking", route_type="Short"), "Most expensive"),
    (parsed(intent="cost_lookup", routes=["Mumbai-Delhi"], latest_week=True, year=2025), "2025-12-22"),
    (parsed(intent="cost_lookup", route_type="Short", date="2025-03-05"), "2025-03-03"),
    (parsed(intent="compare", routes=["Mumbai-Pune", "Delhi-Jaipur"]), "Mumbai-Pune: average"),
    (parsed(intent="counts", year=2025), "14 unexplained"),
    (parsed(intent="route_list"), "7 routes"),
    (parsed(intent="note_list"), "10 context notes"),
    (parsed(intent="note_lookup", note_ids=["N003"]), "Diesel prices rose"),
    (parsed(intent="method"), "total freight cost"),
])
def test_canonical_question_round_trips_through_rules(tfidf_run, p, expect_in_text):
    q = canonical_question(p, ROUTES, NOTES)
    a = answer(q, tfidf_run.output, tfidf_run.audit, tfidf_run.cards)
    assert a.matched and expect_in_text in a.text, (q, a.text[:200])


@pytest.mark.parametrize("p", [
    parsed(intent="unsupported"),
    parsed(intent="note_lookup", note_ids=["N999"]),     # unknown note id is dropped -> nothing to ask
    parsed(intent="compare", routes=["Mumbai-Pune"]),    # needs two routes
    parsed(intent="cost_lookup"),                        # needs a route or a period
    parsed(intent="made_up_intent"),
])
def test_unanswerable_interpretations_give_none(p):
    assert canonical_question(p, ROUTES, NOTES) is None


def test_invented_route_is_dropped():
    q = canonical_question(parsed(intent="unexplained", routes=["Atlantis-Pune"]), ROUTES, NOTES)
    assert q == "Which routes have unexplained increases?"


def test_schema_restricts_routes_and_notes_to_real_ones():
    s = response_schema(ROUTES, NOTES)["properties"]
    assert s["routes"]["items"]["enum"] == ROUTES and s["note_ids"]["items"]["enum"] == NOTES


def _interpreter(monkeypatch, responder):
    g = GeminiInterpreter("test-key", ROUTES, NOTES, "2024-01-01", "2025-12-22")
    monkeypatch.setattr(g, "_post", responder)
    return g


def _reply(obj, usage=None):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}],
            "usageMetadata": usage or {"promptTokenCount": 400, "candidatesTokenCount": 40}}


def test_interpret_success_logs_tokens_and_caches(monkeypatch):
    calls = []

    def responder(body, model):
        calls.append(body)
        assert body["generationConfig"]["temperature"] == 0
        return _reply(parsed(intent="unexplained", routes=["Mumbai-Pune"]))

    g = _interpreter(monkeypatch, responder)
    assert g.interpret("whats wrong with bombay pune lately") == "Why is Mumbai-Pune flagged?"
    assert g.interpret("Whats wrong with  Bombay Pune lately") == "Why is Mumbai-Pune flagged?"
    assert len(calls) == 1  # second ask served from cache
    assert (g.usage.calls, g.usage.input_tokens, g.usage.output_tokens) == (1, 400, 40)


def _http_error(code, message="boom"):
    import io
    body = io.BytesIO(json.dumps({"error": {"message": message}}).encode())
    return urllib.error.HTTPError("u", code, "err", {}, body)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.llm_interpreter.time.sleep", lambda s: None)


@pytest.mark.parametrize("make_error, reason", [
    (lambda: _http_error(429), "quota"), (lambda: _http_error(503), "overloaded"),
    (lambda: urllib.error.URLError("offline"), "no response"), (lambda: TimeoutError(), "no response"),
])
def test_failure_on_every_model_returns_none_with_reason(monkeypatch, make_error, reason):
    tried = []

    def responder(body, model):
        tried.append(model)
        raise make_error()

    g = _interpreter(monkeypatch, responder)
    assert g.interpret("anything") is None
    assert reason in g.last_error and set(tried) == set(g.models)  # backup model was tried too
    assert "test-key" not in json.dumps(g.usage.events)
    # outages are not cached: asking again calls Gemini again
    n = len(tried)
    g.interpret("anything")
    assert len(tried) > n


def test_overloaded_model_falls_back_to_backup(monkeypatch):
    def responder(body, model):
        if model == g.models[0]:
            raise _http_error(503, "The model is overloaded.")
        return _reply(parsed(intent="note_list"))

    g = _interpreter(monkeypatch, responder)
    assert g.interpret("what notes do you have") == "What notes are there?"
    assert g.last_error is None and g.usage.failures == 2  # two tries on the first model, then the backup


def test_google_error_message_is_shown(monkeypatch):
    def responder(body, model):
        raise _http_error(403, "Method doesn't allow unregistered callers.")

    g = _interpreter(monkeypatch, responder)
    assert g.interpret("anything") is None
    assert "HTTP 403" in g.last_error and "unregistered callers" in g.last_error


def test_malformed_reply_returns_none(monkeypatch):
    g = _interpreter(monkeypatch, lambda body, model: {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})
    assert g.interpret("anything") is None and "unexpected reply" in g.last_error


def test_off_topic_question_stays_unanswered(monkeypatch):
    g = _interpreter(monkeypatch, lambda body, model: _reply(parsed(intent="unsupported")))
    assert g.interpret("What will diesel cost next year?") is None and g.last_error is None
    assert g.usage.failures == 0
