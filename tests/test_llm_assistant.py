"""Gemini assistant: no network or key needed; the HTTP call is replaced with canned responses."""
import io
import json
import re
import urllib.error

import pytest

from src.llm_assistant import GeminiAssistant, build_context, check_answer


@pytest.fixture(scope="module")
def pack(tfidf_run):
    return build_context(tfidf_run.output, tfidf_run.audit, tfidf_run.cards)


@pytest.fixture(scope="module")
def ids(tfidf_run):
    return sorted(tfidf_run.output["route"].unique()), sorted(tfidf_run.cards)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.llm_assistant.time.sleep", lambda s: None)


# ---------------------------------------------------------------- data pack
def test_pack_has_every_section_and_every_route_week(pack, tfidf_run):
    for section in ["METHOD", "ROUTES", "NOTES", "FLAGGED OR JUSTIFIED WEEKS", "MONTHLY", "WEEKLY COST"]:
        assert section in pack
    weekly = pack.split("\nWEEKLY COST")[1]
    for r in tfidf_run.output.itertuples():
        assert f"{r.week_of} {r.cost_per_tonne_km}" in weekly
    assert len(re.findall(r"\b20\d\d-\d\d-\d\d \d+\.\d\d\b", weekly)) == len(tfidf_run.output) == 728


def test_pack_figures_match_the_output_csv(pack, tfidf_run):
    o = tfidf_run.output.set_index(["route", "week_of"])
    row = o.loc[("Chennai-Bangalore", "2025-02-24")]
    assert f"Chennai-Bangalore | 2025-02-24 | {row['cost_per_tonne_km']} | +31.5% |" in pack
    assert "N001 | 2025-02-24 | Chennai-Bangalore | 2025-02-24 to 2025-03-08 | yes |" in pack
    assert "N005 | 2024-07-29 | Mumbai-Delhi" in pack and "| no | no week |" in pack


# ---------------------------------------------------------------- answer check
def test_check_accepts_figures_from_the_data(pack, ids):
    text = ("Mumbai-Pune peaked at 4.49 (week of 2025-12-22). Chennai-Bangalore rose +31.5% in the week of "
            "2025-02-24, justified by N001; Pune-Mumbai is the same corridor. Roughly 4.5 at the end.")
    assert check_answer(text, pack, *ids) == []


@pytest.mark.parametrize("text, problem", [
    ("Costs on Mumbai-Pune reached 7.77 last week.", "figure 7.77"),
    ("The spike was in the week of 2026-01-05.", "date 2026-01-05"),
    ("This was explained by note N042.", "note N042"),
    ("Delhi-Goa saw the biggest rise.", "route Delhi-Goa"),
])
def test_check_rejects_anything_not_in_the_data(pack, ids, text, problem):
    assert check_answer(text, pack, *ids) == [problem]


# ---------------------------------------------------------------- Gemini client
def _assistant(monkeypatch, ids, responder):
    g = GeminiAssistant("test-key", *ids)
    monkeypatch.setattr(g, "_post", responder)
    return g


def _reply(text, usage=None):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": usage or {"promptTokenCount": 14000, "candidatesTokenCount": 60}}


def _http_error(code, message="boom"):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(json.dumps({"error": {"message": message}}).encode()))


def test_answer_passes_check_is_returned_logged_and_cached(monkeypatch, pack, ids):
    calls = []

    def responder(body, model):
        calls.append(body)
        assert body["generationConfig"]["temperature"] == 0
        assert pack in body["systemInstruction"]["parts"][0]["text"]
        return _reply("Mumbai-Pune had 12 unexplained weeks; its highest cost was 4.49 (week of 2025-12-22).")

    g = _assistant(monkeypatch, ids, responder)
    a = g.answer("which month was worst for bombay pune?", pack)
    assert a.text and a.error is None and a.problems == []
    assert g.answer("Which month was worst for  Bombay Pune?", pack).text == a.text
    assert len(calls) == 1  # second ask served from cache
    assert (g.usage.calls, g.usage.input_tokens, g.usage.output_tokens) == (1, 14000, 60)


def test_builtin_answer_is_added_to_the_pack(monkeypatch, pack, ids):
    seen = []

    def responder(body, model):
        seen.append(body["systemInstruction"]["parts"][0]["text"])
        return _reply("See N002.")

    _assistant(monkeypatch, ids, responder).answer("What explains Ahmedabad-Mumbai?", pack, "BUILT-IN TEXT")
    assert "BUILT-IN ANSWER TO THIS QUESTION" in seen[0] and "BUILT-IN TEXT" in seen[0]


def test_invented_figure_is_withheld(monkeypatch, pack, ids):
    g = _assistant(monkeypatch, ids, lambda body, model: _reply("Diesel will cost 101.25 next year."))
    a = g.answer("What will diesel cost next year?", pack)
    assert a.text is None and a.error is None and a.problems == ["figure 101.25"]


def test_honest_refusal_passes(monkeypatch, pack, ids):
    g = _assistant(monkeypatch, ids, lambda body, model: _reply(
        "The analysis doesn't contain forecasts. You could ask which routes have unexplained increases."))
    assert g.answer("What will diesel cost next year?", pack).text.startswith("The analysis doesn't")


@pytest.mark.parametrize("make_error, reason", [
    (lambda: _http_error(429), "quota"), (lambda: _http_error(503), "overloaded"),
    (lambda: urllib.error.URLError("offline"), "no response"), (lambda: TimeoutError(), "no response"),
])
def test_failure_on_every_model_reports_the_reason(monkeypatch, pack, ids, make_error, reason):
    tried = []

    def responder(body, model):
        tried.append(model)
        raise make_error()

    g = _assistant(monkeypatch, ids, responder)
    a = g.answer("anything", pack)
    assert a.text is None and reason in a.error and set(tried) == set(g.models)
    assert "test-key" not in json.dumps(g.usage.events)
    n = len(tried)
    g.answer("anything", pack)  # outages are not cached: asking again calls Gemini again
    assert len(tried) > n


def test_overloaded_model_falls_back_to_backup(monkeypatch, pack, ids):
    def responder(body, model):
        if model == g.models[0]:
            raise _http_error(503, "The model is overloaded.")
        return _reply("Mumbai-Pune has 12 unexplained weeks.")

    g = _assistant(monkeypatch, ids, responder)
    a = g.answer("how bad is mumbai pune", pack)
    assert a.text and a.error is None and g.usage.failures == 2  # two tries on the first model


def test_rejected_key_message_is_reported(monkeypatch, pack, ids):
    g = _assistant(monkeypatch, ids, lambda body, model: (_ for _ in ()).throw(
        _http_error(400, "API key not valid. Please pass a valid API key.")))
    a = g.answer("anything", pack)
    assert "HTTP 400" in a.error and "API key not valid" in a.error


def test_empty_reply_is_an_error(monkeypatch, pack, ids):
    g = _assistant(monkeypatch, ids, lambda body, model: _reply("   "))
    a = g.answer("anything", pack)
    assert a.text is None and "unexpected reply" in a.error
