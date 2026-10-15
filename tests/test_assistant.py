from src.assistant import answer


def test_why_chennai_bangalore_march(tfidf_run):
    a = answer("Why did Chennai–Bangalore become expensive in March 2025?", tfidf_run.output)
    assert set(a.rows["route"]) == {"Chennai-Bangalore"}
    assert "N001" in a.text and "2025-03-03" in a.text
    assert "2025-03-10" in a.text and "Unexplained" in a.text  # post-flood weeks are not excused


def test_ahmedabad_mumbai(tfidf_run):
    a = answer("What explains the Ahmedabad-Mumbai increase?", tfidf_run.output)
    assert a.rows["matched_note_id"].tolist() == ["N002"]


def test_unexplained_routes(tfidf_run):
    a = answer("Which routes have unexplained increases?", tfidf_run.output)
    assert set(a.rows["route"]) == {"Chennai-Bangalore", "Delhi-Jaipur", "Mumbai-Pune"}
    assert (a.rows["flagged"] == "Yes").all()


def test_highest(tfidf_run):
    a = answer("Which routes have the highest cost anomalies?", tfidf_run.output)
    assert a.rows.iloc[0]["route"] == "Delhi-Jaipur" and a.rows.iloc[0]["week_of"] == "2024-11-11"


def test_highest_with_full_precision_audit(tfidf_run):
    a = answer("Which routes have the highest cost anomalies?", tfidf_run.output, tfidf_run.audit)
    assert a.rows.iloc[0]["route"] == "Delhi-Jaipur" and a.rows.iloc[0]["week_of"] == "2024-11-11"


def test_may_as_a_verb_is_not_a_month():
    from src.assistant import _find_period
    assert _find_period("Why may costs rise on Mumbai-Pune?") == (None, None)
    assert _find_period("What happened in May 2025?") == (5, 2025)
    assert _find_period("Any change in may?") == (5, None)


def test_period_filter_checks_month_and_year_on_the_same_day():
    import pandas as pd
    from src.assistant import _filter_period
    df = pd.DataFrame({"week_of": ["2024-12-30", "2025-12-29"]})
    assert _filter_period(df, 12, 2025)["week_of"].tolist() == ["2025-12-29"]
    assert _filter_period(df, 1, 2025)["week_of"].tolist() == ["2024-12-30"]
    assert _filter_period(df, 12, 2024)["week_of"].tolist() == ["2024-12-30"]


def test_ungroundable_question_does_not_guess(tfidf_run):
    a = answer("What will diesel cost next year?", tfidf_run.output)
    assert a.rows.empty and "only answer from this analysis" in a.text


def _ask(run, q):
    return answer(q, run.output, run.audit, run.cards)


def test_lowest_anomalies_ascending(tfidf_run):
    a = _ask(tfidf_run, "Which routes have the lowest cost anomalies?")
    assert a.text.startswith("Smallest") and len(a.rows) == 5 and (a.rows["flagged"] != "No").all()


def test_justified_weeks(tfidf_run):
    a = _ask(tfidf_run, "Which weeks were justified by a note?")
    assert sorted(a.rows["matched_note_id"]) == ["N001", "N001", "N002"]


def test_note_lookup_quotes_note_and_its_use(tfidf_run):
    a = _ask(tfidf_run, "What does note N001 say?")
    assert "Heavy flooding" in a.text and "2025-02-24 to 2025-03-08" in a.text and len(a.rows) == 2
    assert "no note with this id" in _ask(tfidf_run, "What does N042 say?").text


def test_method_question(tfidf_run):
    a = _ask(tfidf_run, "How is cost per tonne-km calculated?")
    assert "total freight cost" in a.text and "15%" in a.text and a.rows.empty


def test_why_specific_route_flagged_gives_reasons(tfidf_run):
    a = _ask(tfidf_run, "Why is Mumbai-Pune flagged?")
    assert set(a.rows["route"]) == {"Mumbai-Pune"} and "week by week" in a.text and len(a.rows) == 12


def test_cost_lookup_latest_week_of_year(tfidf_run):
    a = _ask(tfidf_run, "What was the cost on Mumbai-Delhi in the last week of 2025?")
    assert a.rows["week_of"].tolist() == ["2025-12-22"] and "INR per tonne-km" in a.text


def test_cost_lookup_by_date_and_route_type(tfidf_run):
    a = _ask(tfidf_run, "Show costs for short routes on 2025-03-05")
    assert set(a.rows["route"]) == {"Delhi-Jaipur", "Mumbai-Pune"} and set(a.rows["week_of"]) == {"2025-03-03"}


def test_compare_routes(tfidf_run):
    a = _ask(tfidf_run, "Compare Mumbai-Pune and Delhi-Jaipur")
    assert "Delhi-Jaipur: average" in a.text and "Mumbai-Pune: average" in a.text


def test_counts(tfidf_run):
    a = _ask(tfidf_run, "How many weeks were flagged in 2025?")
    assert "14 unexplained" in a.text and "3 justified" in a.text


def test_cheapest_route_ranking(tfidf_run):
    a = _ask(tfidf_run, "Which route is the cheapest?")
    assert a.text.startswith("Cheapest first") and "104 week(s)" in a.text


def test_off_topic_question_gets_help_not_a_guess(tfidf_run):
    a = _ask(tfidf_run, "Tell me a joke")
    assert a.rows.empty and "only answer from this analysis" in a.text and "N003" in a.text


def test_quiet_route_and_month(tfidf_run):
    a = answer("Anything odd on Delhi-Chennai in May 2025?", tfidf_run.output)
    assert a.rows.empty and "No flagged" in a.text
