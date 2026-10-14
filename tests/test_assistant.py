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


def test_quiet_route_and_month(tfidf_run):
    a = answer("Anything odd on Delhi-Chennai in May 2025?", tfidf_run.output)
    assert a.rows.empty and "No flagged" in a.text
