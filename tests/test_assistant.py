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


def test_ungroundable_question_does_not_guess(tfidf_run):
    a = answer("What will diesel cost next year?", tfidf_run.output)
    assert a.rows.empty and "only answer from this analysis" in a.text


def test_quiet_route_and_month(tfidf_run):
    a = answer("Anything odd on Delhi-Chennai in May 2025?", tfidf_run.output)
    assert a.rows.empty and "No flagged" in a.text
