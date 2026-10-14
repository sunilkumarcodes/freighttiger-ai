"""FreightTiger - AI Freight Cost Intelligence (Streamlit dashboard).

    streamlit run app.py

The dashboard runs the same pipeline as run.py and only displays its results; it never
recomputes or reinterprets a verdict.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.assistant import answer
from src.config import DEFAULT_CONFIG, FLAG_JUSTIFIED, FLAG_YES, OUTPUT_COLUMNS
from src.pipeline import run_pipeline, submission_csv_bytes

st.set_page_config(page_title="FreightTiger - AI Freight Cost Intelligence", layout="wide")

# Reference data-viz palette (validated: adjacent CVD ΔE 9.2, normal-vision ΔE 27.6).
SERIES = {"actual": "#2a78d6", "history": "#eb6834", "peer": "#1baf7a"}
STATUS = {"unexplained": "#d03b3b", "justified": "#0ca30c"}


@st.cache_resource(show_spinner="Running pipeline (metrics, retrieval, validation)...")
def load(retriever: str):
    return run_pipeline(cfg=DEFAULT_CONFIG, retriever=retriever)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


with st.sidebar:
    st.header("Settings")
    retriever = st.radio("Retrieval backend", ["semantic", "tfidf"],
                         help="Semantic = local sentence-transformer. Falls back to TF-IDF if the model "
                              "cannot be loaded. Verdicts are decided by deterministic validation either way.")
    run = load(retriever)
    info = run.retrieval_info
    st.caption(f"Backend in use: **{info['backend']}**  \n`{info['model']}`")
    if retriever == "semantic" and info["backend"] != "sentence-transformers":
        st.warning("Sentence-transformer model unavailable here; TF-IDF fallback is in use.")
    st.caption(f"Thresholds: ΔH ≥ {DEFAULT_CONFIG.own_history_threshold_pct:g}% or "
               f"ΔP ≥ {DEFAULT_CONFIG.peer_threshold_pct:g}%, with ΔH > 0")

out, audit = run.output, run.audit
audit = audit.assign(week=pd.to_datetime(audit["week_of"]))

st.title("FreightTiger — AI Freight Cost Intelligence")
st.caption("Weekly cost per tonne-km per route, compared with the route's own trailing 8-week average and "
           "with similar-length routes. Every justification cites a validated context note.")

# ---------------------------------------------------------------- Executive overview
c = st.columns(6)
c[0].metric("Shipments analysed", f"{run.data_quality.rows_kept:,}")
c[1].metric("Routes monitored", out["route"].nunique())
c[2].metric("Weeks processed", out["week_of"].nunique())
c[3].metric("Numeric anomalies", int(audit["numeric_anomaly"].sum()))
c[4].metric("Unexplained (flagged)", int((out["flagged"] == FLAG_YES).sum()))
c[5].metric("Justified by a note", int((out["flagged"] == FLAG_JUSTIFIED).sum()))

tab_route, tab_explain, tab_ask, tab_data = st.tabs(
    ["Route intelligence", "Anomaly explanations", "Ask the assistant", "Data & export"])

# ---------------------------------------------------------------- Route intelligence
with tab_route:
    routes = sorted(out["route"].unique())
    default = routes.index("Chennai-Bangalore") if "Chennai-Bangalore" in routes else 0
    f1, f2 = st.columns([1, 2])
    route = f1.selectbox("Route", routes, index=default)
    weeks = sorted(audit["week"].unique())
    lo, hi = f2.select_slider("Weeks", options=weeks, value=(weeks[0], weeks[-1]),
                              format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    r = audit[(audit["route"] == route) & audit["week"].between(lo, hi)]

    st.markdown(f"**{route} — weekly cost per tonne-km**")
    fig = go.Figure()
    hover = "%{x|%Y-%m-%d}<br>%{y:.3f} INR/t-km<extra>%{fullData.name}</extra>"
    fig.add_scatter(x=r["week"], y=r["cost_per_tonne_km"], name="Actual", mode="lines",
                    line=dict(color=SERIES["actual"], width=2), hovertemplate=hover)
    fig.add_scatter(x=r["week"], y=r["own_history_avg"], name="Own 8-week average (prior weeks)",
                    mode="lines", line=dict(color=SERIES["history"], width=2, dash="dash"), hovertemplate=hover)
    fig.add_scatter(x=r["week"], y=r["peer_avg"], name="Similar routes (same week)", mode="lines",
                    line=dict(color=SERIES["peer"], width=2, dash="dot"), hovertemplate=hover)
    for verdict, label, color, symbol in [(FLAG_YES, "Unexplained (flagged)", STATUS["unexplained"], "x"),
                                          (FLAG_JUSTIFIED, "Justified by note", STATUS["justified"], "circle")]:
        m = r[r["flagged"] == verdict]
        if len(m):
            fig.add_scatter(x=m["week"], y=m["cost_per_tonne_km"], name=label, mode="markers",
                            marker=dict(color=color, size=11, symbol=symbol, line=dict(width=2, color="white")),
                            customdata=m[["own_history_pct", "peer_pct", "matched_note_id"]],
                            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f} INR/t-km<br>ΔH %{customdata[0]:+.1f}% · "
                                          "ΔP %{customdata[1]:+.1f}%<br>note: %{customdata[2]}<extra>"
                                          + label + "</extra>")
    fig.update_layout(height=430, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Cost per tonne-km (INR)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.dataframe(out[(out["route"] == route) & out["week_of"].between(
        pd.Timestamp(lo).strftime("%Y-%m-%d"), pd.Timestamp(hi).strftime("%Y-%m-%d"))],
        hide_index=True, width="stretch")

# ---------------------------------------------------------------- Explanations
with tab_explain:
    an = audit[audit["numeric_anomaly"]].merge(out[["route", "week_of", "reason"]], on=["route", "week_of"])
    if an.empty:
        st.info("No numeric anomalies under the current thresholds.")
    else:
        labels = [f"{x.route} · {x.week_of} · {x.flagged}" for x in an.itertuples()]
        pick = st.selectbox("Anomaly", range(len(an)), format_func=lambda i: labels[i])
        a = an.iloc[pick]
        k = st.columns(4)
        k[0].metric("Cost per tonne-km", f"{a.cost_per_tonne_km:.2f}")
        k[1].metric("vs own history", f"{a.own_history_pct:+.1f}%",
                    help=f"Average of {a.n_prior_weeks} strictly prior week(s): {a.own_history_avg:.3f}")
        k[2].metric("vs similar routes", f"{a.peer_pct:+.1f}%",
                    help=f"Mean of {a.n_peers} other {a.route_type} route(s) this week: {a.peer_avg:.3f}")
        k[3].metric("Verdict", a.flagged)
        if a.flagged == FLAG_JUSTIFIED:
            st.success(f"Supporting note: **{a.matched_note_id}**")
        else:
            st.error("No qualifying note: flagged for human review.")
        st.markdown(f"**Reason (as written to the CSV):** {a.reason}")

        st.subheader("Retrieved candidates and validation checks")
        retrieved = {h["note_id"]: h for h in json.loads(a.retrieved)}
        for v in json.loads(a.validations):
            card = run.cards[v["note_id"]]
            h = retrieved[v["note_id"]]
            status = "ACCEPTED" if v["valid"] else "rejected"
            with st.expander(f"#{h['rank']}  {card.note_id}  (similarity {h['score']:.3f}) — {status}",
                             expanded=v["valid"]):
                st.markdown(f"*{card.date:%Y-%m-%d} · {card.applies_to}* — {card.text}")
                st.caption(f"Parsed: scope={card.scope}, window {card.start:%Y-%m-%d} → {card.end:%Y-%m-%d} "
                           f"({card.window_basis}), impact={card.impact}, magnitude={card.magnitude_pct}")
                st.dataframe(pd.DataFrame([{"check": n, "passed": "✔" if c["passed"] else "✘",
                                            "detail": c["detail"]} for n, c in v["checks"].items()]),
                             hide_index=True, width="stretch")

# ---------------------------------------------------------------- Ask
with tab_ask:
    st.caption("Answers are assembled only from this run's computed results and validated notes (no LLM). "
               "If a question can't be grounded, the assistant says so.")
    examples = ["Why did Chennai-Bangalore become expensive in March 2025?",
                "Which routes have unexplained increases?",
                "What explains the Ahmedabad-Mumbai increase?",
                "Which routes have the highest cost anomalies?"]
    ex = st.selectbox("Example questions", [""] + examples)
    q = st.text_input("Your question", value=ex)
    if q:
        res = answer(q, out, run.audit)
        st.markdown(res.text.replace("\n", "  \n"))
        if not res.rows.empty:
            st.dataframe(res.rows[OUTPUT_COLUMNS], hide_index=True, width="stretch")

# ---------------------------------------------------------------- Data & export
with tab_data:
    st.download_button("Download final_output.csv (submission format)", submission_csv_bytes(out),
                       "final_output.csv", "text/csv")
    st.download_button("Download full audit (full-precision numbers + evidence trail)",
                       csv_bytes(run.audit), "route_week_audit.csv", "text/csv")
    only = st.checkbox("Only flagged / justified rows", value=True)
    st.dataframe(out[out["flagged"] != "No"] if only else out, hide_index=True, width="stretch")
    with st.expander("Data quality report"):
        st.json(run.data_quality.to_dict())
