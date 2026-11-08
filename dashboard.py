import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(__file__).parent / "data" / "evaluations.sqlite3"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Model Regression Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("Model Regression Monitor")
st.caption("AI model performance, evaluation history and regression detection")


@st.cache_data(ttl=5)
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        runs = pd.read_sql_query(
            "SELECT * FROM runs ORDER BY created_at DESC",
            conn,
        )
        results = pd.read_sql_query(
            "SELECT * FROM results",
            conn,
        )
    return runs, results


def trigger_evaluation(save_baseline=False):
    payload = {"save_as_baseline": save_baseline}

    try:
        response = httpx.post(
            f"{API_URL}/evaluate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"Evaluation request failed: {exc}")
        return None


# Evaluation controls
st.subheader("Evaluation controls")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run evaluation", type="primary", use_container_width=True):
        with st.spinner("Evaluating all 24 test cases..."):
            result = trigger_evaluation(False)

        if result:
            st.success("Evaluation completed.")
            st.json({
                "run_id": result.get("run_id"),
                "model": result.get("model"),
                "metrics": result.get("metrics"),
                "regression": result.get("regression"),
            })
            st.cache_data.clear()
            st.rerun()

with col2:
    if st.button("Run and save new baseline", use_container_width=True):
        with st.spinner("Running evaluation and saving baseline..."):
            result = trigger_evaluation(True)

        if result:
            st.success("New baseline saved.")
            st.json({
                "run_id": result.get("run_id"),
                "model": result.get("model"),
                "metrics": result.get("metrics"),
                "saved_as_baseline": result.get("saved_as_baseline"),
            })
            st.cache_data.clear()
            st.rerun()


runs, results = load_data()

if runs.empty:
    st.warning("No evaluation runs found.")
    st.stop()

latest = runs.iloc[0]

st.divider()
st.subheader("Latest evaluation")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{latest['accuracy']:.2%}")
c2.metric("Precision", f"{latest['precision_score']:.2%}")
c3.metric("Recall", f"{latest['recall_score']:.2%}")
c4.metric("F1 Score", f"{latest['f1_score']:.2%}")

st.divider()

baselines = runs[runs["is_baseline"] == 1]

if not baselines.empty:
    baseline = baselines.iloc[0]

    st.subheader("Baseline comparison")

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision_score"),
        ("Recall", "recall_score"),
        ("F1 Score", "f1_score"),
    ]

    cols = st.columns(4)

    for col, (label, key) in zip(cols, metrics):
        delta = float(latest[key] - baseline[key])
        col.metric(
            label,
            f"{latest[key]:.2%}",
            delta=f"{delta:+.2%}",
        )

    current_results = results[results["run_id"] == latest["id"]]
    baseline_results = results[results["run_id"] == baseline["id"]]

    comparison = baseline_results.merge(
        current_results,
        on="case_id",
        suffixes=("_baseline", "_current"),
    )

    regressions = comparison[
        (comparison["correct_baseline"] == 1)
        & (comparison["correct_current"] == 0)
    ]

    st.subheader("Regression alerts")

    if regressions.empty:
        st.success("No previously correct test cases have regressed.")
    else:
        st.error(f"{len(regressions)} regression(s) detected.")

        st.dataframe(
            regressions[
                [
                    "case_id",
                    "input_text_current",
                    "expected_category_current",
                    "predicted_category_baseline",
                    "predicted_category_current",
                ]
            ].rename(
                columns={
                    "case_id": "Test Case",
                    "input_text_current": "Input",
                    "expected_category_current": "Expected",
                    "predicted_category_baseline": "Baseline Prediction",
                    "predicted_category_current": "Current Prediction",
                }
            ),
            use_container_width=True,
        )
else:
    st.warning("No baseline has been saved yet.")

st.divider()
st.subheader("Model performance history")

chart_data = runs.sort_values("created_at").melt(
    id_vars=["created_at"],
    value_vars=[
        "accuracy",
        "precision_score",
        "recall_score",
        "f1_score",
    ],
    var_name="Metric",
    value_name="Score",
)

fig = px.line(
    chart_data,
    x="created_at",
    y="Score",
    color="Metric",
    markers=True,
    title="Performance across evaluation runs",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Evaluation history")

st.dataframe(
    runs[
        [
            "id",
            "model",
            "created_at",
            "accuracy",
            "precision_score",
            "recall_score",
            "f1_score",
            "is_baseline",
            "case_count",
        ]
    ],
    use_container_width=True,
)

st.subheader("Individual test results")

selected_run = st.selectbox(
    "Choose an evaluation run",
    runs["id"].tolist(),
)

selected_results = results[results["run_id"] == selected_run]

st.dataframe(
    selected_results[
        [
            "case_id",
            "input_text",
            "expected_category",
            "predicted_category",
            "correct",
            "error",
        ]
    ],
    use_container_width=True,
)
