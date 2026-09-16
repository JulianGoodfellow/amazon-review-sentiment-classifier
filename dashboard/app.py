"""THE DASHBOARD GENERATOR (assignment Steps 3, 4, 7).

Streamlit app presenting the sentiment/emotion classification results:
headline accuracy, a confusion matrix, an explorable review table, an
LLM-vs-word-list emotion comparison, and descriptive/prediction charts.

Run with: streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_reviews  # noqa: E402

RESULTS_DIR = ROOT / "results"

CLASS_COLORS = {"POSITIVE": "#4C7A6B", "NEUTRAL": "#B08D57", "NEGATIVE": "#A6453D"}
ACCENT = "#C2703D"

st.set_page_config(page_title="Gift Card Review Sentiment", page_icon="🎁", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --accent: #C2703D;
        --bg: #FAF7F2;
        --card-bg: #FFFFFF;
        --text: #20242C;
        --muted: #6B6558;
        --border: #E7DFD0;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
    h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }
    .block-container { padding-top: 2.2rem; max-width: 1180px; }
    [data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: var(--muted); font-size: 0.82rem;
        white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        font-size: 1.55rem !important;
        white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
    }
    .subtitle { color: var(--muted); font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.4rem; }
    .section-gap { margin-top: 2.2rem; }
    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

RUNS = {
    "Balanced 3-class — Step 6 (n=150, seed=42)": RESULTS_DIR / "run_3class_balanced150.json",
    "Imbalanced 2-class — Step 2 (first 100 rows)": RESULTS_DIR / "run_2class_first100.json",
}


@st.cache_data
def load_run(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text())


@st.cache_data(show_spinner="Loading full dataset for descriptive stats...")
def load_rating_distribution() -> pd.Series:
    df = load_reviews()
    return df["rating"].value_counts().sort_index()


def class_scale(labels):
    return alt.Scale(domain=labels, range=[CLASS_COLORS[l] for l in labels])


# ---------- Sidebar ----------
st.sidebar.markdown("### Run")
run_name = st.sidebar.selectbox("Results to display", list(RUNS.keys()), index=0)
run = load_run(str(RUNS[run_name]))
labels = run["labels"]
preds = pd.DataFrame(run["predictions"])
preds["correct"] = preds["true_label"] == preds["predicted_label"]

with st.sidebar.expander("About this run", expanded=False):
    st.write(f"**Model:** {run['model']}")
    st.write(f"**Classes:** {run['n_classes']}-class")
    if run.get("seed") is not None:
        st.write(f"**Sampling:** {run['n_per_class']}/class, seed={run['seed']}")
    else:
        st.write("**Sampling:** first N rows, file order")
    st.write(f"**Emotion detection:** {'yes' if run['with_emotion'] else 'no'}")

with st.sidebar.expander("Data source", expanded=False):
    st.markdown(
        "Amazon Reviews '23 — Gift Cards category, "
        "[McAuley Lab, UC San Diego](https://amazon-reviews-2023.github.io). "
        "Emotion word list: NRC Word-Emotion Association Lexicon "
        "(Mohammad & Turney, 2013)."
    )

# ---------- Header ----------
st.title("Gift Card Review Sentiment & Emotion")
st.markdown(
    f'<div class="subtitle">{run["n_classes"]}-class sentiment vs. star rating, '
    f'two independent emotion detectors, checked against {run["metrics"]["n"]} reviews.</div>',
    unsafe_allow_html=True,
)

# ---------- Headline metrics ----------
metrics = run["metrics"]
cols = st.columns(len(labels) + 1)
cols[0].metric("Overall accuracy", f"{metrics['accuracy']:.1%}", help="Predicted label matches the rating-derived label")
for c, label in zip(cols[1:], labels):
    pc = metrics["per_class"][label]
    cols[cols.index(c)].metric(f"{label.title()} recall", f"{pc['recall']:.0%}", help=f"support={pc['support']}")

# ---------- Confusion matrix ----------
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.subheader("Confusion matrix")
st.caption("Rows = true label (from rating). Columns = model prediction. The model never sees the rating.")

conf_rows = []
for true_label in labels:
    for pred_label in labels:
        conf_rows.append({
            "True label": true_label,
            "Predicted label": pred_label,
            "Count": metrics["confusion_matrix"][true_label][pred_label],
        })
conf_df = pd.DataFrame(conf_rows)

heatmap = alt.Chart(conf_df).mark_rect().encode(
    x=alt.X("Predicted label:N", sort=labels, title="Predicted"),
    y=alt.Y("True label:N", sort=labels, title="Actual"),
    color=alt.Color("Count:Q", scale=alt.Scale(scheme="oranges"), legend=None),
    tooltip=["True label", "Predicted label", "Count"],
).properties(height=220)
text = alt.Chart(conf_df).mark_text(fontSize=15, fontWeight="bold").encode(
    x=alt.X("Predicted label:N", sort=labels),
    y=alt.Y("True label:N", sort=labels),
    text="Count:Q",
    color=alt.condition(alt.datum.Count > conf_df["Count"].max() / 2, alt.value("white"), alt.value("#20242C")),
)
st.altair_chart(heatmap + text, use_container_width=True)

# ---------- Descriptive + prediction visualizations (Step 7) ----------
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.subheader("Descriptive & prediction breakdown")

d1, d2 = st.columns(2)

with d1:
    st.markdown("**Star-rating distribution (full dataset)**")
    rating_dist = load_rating_distribution().reset_index()
    rating_dist.columns = ["rating", "count"]
    rating_dist["rating"] = rating_dist["rating"].astype(str)
    chart = alt.Chart(rating_dist).mark_bar(color=ACCENT).encode(
        x=alt.X("rating:N", title="Star rating"),
        y=alt.Y("count:Q", title="Reviews"),
        tooltip=["rating", "count"],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    total = int(rating_dist["count"].sum())
    five_star_pct = rating_dist.loc[rating_dist["rating"] == "5.0", "count"].sum() / total
    st.caption(f"{total:,} total reviews — {five_star_pct:.0%} are 5-star. The scored sample above is a small, controlled subset of this.")

with d2:
    st.markdown("**Actual vs. predicted counts (this run)**")
    actual_counts = preds["true_label"].value_counts().reindex(labels, fill_value=0)
    pred_counts = preds["predicted_label"].value_counts().reindex(labels, fill_value=0)
    cmp_df = pd.concat([
        pd.DataFrame({"label": labels, "count": actual_counts.values, "source": "Actual (rating)"}),
        pd.DataFrame({"label": labels, "count": pred_counts.values, "source": "Predicted (model)"}),
    ])
    chart = alt.Chart(cmp_df).mark_bar().encode(
        x=alt.X("label:N", title=None, sort=labels),
        y=alt.Y("count:Q", title="Reviews"),
        color=alt.Color("source:N", title=None, scale=alt.Scale(range=[ACCENT, "#20242C"])),
        xOffset="source:N",
        tooltip=["label", "source", "count"],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

st.markdown("**Per-class accuracy (recall)**")
acc_df = pd.DataFrame([
    {"label": label, "recall": metrics["per_class"][label]["recall"], "support": metrics["per_class"][label]["support"]}
    for label in labels
])
chart = alt.Chart(acc_df).mark_bar().encode(
    x=alt.X("label:N", title=None, sort=labels),
    y=alt.Y("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("label:N", scale=class_scale(labels), legend=None),
    tooltip=["label", alt.Tooltip("recall:Q", format=".0%"), "support"],
).properties(height=220)
st.altair_chart(chart, use_container_width=True)

# ---------- Emotion comparison (Step 5) ----------
if run["with_emotion"]:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.subheader("Primary emotion: LLM vs. word list")
    agree = (preds["llm_primary_emotion"] == preds["lexicon_primary_emotion"]).mean()
    e1, e2 = st.columns([1, 2])
    with e1:
        st.metric("Agreement rate", f"{agree:.0%}", help="How often the two independent emotion methods pick the same primary emotion")
    with e2:
        emo_counts = pd.concat([
            preds["llm_primary_emotion"].value_counts().rename("LLM"),
            preds["lexicon_primary_emotion"].value_counts().rename("Word list"),
        ], axis=1).fillna(0).astype(int)
        emo_long = emo_counts.reset_index().melt(id_vars="index", var_name="source", value_name="count")
        emo_long.columns = ["emotion", "source", "count"]
        chart = alt.Chart(emo_long).mark_bar().encode(
            x=alt.X("emotion:N", title=None, sort="-y"),
            y=alt.Y("count:Q", title="Reviews"),
            color=alt.Color("source:N", title=None, scale=alt.Scale(range=[ACCENT, "#20242C"])),
            xOffset="source:N",
            tooltip=["emotion", "source", "count"],
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)

# ---------- Interactive review table (Step 4) ----------
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.subheader("Explore the reviews")

f1, f2, f3 = st.columns([1.2, 1.4, 1.4])
with f1:
    correctness = st.radio("Show", ["All", "Correct only", "Mismatched only"], horizontal=False)
with f2:
    true_filter = st.multiselect("True label", labels, default=labels)
with f3:
    pred_filter = st.multiselect("Predicted label", labels, default=labels)

filtered = preds[preds["true_label"].isin(true_filter) & preds["predicted_label"].isin(pred_filter)]
if correctness == "Correct only":
    filtered = filtered[filtered["correct"]]
elif correctness == "Mismatched only":
    filtered = filtered[~filtered["correct"]]

st.caption(f"Showing **{len(filtered)}** of **{len(preds)}** reviews")

display_cols = ["rating", "true_label", "predicted_label", "correct", "title", "text", "rationale"]
if run["with_emotion"]:
    display_cols += ["llm_primary_emotion", "lexicon_primary_emotion"]

st.dataframe(
    filtered[display_cols].rename(columns={
        "true_label": "true", "predicted_label": "predicted",
        "llm_primary_emotion": "LLM emotion", "lexicon_primary_emotion": "word-list emotion",
    }),
    use_container_width=True,
    height=420,
)
