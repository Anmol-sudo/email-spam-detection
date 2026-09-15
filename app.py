"""
Spam Mail Detector — Streamlit front-end (Phase 8).

This file is deliberately UI-only. Every piece of model logic — loading the
pickle, cleaning text the same way Phase 2 did, predicting and explaining —
lives in `backend.py`, which this module imports. See
`documentation/streamlit_app_guide.md` for how the two halves fit together and
how to deploy this on Streamlit Community Cloud.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import backend
from backend import DEFAULT_THRESHOLD, ModelNotFoundError, SAMPLE_MESSAGES

# ── Page config — must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Test-set results measured on the 1,115 held-out messages (see
# documentation/full_project_explanation.md, Phase 4).
TEST_METRICS = {
    "Accuracy": 98.03,
    "ROC-AUC": 99.30,
    "Spam precision": 97.84,
    "Spam recall": 87.74,
    "Spam F1": 92.52,
}

# How the decision threshold trades false positives against missed spam,
# measured on the same held-out set.
THRESHOLD_TABLE = pd.DataFrame(
    [
        (0.20, 97.85, 94.19, 90.68, 15, 9),
        (0.30, 97.67, 90.97, 92.16, 12, 14),
        (0.35, 98.03, 90.32, 95.24, 7, 15),
        (0.40, 98.03, 89.68, 95.86, 6, 16),
        (0.50, 98.03, 87.74, 97.84, 3, 19),
        (0.60, 97.85, 86.45, 97.81, 3, 21),
        (0.70, 97.22, 81.29, 98.44, 2, 29),
    ],
    columns=[
        "Threshold", "Accuracy %", "Spam recall %", "Spam precision %",
        "Good mail flagged", "Spam missed",
    ],
)

THRESHOLD_PRESETS = {
    "Aggressive — catch the most spam (0.20)": 0.20,
    "Balanced — best spam F1 (0.35)": 0.35,
    "Default — fewest false alarms (0.50)": DEFAULT_THRESHOLD,
    "Cautious — only obvious spam (0.70)": 0.70,
}


# ══════════════════════════════════════════════════════════════════════════════
# Resource loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading the trained model…")
def load_predictor():
    """Load the pipeline once per server process, not once per interaction."""
    return backend.get_predictor()


try:
    predictor = load_predictor()
except ModelNotFoundError as exc:
    st.error("### 🚫 Model file not found")
    st.code(str(exc))
    st.markdown(
        "Run **Phase 7** of `Spam_Mail_Prediction_using_Machine_Learning.ipynb` "
        "to regenerate `spam_model_pipeline.pkl`, then place it in `saved_model/`."
    )
    st.stop()
except Exception as exc:  # corrupt pickle, incompatible scikit-learn, ...
    st.error("### 🚫 The model could not be loaded")
    st.exception(exc)
    st.stop()

INFO = predictor.info()


# ══════════════════════════════════════════════════════════════════════════════
# Styling — token-based so it reads correctly in both light and dark themes
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
  :root { --spam: #e03131; --ham: #2f9e44; --muted: rgba(128,128,128,.18); }
  .verdict {
      border-radius: 14px; padding: 22px 26px; color: #fff;
      display: flex; align-items: center; justify-content: space-between;
      gap: 18px; flex-wrap: wrap;
  }
  .verdict h2 { margin: 0; font-size: 1.7rem; color: #fff; }
  .verdict p  { margin: 4px 0 0; opacity: .92; font-size: .95rem; }
  .verdict .score { font-size: 2.4rem; font-weight: 700; line-height: 1; }
  .verdict-spam { background: linear-gradient(135deg, #f03e3e, #c92a2a); }
  .verdict-ham  { background: linear-gradient(135deg, #37b24d, #2b8a3e); }

  .term-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
  .term-name { width: 140px; text-align: right; font-weight: 600;
               font-size: .88rem; overflow: hidden; text-overflow: ellipsis;
               white-space: nowrap; }
  .term-track { flex: 1; background: var(--muted); border-radius: 5px;
                height: 15px; position: relative; }
  .term-fill { position: absolute; top: 0; height: 15px; border-radius: 5px; }
  .term-val { width: 68px; font-size: .8rem; opacity: .75;
              font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 3px 11px; border-radius: 999px;
          font-size: .78rem; font-weight: 600; margin: 3px 4px 3px 0;
          border: 1px solid var(--muted); }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — controls and model status
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Detector settings")

    preset = st.selectbox(
        "Sensitivity preset",
        list(THRESHOLD_PRESETS),
        index=2,
        help="Presets move the decision threshold applied to P(spam).",
    )
    threshold = st.slider(
        "Decision threshold on P(spam)",
        min_value=0.05, max_value=0.95,
        value=THRESHOLD_PRESETS[preset], step=0.05,
        help="A message is flagged as spam when P(spam) is at or above this value. "
             "Lower = catches more spam but flags more good mail.",
    )
    st.caption(
        f"Flag as spam when **P(spam) ≥ {threshold:.2f}**. "
        "At 0.50 the model misses ~12% of spam but only mislabels 3 in 960 good "
        "messages — see the *Model insights* tab."
    )

    st.divider()
    st.markdown("## 📦 Model status")
    st.markdown(
        f"""
- **Estimator:** {INFO['estimator']} (C = {INFO['C']}, {INFO['penalty']})
- **Features:** {INFO['vocabulary_size']:,} TF-IDF terms, n-grams {INFO['ngram_range']}
- **scikit-learn:** {INFO['sklearn_running']}
- **Text cleaning:** {'NLTK (as trained)' if INFO['nlp_backend'] == 'nltk' else 'built-in fallback'}
"""
    )
    if INFO["version_mismatch"]:
        st.warning(
            f"Model was pickled with scikit-learn {INFO['sklearn_trained']} but "
            f"{INFO['sklearn_running']} is running. Pin "
            f"`scikit-learn=={INFO['sklearn_trained']}` in requirements.txt to "
            "guarantee identical predictions."
        )
    if INFO["nlp_backend"] != "nltk":
        st.warning(
            "NLTK corpora could not be loaded, so an approximate cleaner is in "
            "use. Predictions still work but may differ slightly from training."
        )

    st.divider()
    st.markdown(
        """
### ℹ️ About
Classifies SMS/email text as **Spam** or **Ham** (legitimate).

- **Pipeline:** NLTK cleaning → TF-IDF bigrams → Logistic Regression
- **Trained on:** SMS Spam Collection, 5,572 messages
- **Built by:** Arjun · Internship Project 2026

[Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
"""
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_verdict(result) -> None:
    """Big coloured result banner."""
    css_class = "verdict-spam" if result.is_spam else "verdict-ham"
    icon = "🚨" if result.is_spam else "✅"
    headline = "SPAM DETECTED" if result.is_spam else "Legitimate message (Ham)"
    subtitle = (
        f"P(spam) = {result.spam_prob:.1%} · threshold {result.threshold:.2f}"
    )
    st.markdown(
        f"""
<div class="verdict {css_class}">
  <div><h2>{icon} {headline}</h2><p>{subtitle}</p></div>
  <div style="text-align:right">
    <div class="score">{result.confidence:.0%}</div>
    <p>confidence</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_contributions(contributions) -> None:
    """Horizontal bars showing each term's push toward spam or ham."""
    if not contributions:
        st.info(
            "None of the words in this message are in the model's vocabulary, "
            "so the prediction falls back to the model's baseline (intercept)."
        )
        return

    largest = max(abs(c.contribution) for c in contributions) or 1.0
    rows = []
    for contribution in contributions:
        width = abs(contribution.contribution) / largest * 50  # % of track
        toward_spam = contribution.pushes_toward == "spam"
        color = "var(--spam)" if toward_spam else "var(--ham)"
        # Spam pushes grow left of centre, ham pushes grow right.
        left = 50 - width if toward_spam else 50
        rows.append(
            f'<div class="term-row">'
            f'<div class="term-name">{contribution.term}</div>'
            f'<div class="term-track">'
            f'<div class="term-fill" style="left:{left}%;width:{width}%;'
            f'background:{color}"></div></div>'
            f'<div class="term-val">{contribution.contribution:+.3f}</div>'
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption(
        "🔴 left = pushes toward **spam** · 🟢 right = pushes toward **ham**. "
        "Bar length is the term's TF-IDF weight × its model coefficient — the "
        "actual amount it moved this decision."
    )


def render_weight_bars(terms, color: str) -> None:
    """Ranked left-to-right bars for a list of (term, coefficient) pairs."""
    largest = max(abs(weight) for _, weight in terms) or 1.0
    rows = [
        f'<div class="term-row">'
        f'<div class="term-name" style="width:110px">{term}</div>'
        f'<div class="term-track">'
        f'<div class="term-fill" style="left:0;width:{abs(weight) / largest * 100:.1f}%;'
        f'background:{color}"></div></div>'
        f'<div class="term-val">{weight:+.2f}</div>'
        f"</div>"
        for term, weight in terms
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


def set_sample(text: str) -> None:
    """Button callback: fill the input box before the next rerun draws it."""
    st.session_state.message_input = text


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.title("📧 Spam Mail Detector")
st.markdown(
    "Paste any SMS or email text and the model tells you whether it is **spam** "
    "or **ham** — with the words that drove the decision."
)

tab_classify, tab_batch, tab_insights, tab_how = st.tabs(
    ["🔍 Classify", "📑 Batch check", "📊 Model insights", "🧠 How it works"]
)


# ── Tab 1: single message ────────────────────────────────────────────────────
with tab_classify:
    left, right = st.columns([3, 1], gap="large")

    with right:
        st.markdown("#### Try an example")
        for name, sample in SAMPLE_MESSAGES.items():
            st.button(
                name, key=f"sample_{name}", use_container_width=True,
                on_click=set_sample, args=(sample,),
            )

    with left:
        message = st.text_area(
            "Message text",
            key="message_input",
            height=170,
            placeholder="e.g.  FREE entry! Win £1000 cash prize. Text WIN to 87121 now…",
        )
        classify = st.button(
            "🔍 Classify message", type="primary", use_container_width=True
        )

    if classify and not message.strip():
        st.warning("Enter a message first, or pick one of the examples.")
    elif message.strip():
        result = predictor.predict(message, threshold=threshold)

        st.divider()
        render_verdict(result)

        if result.is_borderline:
            st.info(
                f"⚖️ **Borderline call.** P(spam) = {result.spam_prob:.1%} sits "
                f"close to the {result.threshold:.2f} threshold — nudge the "
                "sensitivity slider in the sidebar to see the verdict flip."
            )

        st.markdown("")
        prob_col, stats_col = st.columns([1, 1], gap="large")

        with prob_col:
            st.markdown("#### 📊 Probability breakdown")
            st.metric("🚨 Spam probability", f"{result.spam_prob:.1%}")
            st.progress(result.spam_prob)
            st.metric("✅ Ham probability", f"{result.ham_prob:.1%}")
            st.progress(result.ham_prob)

        with stats_col:
            st.markdown("#### 📋 Message signals")
            features = result.features
            a, b = st.columns(2)
            a.metric("Characters", features["char_count"])
            b.metric("Words", features["word_count"])
            a.metric("UPPERCASE", f"{features['uppercase_ratio']:.0%}")
            b.metric("Digits", features["digit_count"])
            flags = []
            if features["has_url"]:
                flags.append("🔗 contains a link")
            if features["has_currency_or_prize"]:
                flags.append("💰 money / prize wording")
            if features["has_long_number"]:
                flags.append("📞 long number (shortcode?)")
            if features["exclamation_count"] >= 2:
                flags.append(f"❗ {features['exclamation_count']} exclamation marks")
            st.markdown(
                "".join(f'<span class="pill">{f}</span>' for f in flags)
                or '<span class="pill">no classic spam signals</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "These are the Phase-1 EDA signals, shown for context. The model "
                "itself only reads the TF-IDF terms below."
            )

        st.markdown("#### 🔑 Words that drove this prediction")
        render_contributions(predictor.explain(message, top_n=12))

        with st.expander("🔬 What the model actually saw"):
            st.markdown("**After Phase-2 cleaning:**")
            st.code(result.cleaned or "(nothing left after cleaning)", language="text")
            unknown = predictor.unknown_tokens(message)
            if unknown:
                st.markdown(
                    "**Out-of-vocabulary tokens** (not in the 20,000-term TF-IDF "
                    "vocabulary, so they carry zero weight):"
                )
                st.markdown(
                    " ".join(f'<span class="pill">{t}</span>' for t in unknown[:30]),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Every cleaned token is in the model's vocabulary.")

        # Keep a short running log so several messages can be compared at once.
        history = st.session_state.setdefault("history", [])
        row = result.as_row()
        if not history or history[-1] != row:
            history.append(row)
            del history[:-10]

    if st.session_state.get("history"):
        with st.expander(f"🕑 Session history ({len(st.session_state['history'])})"):
            st.dataframe(
                pd.DataFrame(st.session_state["history"][::-1]),
                use_container_width=True, hide_index=True,
            )
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()


# ── Tab 2: batch ─────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### Check many messages at once")
    st.caption(
        "Paste one message per line, or upload a CSV and pick the text column. "
        "Handy for scoring a whole export in one pass."
    )

    source = st.radio(
        "Where are the messages coming from?",
        ["✍️ Paste text", "📄 Upload CSV"],
        horizontal=True, label_visibility="collapsed",
    )

    messages: list[str] = []

    if source == "✍️ Paste text":
        # A form commits the textarea contents together with the click, so the
        # run button works on the first press instead of needing a blur first.
        with st.form("batch_paste_form"):
            pasted = st.text_area(
                "One message per line", height=200,
                placeholder="Win a free iPhone now!\nLunch at 1pm?\n…",
            )
            run_paste = st.form_submit_button(
                "▶️ Run batch classification", type="primary",
                use_container_width=True,
            )
        if run_paste:
            messages = [line for line in pasted.splitlines() if line.strip()]
            if not messages:
                st.warning("Paste at least one message first.")

    else:
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded, encoding="latin-1")
            except Exception as exc:
                st.error(f"Could not read that CSV: {exc}")
                frame = None

            if frame is not None and not frame.empty:
                # Pre-select the most likely text column so one click usually does it.
                columns = list(frame.columns)
                guesses = [c for c in columns
                           if str(c).lower() in {"message", "text", "body", "v2", "sms"}]
                column = st.selectbox(
                    "Which column holds the message text?",
                    columns, index=columns.index(guesses[0]) if guesses else 0,
                )
                st.caption(f"{len(frame):,} rows loaded. Preview:")
                st.dataframe(frame[[column]].head(3), use_container_width=True,
                             hide_index=True)
                if st.button("▶️ Run batch classification", type="primary",
                             use_container_width=True):
                    messages = frame[column].dropna().astype(str).tolist()
            elif frame is not None:
                st.warning("That CSV is empty.")

    if messages:
        with st.spinner(f"Classifying {len(messages):,} messages…"):
            results = predictor.predict_many(messages, threshold=threshold)
        frame = pd.DataFrame(r.as_row() for r in results)

        spam_count = int((frame["prediction"] == "Spam").sum())
        a, b, c = st.columns(3)
        a.metric("Messages", f"{len(frame):,}")
        b.metric("Flagged spam", f"{spam_count:,}")
        c.metric("Spam rate", f"{spam_count / len(frame):.1%}")

        st.dataframe(
            frame,
            use_container_width=True, hide_index=True,
            column_config={
                "message": st.column_config.TextColumn("Message", width="large"),
                "prediction": st.column_config.TextColumn("Verdict", width="small"),
                "spam_probability": st.column_config.ProgressColumn(
                    "P(spam)", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                # Both are implied by P(spam); hide them here but keep them in
                # the CSV download, where they are useful for analysis.
                "ham_probability": None,
                "confidence": None,
            },
        )

        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        st.download_button(
            "⬇️ Download results as CSV", buffer.getvalue(),
            file_name="spam_predictions.csv", mime="text/csv",
        )


# ── Tab 3: model insights ────────────────────────────────────────────────────
with tab_insights:
    st.markdown("### Held-out test-set performance")
    st.caption(
        "1,115 messages the model never saw during training "
        "(20% split, `random_state=3`)."
    )
    metric_cols = st.columns(len(TEST_METRICS))
    for column, (name, value) in zip(metric_cols, TEST_METRICS.items()):
        column.metric(name, f"{value:.2f}%")

    st.info(
        "**Why spam recall (87.7%) is lower than accuracy (98.0%):** only 13% of "
        "the dataset is spam, so accuracy is dominated by the easy ham majority. "
        "Spam recall is the number that actually matters for a filter — and it is "
        "what the threshold slider buys you."
    )

    st.markdown("### Threshold trade-off")
    st.caption(
        "Measured on the same held-out set. Moving the threshold down catches "
        "more spam but sends more good mail to the spam folder."
    )
    st.dataframe(
        THRESHOLD_TABLE.style.format({
            "Threshold": "{:.2f}", "Accuracy %": "{:.2f}",
            "Spam recall %": "{:.2f}", "Spam precision %": "{:.2f}",
        }).apply(
            lambda row: [
                "font-weight:700" if abs(row["Threshold"] - threshold) < 0.026 else ""
            ] * len(row),
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### What the model learned overall")
    spam_terms, ham_terms = predictor.global_top_terms(n=20)
    spam_col, ham_col = st.columns(2, gap="large")
    with spam_col:
        st.markdown("#### 🚨 Strongest spam indicators")
        render_weight_bars(spam_terms, "var(--spam)")
    with ham_col:
        st.markdown("#### ✅ Strongest ham indicators")
        render_weight_bars(ham_terms, "var(--ham)")
    st.caption(
        "These are the Logistic Regression coefficients — the model's global "
        "view, independent of any single message."
    )


# ── Tab 4: how it works ──────────────────────────────────────────────────────
with tab_how:
    st.markdown(
        f"""
### From raw text to a verdict

```
raw message
   ↓  backend.clean_text()        lowercase → strip URLs → strip digits
   ↓                              → strip punctuation → drop stopwords
   ↓                              → lemmatize          (Phase 2)
cleaned text
   ↓  TfidfVectorizer             {INFO['vocabulary_size']:,} terms, n-grams {INFO['ngram_range']}
   ↓                              (fitted during Phase 5 tuning)
sparse vector
   ↓  LogisticRegression          C = {INFO['C']}, penalty = {INFO['penalty']}
   ↓                              score = intercept + Σ (tfidf · coefficient)
P(spam), P(ham)
   ↓  threshold ({threshold:.2f})
verdict
```

**Why the two files are separate**

| File | Responsibility |
|---|---|
| `backend.py` | Loads the pickle, cleans text, predicts, explains. No Streamlit import — reusable from a CLI, a test, or a FastAPI service. |
| `app.py` | Widgets, layout, charts. No model code — the UI can be rewritten without touching model behaviour. |

The label convention is fixed by the notebook: **spam = 0, ham = 1**, so
`predict_proba(...)[0]` is P(spam). `app.py` never re-derives it — it reads
`Prediction.spam_prob` from the backend, which is the single place that
encodes it.

**Explanations are exact, not approximated.** Logistic regression scores a
message as `intercept + Σ (tfidf_i × coef_i)`. The *Words that drove this
prediction* chart ranks terms by that product, so it reports the model's real
arithmetic rather than an estimate of it.

**Known limits**

- Trained on 2000s-era UK SMS text — modern email spam, other languages and
  emoji-heavy messages are out of distribution.
- Only words the 20,000-term vocabulary has seen carry any weight; the
  *What the model actually saw* expander shows which of your words were ignored.
- Rewritten spam that avoids trigger words ("adversarial" spam) will slip
  through; this is a bag-of-words model, not a semantic one.
"""
    )

st.divider()
st.caption(
    "📧 Spam Mail Detector · Scikit-Learn + Streamlit · Internship Project 2026 · "
    f"model: `{INFO['model_path'].split('/')[-1]}`"
)
