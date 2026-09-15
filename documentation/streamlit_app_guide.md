# Streamlit App Guide — architecture, running it, deploying it

Everything you need to get the app running locally and live on Streamlit
Community Cloud, plus what to do when it breaks.

---

## 1. How the two files fit together

```
                    ┌──────────────────────────────────────────┐
  browser  ────────►│  app.py                                  │
                    │  ───────                                 │
                    │  st.set_page_config, CSS, tabs           │
                    │  sidebar: threshold slider, model status │
                    │  tab 1: single message + explanation     │
                    │  tab 2: batch (paste or CSV)             │
                    │  tab 3: metrics + global coefficients    │
                    │  tab 4: how it works                     │
                    └───────────────┬──────────────────────────┘
                                    │ imports
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  backend.py                              │
                    │  ──────────                              │
                    │  resolve_model_path()  find the .pkl     │
                    │  clean_text()          Phase-2 cleaning  │
                    │  SpamPredictor.predict()      → Prediction
                    │  SpamPredictor.predict_many() → [Prediction]
                    │  SpamPredictor.explain()      → [TokenContribution]
                    │  SpamPredictor.global_top_terms()        │
                    │  SpamPredictor.info()                    │
                    └───────────────┬──────────────────────────┘
                                    │ joblib.load
                                    ▼
                    saved_model/spam_model_pipeline.pkl
                    (TfidfVectorizer → LogisticRegression)
```

**The rule: `backend.py` never imports Streamlit, and `app.py` never touches the
model.** That is what makes the backend testable without a browser and the UI
safe to redesign.

### The contract between them

| Backend gives the UI | Shape |
|---|---|
| `Prediction` | `.label_name`, `.is_spam`, `.spam_prob`, `.ham_prob`, `.confidence`, `.is_borderline`, `.cleaned`, `.features`, `.as_row()` |
| `TokenContribution` | `.term`, `.tfidf_value`, `.coefficient`, `.contribution`, `.pushes_toward` |
| `info()` | model path, estimator, `C`, vocabulary size, n-gram range, scikit-learn versions, `version_mismatch`, `nlp_backend` |
| `SAMPLE_MESSAGES` | the quick-test examples rendered as buttons |

`spam = 0, ham = 1` is encoded **once**, in `backend.py`. The UI reads
`.spam_prob` and never re-derives it from a probability array.

---

## 2. Run it locally

```bash
git clone <your-repo-url>
cd email-spam-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Opens on <http://localhost:8501>.

**Check the backend on its own** (no browser, prints predictions for every
sample message):

```bash
python backend.py
```

**Run the tests** before you push anything:

```bash
pip install pytest
pytest -q                          # 30 tests
```

---

## 3. Deploy to Streamlit Community Cloud

1. Push everything to GitHub, **including `saved_model/*.pkl`**. The pipeline
   file is ~950 KB, comfortably under GitHub's 100 MB limit — no Git LFS needed.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → pick the repo → set:
   - **Branch:** `testing` (or `main` once merged)
   - **Main file path:** `app.py`
   - **Python version:** 3.11 (3.9–3.12 all work)
4. Click **Deploy**. The first build takes 2–4 minutes while it installs
   `requirements.txt`.

### What the cloud runner does

- Installs **`requirements.txt` only** — this is why notebook-only libraries
  live in `requirements-notebook.txt`. Every extra package is build time and one
  more chance of a resolver conflict.
- Reads `.streamlit/config.toml` for the theme and the 20 MB upload cap.
- Downloads the NLTK corpora on first run (~10 MB, a few seconds). If that is
  blocked, `backend.py` falls back to a bundled stopword list and the sidebar
  says **"Text cleaning: built-in fallback"** instead of failing.

### Pre-flight checklist

- [ ] `saved_model/spam_model_pipeline.pkl` is committed (`git ls-files saved_model/`)
- [ ] `requirements.txt` pins `scikit-learn==1.6.1`
- [ ] `pytest -q` passes locally
- [ ] `streamlit run app.py` works from a **clean clone** in a fresh virtualenv
- [ ] No absolute paths like `/content/...` anywhere in `app.py` or `backend.py`

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **"Model file not found"** on start-up | The `.pkl` was never committed, or is somewhere unexpected | `git add -f saved_model/spam_model_pipeline.pkl`. The error message lists every path that was searched. |
| Sidebar warns **"Model was pickled with scikit-learn 1.6.1 but X is running"** | The pin was loosened or overridden | Restore `scikit-learn==1.6.1` in `requirements.txt` and reboot the app. Predictions may be wrong until you do. |
| Sidebar says **"built-in fallback"** for text cleaning | NLTK corpora could not be downloaded | Usually transient — reboot the app. Predictions still work, with a small accuracy cost. |
| `ModuleNotFoundError: No module named 'sklearn'` | Wrong requirements file, or the app was pointed at the wrong main file | The package is `scikit-learn`; the import is `sklearn`. Check `requirements.txt` was picked up in the build log. |
| App boots then dies with **"Oh no. Error running app"** | Almost always a missing dependency or an exception at import time | Open **Manage app → Logs** in the cloud dashboard; the traceback is at the bottom. |
| Everything is slow on every click | The model is being reloaded per interaction | `load_predictor()` must keep its `@st.cache_resource` decorator. |
| `UnicodeDecodeError` on a batch CSV upload | The file is not UTF-8 | The app already reads with `encoding='latin-1'`; if a file still fails, re-save it as UTF-8. |
| Exceeded resource limits | The free tier caps at ~1 GB RAM | This app uses ~200 MB. If you hit it, you are probably uploading a very large CSV — the cap is set to 20 MB in `.streamlit/config.toml`. |

---

## 5. Demo script for the live app

Five minutes, in this order:

1. **Ham example** → *"Hey, are you coming to the party tonight?"* → green,
   99% confidence. Establishes the happy path.
2. **Spam example** → the FA Cup prize draw → red, 93%. Scroll to the
   explanation and name the drivers: `txt`, `win`, `free`, **and the bigram
   `free entry`** — proof the phrase-level feature is doing work.
3. **Phishing example** → lands at P(spam) ≈ 45%, which the app flags as a
   **borderline call**. This is the moment worth planning your demo around.
4. **Move the sidebar slider to 0.35** → the same message flips to spam.
   Explain: the model did not change, the *policy* did. Then open **Model
   insights** and show the trade-off table — 0.20 catches 94% of spam but flags
   12 more good messages.
5. **Batch tab** → paste five mixed messages → instant table with a per-row
   P(spam) bar and a CSV download. Shows it is a tool, not a toy.

**If the app is cold**, load it a minute before you present — the first request
wakes the container and downloads NLTK data.

---

## 6. Extending it

| Want to… | Do this |
|---|---|
| Swap in a retrained model | Overwrite `saved_model/spam_model_pipeline.pkl` and update `TRAINED_SKLEARN_VERSION` in `backend.py`. Nothing else changes. |
| Add a REST API | `from backend import get_predictor` in a FastAPI route. No Streamlit involved. |
| Add LIME to the UI | `pip install lime`, call it inside `render_contributions`. Expect ~500 ms per explanation versus ~1 ms now. |
| Log predictions | Append `result.as_row()` to a CSV or a database in the classify branch. Note that Streamlit Cloud's filesystem is ephemeral. |
| Change the theme | Edit `.streamlit/config.toml`. The CSS uses `var(--spam)` / `var(--ham)` tokens, so colours change in one place. |
