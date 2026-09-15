# 📧 Spam Mail Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> An end-to-end Machine Learning project that classifies SMS/email messages as
> **Spam** or **Ham** (legitimate) — and explains every prediction it makes.
> Built as a 45-day internship project.

---

## 📌 What this is

A complete NLP pipeline, not just a notebook:

- **98.03% accuracy / 99.30% ROC-AUC** on 1,115 held-out messages
- **Six algorithms compared** before choosing Logistic Regression, then tuned
  across 180 cross-validated fits
- **Explains itself** — every prediction is broken down word by word using the
  model's own arithmetic, not an approximation
- **Deployed** as a Streamlit web app with an adjustable sensitivity threshold
  and batch CSV scoring
- **Tested** — 30 automated tests run against the real model without a browser

---

## 🚀 Quick start

```bash
git clone <your-repo-url>
cd email-spam-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py               # → http://localhost:8501
```

**Other entry points:**

```bash
python backend.py                  # smoke-test the model, no UI
pytest -q                          # run the test suite (needs: pip install pytest)
```

**To re-run the notebook** (training, charts, SMOTE, LIME):

```bash
pip install -r requirements.txt -r requirements-notebook.txt
jupyter notebook
```

Deploying to Streamlit Community Cloud: see
[`documentation/streamlit_app_guide.md`](documentation/streamlit_app_guide.md).

---

## 📂 Project structure

```
email-spam-detection/
│
├── app.py                          # Streamlit UI — widgets and layout only
├── backend.py                      # All inference: loading, cleaning, predicting, explaining
│
├── saved_model/
│   ├── spam_model_pipeline.pkl     # ← the one the app loads (TF-IDF + LogisticRegression)
│   ├── vectorizer.pkl              # TF-IDF alone, for inspection
│   └── model.pkl                   # classifier alone, for inspection
│
├── data/
│   └── spam.csv                    # SMS Spam Collection, 5,572 messages
│
├── Spam_Mail_Prediction_using_Machine_Learning.ipynb   # all 10 phases
│
├── tests/
│   └── test_backend.py             # 30 tests: labels, cleaning, thresholds, explanations
│
├── documentation/
│   ├── full_project_explanation.md # deep phase-by-phase walkthrough + interview Q&A
│   ├── streamlit_app_guide.md      # architecture, deployment, troubleshooting
│   └── phase_9_10_code_for_notebook.md
│
├── presentation/
│   ├── gamma_ai_ppt_prompt.txt     # generates the 13-slide deck
│   └── ppt_presentation_script.md  # timed speaker script + Q&A prep
│
├── .streamlit/config.toml          # theme and server settings
├── requirements.txt                # app dependencies (kept minimal for cloud deploy)
├── requirements-notebook.txt       # notebook-only extras (matplotlib, lime, SMOTE, …)
├── goals.md                        # phase-by-phase progress tracker
└── README.md
```

### Why `app.py` and `backend.py` are separate

`backend.py` imports no Streamlit; `app.py` contains no model code. That means
the inference layer can be unit-tested without a browser, reused behind a REST
API, and the UI can be redesigned with zero risk to model behaviour.

```
app.py  ──imports──►  backend.py  ──joblib.load──►  saved_model/spam_model_pipeline.pkl
(UI only)             (no Streamlit)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [SMS Spam Collection — Kaggle (UCI)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| File | `data/spam.csv` |
| Messages | 5,572 |
| Ham | 4,825 (86.6%) |
| Spam | 747 (13.4%) |
| Split | 4,457 train / 1,115 test (80/20, `random_state=3`) |

The 86.6/13.4 imbalance drove the whole project: a model that always answers
"ham" scores 86.6% accuracy and is useless, which is why every result below is
reported with precision, recall and F1 alongside accuracy.

---

## 🔬 Project phases

| Phase | Title | What it produced |
|---|---|---|
| 1 | Exploratory Data Analysis | Found the imbalance; spam is 2× longer with more caps, digits and links |
| 2 | Advanced Text Preprocessing (NLTK) | `clean_text()` — lowercase → strip URLs/digits/punctuation → stopwords → lemmatize |
| 3 | Multi-Model Comparison | 6 classifiers × 5 metrics; Logistic Regression chosen |
| 4 | Comprehensive Evaluation | Confusion matrix, ROC/PR curves, 5-fold CV |
| 5 | Hyperparameter Tuning | GridSearchCV, 36 configs × 5 folds = 180 fits |
| 6 | Class Imbalance (SMOTE) | Measured SMOTE vs `class_weight` vs nothing |
| 7 | Model Saving | `spam_model_pipeline.pkl` via joblib |
| 8 | Streamlit Web App | `app.py` + `backend.py` |
| 9 | Explainability | Coefficient contributions in the app; LIME in the notebook |
| 10 | Documentation & Tests | This README, the deep-dive docs, the deck, 30 tests |

Full technical walkthrough with code, reasoning and interview questions:
[`documentation/full_project_explanation.md`](documentation/full_project_explanation.md).

---

## 📈 Results

Measured on the 1,115 messages held out from training:

| Metric | Value |
|---|---|
| **Accuracy** | **98.03%** |
| **ROC-AUC** | **99.30%** |
| Spam precision | 97.84% |
| Spam recall | 87.74% |
| Spam F1 | 92.52% |

**Confusion matrix**

|  | Predicted Spam | Predicted Ham |
|---|---|---|
| **Actually Spam** (155) | 136 ✅ | 19 ❌ missed |
| **Actually Ham** (960) | 3 ❌ false alarms | 957 ✅ |

> **The honest reading:** only 3 of 960 legitimate messages were wrongly flagged
> — but 19 of 155 spam messages got through. Spam recall is **87.7%**, not 98%.
> Accuracy is flattered by the easy ham majority, which is exactly why it is
> never quoted on its own here.

### The threshold trade-off

The model outputs a probability; turning it into a verdict needs a cut-off, and
the app exposes it as a slider instead of hiding it in a constant:

| Threshold | Accuracy | Spam recall | Spam precision | Good mail flagged | Spam missed |
|---|---|---|---|---|---|
| 0.20 | 97.85% | **94.19%** | 90.68% | 15 | 9 |
| 0.35 | 98.03% | 90.32% | 95.24% | 7 | 15 |
| **0.50** (default) | **98.03%** | 87.74% | **97.84%** | **3** | 19 |
| 0.70 | 97.22% | 81.29% | 98.44% | 2 | 29 |

### The final model

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              stop_words='english')),
    ('clf',   LogisticRegression(C=10.0, max_iter=1000, penalty='l2')),
])
```

Strongest learned signals — the model rediscovered SMS spam vocabulary with no
prompting:

| Spam indicators (negative coefficients) | Ham indicators (positive) |
|---|---|
| txt (−9.12), claim (−7.71), mobile (−7.68), service (−6.75), reply (−6.61), prize (−5.80), win (−5.01), free (−4.77), urgent (−4.50), cash (−4.47) | ok (+3.92), home (+2.94), road (+2.86), say (+2.69), later (+2.60), good (+2.52), way (+2.42) |

---

## 🖥️ What the app does

| Tab | Contents |
|---|---|
| **Classify** | Single message → verdict, confidence, probability bars, per-word contribution chart, message signals, cleaned-text inspector, session history |
| **Batch check** | Paste a list or upload a CSV, score everything, download results |
| **Model insights** | Live test-set metrics, threshold trade-off table, global learned coefficients |
| **How it works** | Pipeline diagram, architecture notes, honest limitations |

The sidebar carries the sensitivity slider plus a model-status panel that warns
if the running scikit-learn ever drifts from the 1.6.1 the pickle was written
with, or if NLTK corpora had to fall back to the built-in cleaner.

---

## 🛠️ Tech stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn **1.6.1** (pinned) | TF-IDF, 6 models, GridSearchCV, metrics |
| NLTK | Stopwords, WordNet lemmatization |
| pandas / NumPy | Data manipulation |
| Streamlit | Web app |
| joblib | Model serialization |
| pytest | Test suite |
| matplotlib / seaborn / wordcloud | Notebook charts *(notebook only)* |
| imbalanced-learn | SMOTE *(notebook only)* |
| LIME | Model-agnostic explanations *(notebook only)* |

> **Why scikit-learn is pinned:** a pickle stores structure, not code. Loading
> the model under a different version raises `InconsistentVersionWarning` and can
> silently change predictions, so the version is fixed and checked at runtime.

---

## ⚠️ Known limitations

1. Trained on UK SMS text from around 2005 — modern email spam, other languages
   and emoji-heavy messages are out of distribution.
2. Bag-of-words has no sense of word order; "not free" and "free not" are
   identical to it. Bigrams only partly compensate.
3. Deliberately obfuscated spam (`F.R.E.E`, `pr1ze`) slips straight through.
4. 12% of spam gets past the default threshold — the slider trades that against
   false alarms, but cannot eliminate it.
5. Only the 20,000 terms seen during training carry weight; the app shows which
   of your words were ignored.

**Next steps:** fine-tune DistilBERT for word order, retrain on a modern corpus
(Enron / current phishing sets), add character n-grams for obfuscation, and add
production monitoring — spam evolves, so an unretrained model decays.

---

## 👨‍💻 Author

**Arjun** — Internship Project, 2026
Built with Python & Scikit-Learn

## 📄 License

Open source under the MIT License.
