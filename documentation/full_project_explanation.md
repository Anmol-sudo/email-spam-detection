# Spam Mail Prediction — Complete Phase-by-Phase Explanation

> **Who this is for:** you, the week before the demo. Every phase below has the
> real code, what each line does, what it produced, and the question an
> interviewer will ask about it.
>
> **Project:** Spam Mail Prediction using Machine Learning
> **Author:** Arjun · 45-day internship, 2026
> **Dataset:** SMS Spam Collection (UCI / Kaggle) — 5,572 labelled messages
> **Final model:** TF-IDF (20,000 terms, unigrams + bigrams) → Logistic Regression (C = 10.0)
> **Headline result:** 98.03% accuracy, 99.30% ROC-AUC on 1,115 held-out messages

---

## Table of contents

| # | Phase | What it delivered |
|---|---|---|
| 0 | [Setup & data loading](#phase-0--setup--data-loading) | A clean two-column DataFrame |
| 1 | [Exploratory Data Analysis](#phase-1--exploratory-data-analysis-eda) | Found the 87/13 imbalance and the signals that separate spam |
| 2 | [Text preprocessing (NLTK)](#phase-2--advanced-text-preprocessing-nltk) | `clean_text()` — the function the whole project depends on |
| 3 | [Multi-model comparison](#phase-3--multi-model-comparison) | 6 classifiers scored on 5 metrics |
| 4 | [Evaluation metrics](#phase-4--comprehensive-evaluation-metrics) | Confusion matrix, ROC/PR curves, 5-fold CV |
| 5 | [Hyperparameter tuning](#phase-5--hyperparameter-tuning) | GridSearchCV over 36 configurations |
| 6 | [Class imbalance (SMOTE)](#phase-6--class-imbalance-handling-smote) | Measured SMOTE vs class weights vs nothing |
| 7 | [Model saving](#phase-7--model-saving--deployment-pipeline) | `spam_model_pipeline.pkl` |
| 8 | [Streamlit web app](#phase-8--streamlit-web-application) | `app.py` + `backend.py` |
| 9 | [Explainability](#phase-9--feature-importance--lime-explainability) | Why each prediction happened |
| 10 | [Documentation](#phase-10--documentation--handover) | This file, the README, the deck |

---

## The one-paragraph version

We take 5,572 SMS messages labelled spam or ham, clean the text with NLTK
(lowercase, strip URLs/digits/punctuation, drop stopwords, lemmatize), turn it
into numbers with TF-IDF over unigrams and bigrams, and train a Logistic
Regression classifier. We compared six algorithms before picking it, tuned it
with GridSearchCV, checked whether SMOTE helped with the class imbalance, saved
the winning pipeline to disk, and wrapped it in a Streamlit web app that
explains every prediction it makes.

---

## Phase 0 — Setup & data loading

### The code

```python
raw_mail_data = pd.read_csv('/content/data/spam.csv', encoding='latin-1')
raw_mail_data = raw_mail_data[['v1', 'v2']]
raw_mail_data.columns = ['Category', 'Message']
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
```

### Line by line

| Line | What it does | Why |
|---|---|---|
| `encoding='latin-1'` | Reads the file as Latin-1, not UTF-8 | The file contains `£` and other bytes that crash a UTF-8 read with `UnicodeDecodeError`. This is the single most common error people hit on this dataset. |
| `[['v1', 'v2']]` | Keeps only the first two columns | The CSV has three trailing unnamed columns that are almost entirely `NaN` — artefacts of how the file was exported. |
| `.columns = [...]` | Renames `v1`/`v2` | `Category` and `Message` are self-documenting; `v1` is not. |
| `.where(pd.notnull(...), '')` | Replaces any `NaN` with an empty string | A `NaN` message would crash `clean_text()` later with `AttributeError: 'float' has no attribute 'lower'`. |

### Result

5,572 rows × 2 columns, no nulls.

> **Interview question:** *Why `latin-1` and not `utf-8`?*
> Because the file was encoded in Latin-1 and contains the byte `0xA3` (`£`),
> which is not valid UTF-8. Latin-1 also never fails on any byte value, which
> makes it a safe fallback for unknown Western-European text.

---

## Phase 1 — Exploratory Data Analysis (EDA)

**Goal:** understand the data *before* modelling it, so later decisions are
informed rather than guessed.

### 1.1 Class distribution

```python
class_counts = mail_data['Category'].value_counts()
```

| Class | Count | Share |
|---|---|---|
| ham | 4,825 | 86.6% |
| spam | 747 | 13.4% |

**This one number shaped the rest of the project.** With 87% ham, a model that
blindly answers "ham" every time scores 87% accuracy while being completely
useless. That is why Phase 4 reports precision, recall and F1 instead of
accuracy alone, and why Phase 6 exists at all.

### 1.2 Engineered length and character features

```python
mail_data['char_count']      = mail_data['Message'].apply(len)
mail_data['word_count']      = mail_data['Message'].apply(lambda x: len(x.split()))
mail_data['uppercase_ratio'] = mail_data['Message'].apply(
    lambda x: sum(c.isupper() for c in x) / (len(x) + 1))
mail_data['digit_count']     = mail_data['Message'].apply(
    lambda x: sum(c.isdigit() for c in x))
mail_data['has_url']         = mail_data['Message'].apply(
    lambda x: 1 if re.search(r'http|www|\.com', x, re.I) else 0)
```

Note the `+ 1` in `uppercase_ratio` — it prevents a `ZeroDivisionError` on an
empty message. A small defensive habit worth pointing out in a demo.

**What the histograms showed:** spam messages are roughly **twice as long** as
ham (they have to fit an entire sales pitch, a shortcode and the small print),
and they carry far more uppercase letters and digits (`WIN`, `FREE`, `87121`,
`£1000`).

### 1.3 WordClouds and top-20 words

Spam clusters around **free, call, txt, claim, prize, mobile, win, urgent**.
Ham clusters around **ok, got, come, home, later, love, time** — the vocabulary
of actual human conversation. These two word lists are visually the whole
project: the classes really are separable by words alone, which is exactly why
a simple bag-of-words model works so well here.

### 1.4 Correlation heatmap

```python
mail_data['label'] = mail_data['Category'].map({'spam': 0, 'ham': 1})
corr = mail_data[corr_cols].corr()
```

Confirms numerically what the histograms showed: `char_count`, `digit_count`
and `has_currency` all correlate with the spam label.

> **Interview question:** *You engineered all these features and then never fed
> them to the model. Why?*
> They served as evidence, not input. They proved the classes are separable and
> justified the TF-IDF approach. Adding them to the TF-IDF matrix would have
> meant scaling them to match sparse TF-IDF values for maybe a fraction of a
> point of F1. They still appear in the app as the *Message signals* panel, so
> a user can see the same evidence.

---

## Phase 2 — Advanced Text Preprocessing (NLTK)

**This is the most important function in the project.** Everything downstream —
the vectorizer's vocabulary, the model's coefficients, and every prediction the
app ever makes — is defined by what comes out of it.

### The code

```python
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text, use_stemming=False):
    text = text.lower()                                      # 1
    text = re.sub(r'http\S+|www\S+', '', text)               # 2
    text = re.sub(r'\d+', '', text)                          # 3
    text = text.translate(str.maketrans('', '', string.punctuation))  # 4
    tokens = text.split()                                    # 5
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]  # 6
    tokens = [lemmatizer.lemmatize(t) for t in tokens]       # 7
    return ' '.join(tokens)
```

### Step by step

| # | Step | Why it matters |
|---|---|---|
| 1 | **Lowercase** | `FREE`, `Free` and `free` are the same signal. Without this the model learns three separate, weaker features. |
| 2 | **Strip URLs** | Every spam URL is different, so each one becomes a unique token the model can never generalise from. *Whether* a URL exists is a signal; *which* URL it is, is noise. |
| 3 | **Strip digits** | Same logic: `87121` and `87131` are different tokens but the same idea. |
| 4 | **Strip punctuation** | `free!` and `free` should collapse into one token. |
| 5 | **Tokenize** | `.split()` on whitespace — deliberately simpler and ~10× faster than `word_tokenize`, and sufficient once punctuation is gone. |
| 6 | **Drop stopwords and 1-char tokens** | "the", "is", "and" appear in everything, so they carry no discriminative power. Removing them shrinks the vocabulary and sharpens the signal. |
| 7 | **Lemmatize** | `offers` → `offer`, `prizes` → `prize`. Collapses inflections into one feature instead of several thin ones. |

### Lemmatization vs stemming

The function supports both; we chose lemmatization.

| | Stemming (Porter) | Lemmatization (WordNet) |
|---|---|---|
| Method | Chops suffixes by rule | Dictionary lookup to the real base word |
| `studies` → | `studi` (not a word) | `study` |
| Speed | Faster | Slower |
| Output | Unreadable | Readable |

Readable output matters here because Phase 9 shows these exact tokens to users
as the explanation. A chart labelled `studi` and `claimi` undermines the demo;
`study` and `claim` sell it.

### The effect

Raw: `"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005..."`
Cleaned: `"free entry wkly comp win fa cup final tkts st may text fa receive entry question std txt rate"`

Average word count drops by roughly 40%, and what remains is almost entirely
signal.

### The bigram decision

```python
tfidf_clean = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=15000)
```

`ngram_range=(1, 2)` means the vectorizer keeps single words **and** adjacent
word pairs. This lets the model learn `"free entry"` and `"call now"` as units.
Look at the app's explanation panel on the prize-draw example: `free entry`
appears as its own feature alongside `free` and `entry`. That is the bigram
earning its place.

> **Interview question:** *What is TF-IDF, in one sentence?*
> Term Frequency × Inverse Document Frequency: a word scores high in a message
> if it appears often *in that message* but rarely *across the corpus* — so
> common words get damped and distinctive words like "claim" get amplified.

> **Interview question:** *Your app cleans text before calling the pipeline, but
> the pipeline has its own vectorizer. Isn't that doubled up?*
> No — the vectorizer only tokenizes and weights; it never lemmatizes or strips
> URLs. The pipeline was *fitted* on `clean_text()` output, so inference must
> apply exactly the same function. If the app skipped it, every message would
> arrive in a different shape than training and accuracy would silently drop.
> That is why `clean_text()` lives in `backend.py` — one definition, used by the
> app, the tests and any future service.

---

## Phase 3 — Multi-Model Comparison

**Goal:** choose a model with evidence instead of habit.

### The code

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'Naive Bayes':         MultinomialNB(alpha=0.1),
    'Linear SVM':          LinearSVC(C=1.0, max_iter=2000),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
}

for name, clf in models.items():
    clf.fit(X_train_clean, Y_train_c)
    preds = clf.predict(X_test_clean)
    proba = (clf.predict_proba(X_test_clean)[:, 1] if hasattr(clf, 'predict_proba')
             else clf.decision_function(X_test_clean))
```

### The detail worth pointing out

```python
if hasattr(clf, 'predict_proba'):
    proba = clf.predict_proba(X_test_clean)[:, 1]
else:
    proba = clf.decision_function(X_test_clean)
```

`LinearSVC` has no `predict_proba` — an SVM outputs a signed distance from the
decision boundary, not a probability. ROC-AUC only needs a *ranking*, so the raw
decision function works fine. Calling `predict_proba` on it would raise
`AttributeError`, and this branch is why the loop does not crash.

### Why each model is in the list

| Model | Why it was included |
|---|---|
| **Logistic Regression** | The linear baseline for text. Fast, calibrated probabilities, and every coefficient is directly readable. |
| **Naive Bayes** | The classical spam-filter algorithm. Assumes words are independent — wrong in principle, strong in practice on text. |
| **Linear SVM** | Usually the strongest linear text classifier; maximises the margin between classes. |
| **Random Forest** | A non-linear, bagged tree ensemble — tests whether non-linearity helps at all. |
| **Gradient Boosting** | Trees built sequentially to fix prior errors. Slowest to train here. |
| **KNN** | A deliberate control. Distance in 15,000-dimensional sparse space is nearly meaningless, and its score shows that. |

### Outcome

Logistic Regression and Linear SVM led on F1. **We chose Logistic Regression**
because it ties on accuracy, trains in under a second, returns real
probabilities (which the app's threshold slider and confidence display need —
an SVM cannot give you those), and its coefficients are the explanation shown
in Phase 9. KNN trailed badly, exactly as predicted.

> **Interview question:** *Why not use the ensemble models — aren't they more powerful?*
> Power without payoff. Text after TF-IDF is high-dimensional and close to
> linearly separable, so trees add complexity for no measurable gain here, train
> far slower, and cannot hand me an explanation or a calibrated probability.
> The "No Free Lunch" theorem in practice: the best model is the one that fits
> *this* problem.

---

## Phase 4 — Comprehensive Evaluation Metrics

**Goal:** prove the model works, and be honest about where it does not.

### Measured results on the 1,115 held-out messages

| Metric | Value |
|---|---|
| Accuracy | **98.03%** |
| ROC-AUC | **99.30%** |
| Precision (ham class) | 98.05% |
| Recall (ham class) | 99.69% |
| F1 (ham class) | 98.86% |
| **Precision (spam class)** | **97.84%** |
| **Recall (spam class)** | **87.74%** |
| **F1 (spam class)** | **92.52%** |

### The confusion matrix

|  | Predicted Spam | Predicted Ham |
|---|---|---|
| **Actually Spam** (155) | 136 ✅ | **19 ❌ missed spam** |
| **Actually Ham** (960) | **3 ❌ false alarms** | 957 ✅ |

**Read this table out loud in the demo — it is the most honest slide you have.**
Of 960 legitimate messages, only **3** were wrongly flagged. Of 155 real spam
messages, **19 slipped through**.

### Why spam recall (87.7%) is so much lower than accuracy (98.0%)

Accuracy is computed over all 1,115 messages, and 960 of them are easy ham. The
majority class drowns out the minority. Spam recall — *of all real spam, how
much did we catch?* — is the number a spam filter actually lives or dies by, and
it is 10 points lower. **Quoting only the 98% would be hiding the weak spot.**

### Which error is worse?

A **false positive** (good mail → spam folder) is far worse than a **false
negative** (spam reaching the inbox). Missing spam costs a user two seconds of
annoyance; losing a job offer or an invoice to the spam folder is a real
failure. The default 0.50 threshold reflects that: it is the setting that
produces just 3 false alarms.

### ROC and Precision-Recall curves

An ROC-AUC of 0.993 means: pick one random spam and one random ham message, and
the model gives the spam a higher spam score 99.3% of the time. The
precision-recall curve matters more on imbalanced data because it ignores the
easy true negatives entirely.

### 5-fold cross-validation

```python
scores = cross_val_score(clf, X_train_clean, Y_train_c, cv=5, scoring='f1', n_jobs=-1)
```

Trains five times on different 80/20 splits of the training data. A single
test-set score could be luck; five consistent scores with a small standard
deviation mean the result is real and the model is not over-fitted to one split.

> **Interview question:** *Your model is 98% accurate. Is that good?*
> It is good *relative to the 86.6% baseline of always guessing ham* — but the
> number that matters is spam recall at 87.7%. Accuracy on data this imbalanced
> flatters every model; I report it alongside precision, recall and F1 for
> exactly that reason.

---

## Phase 5 — Hyperparameter Tuning

**Goal:** stop hand-picking settings and search for them systematically.

### The code

```python
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('clf',   LogisticRegression(max_iter=1000)),
])

param_grid = {
    'tfidf__ngram_range':  [(1, 1), (1, 2)],
    'tfidf__max_features': [5000, 10000, 20000],
    'tfidf__min_df':       [1, 2],
    'clf__C':              [0.1, 1.0, 10.0],
    'clf__penalty':        ['l2'],
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_c, Y_train_c)
```

### Why a `Pipeline` and not two separate objects

This is the most important technical idea in the phase. A `Pipeline` chains the
vectorizer and the classifier into one estimator, which gives us three things:

1. **It prevents data leakage.** With 5-fold CV, the vectorizer is re-fitted on
   each training fold only. Vectorizing everything up front would leak
   vocabulary and IDF statistics from the validation fold into training, and
   inflate the score.
2. **It lets us tune the vectorizer too.** The `tfidf__` prefix reaches into the
   pipeline's `tfidf` step, so `ngram_range` is searched alongside `C`. The
   double underscore is scikit-learn's "step name, then parameter" syntax.
3. **It ships as one object.** Phase 7 pickles the pipeline, so the app calls
   `.predict()` once and the vectorizer is applied automatically — impossible to
   forget, impossible to mismatch.

### The search space

2 × 3 × 2 × 3 × 1 = **36 combinations**, each 5-fold cross-validated =
**180 model fits**, parallelised with `n_jobs=-1`. Scored on **F1, not
accuracy** — on imbalanced data, optimising accuracy would tune toward "predict
ham more often", which is the opposite of what we want.

### The winning configuration (what is inside the .pkl)

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              stop_words='english')),
    ('clf',   LogisticRegression(C=10.0, max_iter=1000, penalty='l2')),
])
```

Three things the search decided for us:

- **`ngram_range=(1, 2)`** — bigrams won. Phrases carry real information.
- **`max_features=20000`** — the largest option won, so vocabulary was still a
  binding constraint. A larger grid might have gone higher.
- **`C=10.0`** — the *weakest* regularisation won. `C` is the inverse of
  regularisation strength, so a high `C` means the model is trusted to use large
  coefficients. With 20,000 clean, highly informative features and little noise,
  constraining it was hurting more than helping.

> **Interview question:** *Why GridSearchCV over RandomizedSearchCV?*
> The grid was small enough (36 points, ~2 minutes) to search exhaustively.
> Randomized search wins when the space is too big to enumerate; here it would
> only add the risk of missing the best cell.

---

## Phase 6 — Class Imbalance Handling (SMOTE)

**Goal:** test whether rebalancing the 87/13 split actually helps.

### The code

```python
smote = SMOTE(random_state=42)
X_train_sm, Y_train_sm = smote.fit_resample(X_train_clean, Y_train_c)
```

### How SMOTE works

**Synthetic Minority Over-sampling Technique.** It does not copy existing spam
rows — duplicates teach a model nothing new and encourage memorisation. For each
minority sample it finds its *k* nearest minority neighbours, picks one, and
generates a new point somewhere along the line between them. The result is new,
plausible, slightly-different spam examples until both classes are equal:

| | Spam | Ham |
|---|---|---|
| Before SMOTE | 592 | 3,865 |
| After SMOTE | 3,865 | 3,865 |

### The critical rule

**SMOTE is applied to the training set only, never the test set.** The test set
must stay a faithful sample of reality — and reality is 13% spam. Synthesising
test data would be measuring the model against messages that never existed.

### The three-way comparison

| Method | What it does |
|---|---|
| No balancing | The plain baseline |
| `class_weight='balanced'` | Tells the loss function to weight each spam error ~6.5× more heavily |
| SMOTE | Generates synthetic spam until classes are 50/50 |

Both balancing methods **raise spam recall and lower spam precision** — they
push the model to say "spam" more readily, catching more real spam at the cost
of more false alarms. That trade-off is genuine, and it is the same lever the
app's threshold slider exposes at inference time.

**The final shipped model does not use SMOTE.** The tuned pipeline from Phase 5
scored best on F1 and produces only 3 false positives out of 960 ham messages,
which matters more for a spam filter than squeezing out extra recall. Phase 6's
value is that this is now a *measured* decision rather than an assumption.

> **Interview question:** *Isn't SMOTE on TF-IDF vectors strange? You are
> interpolating between sparse text vectors.*
> Yes, and that is a fair criticism — a synthetic point between two spam vectors
> does not correspond to any real sentence. It still works as regularisation in
> feature space, but it is exactly why I compared it against `class_weight` and
> against doing nothing rather than assuming it would help. On this dataset it
> did not earn its place.

---

## Phase 7 — Model Saving & Deployment Pipeline

**Goal:** stop retraining. Train once, load everywhere.

### The code

```python
joblib.dump(best_tuned, f'{save_dir}/spam_model_pipeline.pkl')
joblib.dump(best_tuned.named_steps['tfidf'], f'{save_dir}/vectorizer.pkl')
joblib.dump(best_tuned.named_steps['clf'],   f'{save_dir}/model.pkl')
```

Three files are saved:

| File | Contents | Used by |
|---|---|---|
| `spam_model_pipeline.pkl` | Vectorizer **and** classifier as one object | **The app — this is the one that matters** |
| `vectorizer.pkl` | The fitted TF-IDF only | Convenience / inspection |
| `model.pkl` | The fitted classifier only | Convenience / inspection |

### Why `joblib` and not `pickle`

Both serialise Python objects; `joblib` is optimised for the large NumPy arrays
inside a fitted model, so it is faster and produces smaller files. It is what
scikit-learn's own documentation recommends.

### Two deployment traps this project hit

**1. Version pinning.** A pickle stores the object's *structure*, not its code.
Load a model pickled by scikit-learn 1.6.1 under a different version and you get
`InconsistentVersionWarning` — and in the worst case, silently different
predictions. `requirements.txt` therefore pins `scikit-learn==1.6.1`, and
`backend.py` compares the running version against the trained one and surfaces a
warning in the sidebar if they ever drift.

**2. Path resolution.** The first version of the app loaded
`"saved_model/spam_model_pipeline.pkl"` — a path relative to whatever directory
the process happened to start in. `backend.resolve_model_path()` now searches
several locations relative to the module file itself and raises a clear,
actionable error instead of a stack trace.

### The prediction function

```python
def predict_email(text):
    cleaned = clean_text(text)
    pred    = loaded_pipeline.predict([cleaned])[0]
    proba   = loaded_pipeline.predict_proba([cleaned])[0]
    label   = '✅ HAM (Legitimate)' if pred == 1 else '🚨 SPAM'
    return label, proba[pred] * 100
```

Note `[cleaned]` — a list, not a string. The vectorizer expects an iterable of
documents. Passing a bare string makes it treat each *character* as a document,
which produces no error and completely wrong output. A classic silent bug.

> **Interview question:** *What is the spam = 0 / ham = 1 convention and why does it matter?*
> Phase 4 of the notebook mapped `spam → 0, ham → 1`, so
> `predict_proba(...)[0]` is P(spam) and `[1]` is P(ham). Get it backwards and
> the app confidently reports the exact opposite of the truth — with no error to
> warn you. It is encoded once in `backend.py` as the `SPAM`/`HAM` constants,
> and a unit test asserts `classifier.classes_ == [0, 1]` so a future retrain
> cannot flip it unnoticed.

---

## Phase 8 — Streamlit Web Application

**Goal:** turn the model into something a non-technical person can use.

### The architecture

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│  app.py  (front-end)    │ ─────► │  backend.py  (inference)     │
│                         │        │                              │
│  • widgets and layout   │        │  • resolve_model_path()      │
│  • charts and styling   │        │  • clean_text()   (Phase 2)  │
│  • threshold slider     │        │  • SpamPredictor.predict()   │
│  • batch upload         │        │  • SpamPredictor.explain()   │
│                         │ ◄───── │  • returns Prediction objects│
│  imports streamlit      │ result │  imports NO streamlit        │
└─────────────────────────┘        └──────────────────────────────┘
                                              │
                                              ▼
                                   saved_model/spam_model_pipeline.pkl
```

**Why split the file in two?** Three concrete payoffs:

1. **Testability.** `tests/test_backend.py` runs 30 assertions against the real
   model with no browser and no Streamlit server.
2. **Reuse.** The same `SpamPredictor` can back a FastAPI endpoint or a CLI
   without touching a line of UI code.
3. **Safety.** Redesigning the UI cannot change model behaviour, because the UI
   contains no model logic at all.

### The pieces that matter

**Caching.** `@st.cache_resource` loads the 957 KB pickle once per server
process instead of on every interaction. Streamlit re-runs the entire script on
every click; without this, every keystroke would re-read the model from disk.

**The threshold slider.** The model outputs a probability; turning it into a
verdict requires a cut-off, and 0.50 is a default, not a law. Measured on the
held-out set:

| Threshold | Accuracy | Spam recall | Spam precision | Good mail flagged | Spam missed |
|---|---|---|---|---|---|
| 0.20 | 97.85% | **94.19%** | 90.68% | 15 | 9 |
| 0.35 | 98.03% | 90.32% | 95.24% | 7 | 15 |
| **0.50** (default) | **98.03%** | 87.74% | **97.84%** | **3** | 19 |
| 0.70 | 97.22% | 81.29% | 98.44% | 2 | 29 |

Lowering the threshold catches 10 more spam messages but wrongly flags 12 more
legitimate ones. **There is no single correct setting — it is a business
decision**, and the app makes it visible instead of burying it.

**Explanations.** Logistic regression scores a message as
`intercept + Σ (tfidf_i × coef_i)`. The app ranks terms by that **product**, not
by the raw coefficient, so it reports what actually moved *this* message: a word
with a huge coefficient that barely appears contributes little. A unit test
verifies the reconstruction matches `decision_function()` exactly.

**Graceful degradation.** If NLTK's corpora cannot be downloaded (blocked
network, cold container), `backend.py` falls back to a bundled stopword list and
a rule-based lemmatizer, and the sidebar says so. The app stays up.

### Bugs found and fixed while building this

| Bug | Symptom | Fix |
|---|---|---|
| Model path was relative to the working directory, and the `.pkl` files were in the repo root rather than `saved_model/` | App crashed on start-up with `FileNotFoundError` | Moved artifacts to `saved_model/`; resolve the path relative to the module, with fallbacks |
| Example buttons wrote to `st.session_state`, then deleted the key before `st.rerun()` | Clicking an example left the box empty | Keyed widget plus an `on_click` callback, which runs *before* the re-run renders |
| `requirements.txt` pinned nothing | Cloud installs a newer scikit-learn than the pickle expects | `scikit-learn==1.6.1`, plus a runtime version check |
| Batch "Run" button only appeared after the textarea lost focus | Had to click twice | Wrapped the paste path in `st.form`, which commits text and click together |
| `requirements.txt` carried matplotlib, seaborn, wordcloud, lime, imbalanced-learn | Slow cloud builds; more to break; none imported by the app | Split into `requirements-notebook.txt` |

---

## Phase 9 — Feature Importance & LIME Explainability

**Goal:** answer *why*, not just *what*.

### Two complementary views

**Global — the model's coefficients.** What the model believes in general,
independent of any message:

```python
coef = best_tuned.named_steps['clf'].coef_[0]
feature_names = np.array(best_tuned.named_steps['tfidf'].get_feature_names_out())
top_spam_idx = coef.argsort()[:25]        # most negative  → spam
top_ham_idx  = coef.argsort()[-25:][::-1] # most positive  → ham
```

Because `spam = 0` and `ham = 1`, **negative coefficients point to spam** and
positive ones to ham. The actual top learned terms:

| Strongest spam indicators | Strongest ham indicators |
|---|---|
| txt (−9.12), claim (−7.71), mobile (−7.68), service (−6.75), reply (−6.61), text (−6.45), tone (−5.87), prize (−5.80), stop (−5.67), win (−5.01), free (−4.77), urgent (−4.50), cash (−4.47) | ltgt (+4.40), ok (+3.92), ill (+3.49), home (+2.94), road (+2.86), say (+2.69), later (+2.60), good (+2.52), im (+2.52), way (+2.42) |

Worth noticing in a demo: the model independently rediscovered every word a
human would list for SMS spam — and the ham side is pure casual conversation.
`ltgt` is the HTML entity `&lt;&gt;` left in the raw dataset, a nice honest
example of data artefacts leaking into a model.

**Local — LIME.** *Local Interpretable Model-agnostic Explanations* explains one
specific prediction by perturbing the input (randomly deleting words), watching
how the prediction moves, and fitting a simple local model to that behaviour:

```python
explainer = LimeTextExplainer(class_names=['Spam', 'Ham'])

def lime_predict(texts):
    return best_tuned.predict_proba([clean_text(t) for t in texts])

exp = explainer.explain_instance(spam_msg, lime_predict, num_features=12, num_samples=500)
```

`num_samples=500` is the perturbation budget: 500 variations of the message,
each scored. More samples means a more stable explanation and a slower run.

### What the app uses, and why it is better here

The app does **not** run LIME. For a linear model it does not need to: the exact
contribution of every term is `tfidf_value × coefficient`, which is the model's
own arithmetic rather than an approximation of it — and it is instant, while
LIME needs 500 extra predictions per explanation.

**LIME earns its place when the model is a black box** (a random forest, a
gradient-boosted ensemble, a neural network), where no such closed form exists.
Knowing when *not* to reach for the fancier tool is the point.

> **Interview question:** *Why does your app explain predictions with coefficients instead of LIME, when Phase 9 is about LIME?*
> Because the shipped model is linear, so the exact answer is available in
> closed form — `tfidf × coefficient` *is* the contribution, not an estimate of
> it. LIME approximates that for models where it cannot be computed directly.
> Phase 9 demonstrates the technique and confirms both views agree; the app uses
> the exact one because it is both more faithful and ~500× cheaper.

---

## Phase 10 — Documentation & Handover

**What exists and what each file is for:**

| File | Purpose |
|---|---|
| `README.md` | The GitHub landing page — what, how to run, results |
| `documentation/full_project_explanation.md` | This file — the deep technical walkthrough |
| `documentation/streamlit_app_guide.md` | How `app.py` and `backend.py` interact; Streamlit Cloud deployment |
| `documentation/phase_9_10_code_for_notebook.md` | Copy-paste cells for the LIME phase |
| `presentation/gamma_ai_ppt_prompt.txt` | Prompt that generates the slide deck |
| `presentation/ppt_presentation_script.md` | Word-for-word speaker script with Q&A prep |
| `goals.md` | Phase-by-phase progress tracker |
| `tests/test_backend.py` | 30 tests that gate a deploy |

---

## Numbers to have memorised

| Question | Answer |
|---|---|
| Dataset size | 5,572 messages (4,825 ham / 747 spam) |
| Class balance | 86.6% ham / 13.4% spam |
| Train / test split | 4,457 / 1,115 (80/20, `random_state=3`) |
| Models compared | 6 |
| Grid search size | 36 combinations × 5 folds = 180 fits |
| Vocabulary | 20,000 TF-IDF terms, unigrams + bigrams |
| Final `C` | 10.0 (L2 penalty) |
| Accuracy | 98.03% |
| ROC-AUC | 99.30% |
| Spam recall / precision | 87.74% / 97.84% |
| False positives | 3 out of 960 ham |
| Missed spam | 19 out of 155 |

---

## Honest limitations

Say these before someone else finds them — it reads as rigour, not weakness.

1. **The data is old and narrow.** UK SMS messages from around 2005. Modern
   email spam, other languages and emoji-heavy text are out of distribution.
2. **Bag of words has no memory of order.** "Not free" and "free not" are
   identical to this model; bigrams only partially compensate.
3. **It is beatable on purpose.** Spam written to avoid trigger words —
   `F.R.E.E`, `pr1ze` — slips straight through. Real filters add sender
   reputation, link analysis and header checks for this reason.
4. **Recall is the weak spot.** 12% of spam gets through at the default
   threshold. The slider can trade that away, but only against false alarms.
5. **Closed vocabulary.** Only the 20,000 terms seen in training carry weight;
   everything else is invisible. The app shows exactly which words were ignored.

## If this project continued

- **Transformer fine-tuning** (DistilBERT/BERT) — understands word order and
  context, and would likely close most of the recall gap.
- **A larger, modern corpus** (Enron, or a current phishing set) for real email
  rather than 2005 SMS.
- **Character n-grams** — would catch `F.R.E.E` and `pr1ze`, which word tokens
  cannot.
- **Threshold calibration on a validation set** rather than a hand-picked
  default.
- **Monitoring in production** — spam evolves, so a model that is never
  retrained decays. Track live precision/recall and retrain on a schedule.
