# 🎯 Project Goals — Spam Mail Prediction

> **Internship Project | 45-Day Plan**
> Phase-by-phase progress tracker.

---

## ✅ Completed

- [x] **Setup & Data Loading**
  - [x] Load dataset from `data/spam.csv`
  - [x] Handle null values
  - [x] Rename columns (`v1` → `Category`, `v2` → `Message`)
  - [x] Explore basic shape and structure

- [x] **Baseline Model (Logistic Regression)**
  - [x] TF-IDF vectorization
  - [x] Train/test split (80/20, random_state=3)
  - [x] Logistic Regression training
  - [x] Accuracy on training data (~96.70%)
  - [x] Accuracy on test data (~96.59%)

- [x] **Phase 1: Exploratory Data Analysis (EDA)**
  - [x] Class distribution bar chart and pie chart
  - [x] Message character count & word count stats per class
  - [x] Length distribution histograms (spam vs ham)
  - [x] Box plots for length comparison
  - [x] WordCloud — spam messages
  - [x] WordCloud — ham messages
  - [x] Top 20 most frequent words per class
  - [x] Character-level feature engineering:
    - [x] Uppercase ratio
    - [x] Digit count
    - [x] Exclamation mark count
    - [x] URL presence flag
    - [x] Currency/reward word flag
  - [x] Correlation heatmap of engineered features vs label
  - [x] EDA Summary table

- [x] **Phase 2: Advanced Text Preprocessing (NLTK)**
  - [x] Lowercase conversion
  - [x] URL removal using regex
  - [x] Number/digit removal
  - [x] Punctuation removal
  - [x] Tokenization
  - [x] Stopword removal (NLTK English stopwords)
  - [x] Lemmatization (WordNetLemmatizer)
  - [x] `clean_text()` pipeline function
  - [x] TF-IDF with bigrams (`ngram_range=(1,2)`, `max_features=15000`)
  - [x] Before vs after word count comparison

- [x] **Phase 3: Multi-Model Comparison**
  - [x] Logistic Regression
  - [x] Naive Bayes (MultinomialNB)
  - [x] Linear SVM (LinearSVC)
  - [x] Random Forest (200 estimators)
  - [x] Gradient Boosting Classifier
  - [x] K-Nearest Neighbors
  - [x] Metrics table (Accuracy, Precision, Recall, F1, ROC-AUC)
  - [x] Grouped bar chart comparing all models × all metrics

- [x] **Phase 4: Comprehensive Evaluation Metrics**
  - [x] Classification Report (precision, recall, F1 per class)
  - [x] Confusion Matrix heatmap with TP/TN/FP/FN labels
  - [x] False Positive Rate & False Negative Rate analysis
  - [x] ROC Curves — all 6 models on one chart
  - [x] Precision-Recall Curves — all 6 models
  - [x] 5-Fold Cross-Validation with box plots

- [x] **Phase 5: Hyperparameter Tuning**
  - [x] Sklearn `Pipeline` (TF-IDF → LR)
  - [x] GridSearchCV over TF-IDF params (`ngram_range`, `max_features`, `min_df`)
  - [x] GridSearchCV over LR params (`C`, `penalty`)
  - [x] 5-fold CV scoring (`f1`)
  - [x] Before vs after metrics comparison bar chart

- [x] **Phase 6: Class Imbalance Handling (SMOTE)**
  - [x] Visualise class imbalance clearly
  - [x] Apply SMOTE oversampling
  - [x] Apply `class_weight='balanced'` in LR
  - [x] Compare: No balancing vs class_weight vs SMOTE
  - [x] Before/after class distribution bar charts

- [x] **Phase 7: Model Saving & Deployment Pipeline**
  - [x] Save full pipeline with `joblib` → `spam_model_pipeline.pkl`
  - [x] Save vectorizer separately → `vectorizer.pkl`
  - [x] Save model separately → `model.pkl`
  - [x] Load model fresh and verify predictions
  - [x] `predict_email(text)` function (raw text → label + confidence %)
  - [x] Test on 5 diverse example messages with confidence scores
  - [x] Colab `files.download()` for local use
  - [x] Feature importance chart (top 25 spam/ham indicator words from LR coefficients)

- [x] **Project Documentation**
  - [x] `README.md` — updated for all 7 phases with results table
  - [x] `goals.md` — this file

---

## 🔄 In Progress

- [ ] **Phase 8: Streamlit Web Application**
  - [ ] Create `app.py`
  - [ ] Text area input for user to paste message
  - [ ] "Classify" button → Spam / Ham output with emoji
  - [ ] Confidence score display (probability %)
  - [ ] Top contributing keywords shown
  - [ ] Test that `streamlit run app.py` works locally

---

## ⬜ Pending

- [ ] **Phase 9: Feature Importance & LIME Explainability**
  - [ ] LIME explanation for individual predictions
  - [ ] Show which words drove a specific classification

- [ ] **Phase 10: Final Documentation**
  - [ ] `requirements.txt`
  - [ ] Final summary section in notebook
  - [ ] Multi-dataset test (Enron dataset generalization)
  - [ ] Final README polish

---

## 📊 Progress Tracker

```
Phase 1  ████████████████████  100%  ✅ EDA
Phase 2  ████████████████████  100%  ✅ Text Preprocessing
Phase 3  ████████████████████  100%  ✅ Multi-Model
Phase 4  ████████████████████  100%  ✅ Evaluation Metrics
Phase 5  ████████████████████  100%  ✅ Hyperparameter Tuning
Phase 6  ████████████████████  100%  ✅ SMOTE
Phase 7  ████████████████████  100%  ✅ Model Saving
Phase 8  ░░░░░░░░░░░░░░░░░░░░    0%  🔄 Streamlit App
Phase 9  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Explainability
Phase 10 ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Final Docs

Overall: ██████████████░░░░░░   ~72% Complete
```

---

_Last updated: September 2026_
