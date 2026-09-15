# 🎯 Project Goals — Spam Mail Prediction

> **Internship Project | 45-Day Plan**
> Tracking all planned phases and tasks for the project.

---

## ✅ Completed

- [x] **Setup & Data Loading**
  - [x] Load dataset from `data/spam.csv`
  - [x] Handle null values
  - [x] Rename columns to `Category` and `Message`
  - [x] Explore basic shape and structure of the data

- [x] **Baseline Model (Logistic Regression)**
  - [x] TF-IDF vectorization
  - [x] Train/test split (80/20)
  - [x] Logistic Regression training
  - [x] Accuracy score on training data (~96.70%)
  - [x] Accuracy score on test data (~96.59%)
  - [x] Basic predictive system for single message input

- [x] **Phase 1: Exploratory Data Analysis (EDA)**
  - [x] Class distribution bar chart and pie chart
  - [x] Message character count & word count statistics per class
  - [x] Length distribution histograms (spam vs ham)
  - [x] Box plots for length comparison
  - [x] WordCloud — top words in spam messages
  - [x] WordCloud — top words in ham messages
  - [x] Top 20 most frequent words per class (horizontal bar charts)
  - [x] Character-level feature engineering:
    - [x] Uppercase ratio
    - [x] Digit count
    - [x] Exclamation mark count
    - [x] URL presence flag
    - [x] Currency/reward word flag
  - [x] Correlation heatmap of engineered features vs label
  - [x] EDA Summary table of all findings

- [x] **Project Documentation — First Commit**
  - [x] `README.md` — GitHub project page
  - [x] `goals.md` — This file

---

## 🔄 In Progress

- [ ] **Phase 2: Advanced Text Preprocessing (NLTK)**
  - [ ] Install and import NLTK
  - [ ] Tokenization of messages
  - [ ] Stopword removal
  - [ ] Stemming (PorterStemmer)
  - [ ] Lemmatization (WordNetLemmatizer)
  - [ ] Build a `clean_text()` pipeline function
  - [ ] Compare TF-IDF matrix before and after cleaning

---

## ⬜ Pending

- [ ] **Phase 3: Multi-Model Comparison**
  - [ ] Naive Bayes (MultinomialNB)
  - [ ] Support Vector Machine (SVC)
  - [ ] Random Forest Classifier
  - [ ] Gradient Boosting (XGBoost)
  - [ ] K-Nearest Neighbors (KNN)
  - [ ] Side-by-side metrics comparison table
  - [ ] Grouped bar chart of all model metrics

- [ ] **Phase 4: Comprehensive Evaluation Metrics**
  - [ ] Confusion Matrix heatmap (seaborn)
  - [ ] Classification Report (precision, recall, F1 per class)
  - [ ] ROC Curve + AUC score
  - [ ] Precision-Recall Curve
  - [ ] 5-Fold Cross-Validation scores

- [ ] **Phase 5: Hyperparameter Tuning**
  - [ ] GridSearchCV on TF-IDF parameters (`ngram_range`, `max_features`, `min_df`, `max_df`)
  - [ ] GridSearchCV on best model parameters
  - [ ] Before vs after metrics comparison

- [ ] **Phase 6: Class Imbalance Handling (SMOTE)**
  - [ ] Show class imbalance clearly
  - [ ] Apply SMOTE oversampling
  - [ ] Apply `class_weight='balanced'` in models
  - [ ] Compare model metrics before vs after balancing

- [ ] **Phase 7: Model Saving & Deployment Pipeline**
  - [ ] Save best model using `joblib` → `model.pkl`
  - [ ] Save TF-IDF vectorizer → `vectorizer.pkl`
  - [ ] Write reusable `predict_email(text)` function
  - [ ] Demo: load model fresh and classify 5 example emails

- [ ] **Phase 8: Streamlit Web Application**
  - [ ] Create `app.py`
  - [ ] Text area input for user message
  - [ ] Classify button → Spam / Ham output
  - [ ] Show confidence score (probability %)
  - [ ] Show top contributing keywords
  - [ ] Test and verify app runs locally

- [ ] **Phase 9: Feature Importance & Explainability**
  - [ ] Top spam/ham words from LR/SVM model coefficients
  - [ ] Horizontal bar chart of feature weights
  - [ ] LIME explanation for individual predictions

- [ ] **Phase 10: Final Documentation**
  - [ ] Update `README.md` with final results
  - [ ] Add `requirements.txt`
  - [ ] Final project summary section in notebook
  - [ ] Add multi-dataset generalization test (Enron dataset)

---

## 📊 Progress Tracker

```
Phase 1  ████████████████████  100%  ✅ EDA
Phase 2  ░░░░░░░░░░░░░░░░░░░░    0%  🔄 Text Preprocessing
Phase 3  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Multi-Model
Phase 4  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Evaluation Metrics
Phase 5  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Hyperparameter Tuning
Phase 6  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ SMOTE
Phase 7  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Model Saving
Phase 8  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Streamlit App
Phase 9  ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Explainability
Phase 10 ░░░░░░░░░░░░░░░░░░░░    0%  ⬜ Final Docs

Overall: ██░░░░░░░░░░░░░░░░░░   ~12% Complete
```

---

_Last updated: September 2026_
