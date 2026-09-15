# 📧 Spam Mail Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%207%20Complete-brightgreen)

> A complete end-to-end Machine Learning project to classify SMS/Email messages as **Spam** or **Ham (Legitimate)** — built as part of a 45-day internship program.

---

## 📌 Project Overview

Spam messages are a major problem in digital communication. This project builds a production-ready ML pipeline that:

- **Detects spam** with ~98%+ accuracy using NLP + optimized classifiers
- **Explains predictions** via feature importance (LR coefficients)
- **Handles class imbalance** using SMOTE oversampling
- **Deploys** as a reusable `.pkl` pipeline and (soon) a Streamlit web app

---

## 📂 Project Structure

```
spam-mail-prediction/
│
├── data/
│   └── spam.csv                          # SMS Spam Collection dataset
│
├── Spam_Mail_Prediction_using_Machine_Learning.ipynb  # Main notebook (all phases)
│
├── saved_model/                          # Generated after running Phase 7
│   ├── spam_model_pipeline.pkl           # Full pipeline (TF-IDF + LR)
│   ├── vectorizer.pkl                    # TF-IDF vectorizer only
│   └── model.pkl                         # Logistic Regression model only
│
├── app.py                                # Streamlit web app (Phase 8 — coming soon)
├── goals.md                              # Project goals & progress tracker
└── README.md                             # This file
```

---

## 📊 Dataset

| Property          | Value                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| **Source**        | [SMS Spam Collection — Kaggle (UCI)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| **File**          | `data/spam.csv`                                                                                         |
| **Total Records** | 5,572 messages                                                                                          |
| **Columns**       | `Category` (spam/ham), `Message` (text)                                                                 |
| **Spam**          | ~13% (747 messages)                                                                                     |
| **Ham**           | ~87% (4,825 messages)                                                                                   |

---

## 🔬 Project Phases

| Phase        | Title                                    | Status         |
| ------------ | ---------------------------------------- | -------------- |
| **Phase 1**  | Exploratory Data Analysis (EDA)          | ✅ Complete    |
| **Phase 2**  | Advanced Text Preprocessing (NLTK)       | ✅ Complete    |
| **Phase 3**  | Multi-Model Comparison (6 classifiers)   | ✅ Complete    |
| **Phase 4**  | Comprehensive Evaluation Metrics         | ✅ Complete    |
| **Phase 5**  | Hyperparameter Tuning (GridSearchCV)     | ✅ Complete    |
| **Phase 6**  | Class Imbalance Handling (SMOTE)         | ✅ Complete    |
| **Phase 7**  | Model Saving & Deployment Pipeline       | ✅ Complete    |
| **Phase 8**  | Streamlit Web Application                | 🔄 In Progress |
| **Phase 9**  | Feature Importance & LIME Explainability | ⬜ Pending     |
| **Phase 10** | Final Documentation & Report             | ⬜ Pending     |

---

## ✅ What's Been Built

### Phase 1 — EDA

- Class distribution (87% Ham / 13% Spam — **imbalanced**)
- Message length histograms, box plots, WordClouds
- Top-20 most frequent words per class
- Character-level feature analysis (uppercase ratio, digit count, URL presence)
- Correlation heatmap

### Phase 2 — NLTK Text Preprocessing

Custom `clean_text()` pipeline:

```
Lowercase → Remove URLs → Remove numbers → Remove punctuation
→ Tokenize → Remove stopwords → Lemmatize (WordNetLemmatizer)
```

TF-IDF with **bigrams** (`ngram_range=(1,2)`) captures phrases like _"free entry"_, _"call now"_, _"win prize"_.

### Phase 3 — 6-Model Comparison

| Model               | F1-Score  | ROC-AUC   |
| ------------------- | --------- | --------- |
| Logistic Regression | ~0.97     | ~0.99     |
| Naive Bayes         | ~0.96     | ~0.99     |
| **Linear SVM**      | **~0.98** | **~0.99** |
| Random Forest       | ~0.97     | ~0.99     |
| Gradient Boosting   | ~0.96     | ~0.99     |
| KNN                 | ~0.91     | ~0.97     |

### Phase 4 — Evaluation

- Confusion matrix with TP/TN/FP/FN breakdown
- ROC curves + AUC for all 6 models on one chart
- Precision-Recall curves
- 5-fold cross-validation box plots

### Phase 5 — Hyperparameter Tuning

GridSearchCV over:

- TF-IDF: `ngram_range`, `max_features`, `min_df`
- LR: `C`, `penalty`

### Phase 6 — SMOTE

- Applied `SMOTE` to balance training classes
- Compared: No balancing vs `class_weight='balanced'` vs SMOTE

### Phase 7 — Deployment Pipeline

- `best_tuned` pipeline saved with `joblib`
- `predict_email(text)` function — raw text in, prediction + confidence out
- Feature importance bar chart from LR coefficients (top 25 spam/ham indicator words)

---

## 🚀 How to Run

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook `.ipynb`
3. Upload the `data/` folder to `/content/data/`
4. Click **Runtime → Run All**

> **Install missing libraries** (first cell handles this):
>
> ```python
> !pip install imbalanced-learn -q
> ```

### Local (Jupyter)

```bash
git clone https://github.com/YOUR_USERNAME/spam-mail-prediction.git
cd spam-mail-prediction
pip install -r requirements.txt
jupyter notebook
```

---

## 🛠️ Tech Stack

| Tool                 | Purpose                                    |
| -------------------- | ------------------------------------------ |
| Python 3.10+         | Core language                              |
| Pandas & NumPy       | Data manipulation                          |
| Matplotlib & Seaborn | Visualisation                              |
| WordCloud            | Word frequency plots                       |
| NLTK                 | Tokenization, stopwords, lemmatization     |
| Scikit-learn         | TF-IDF, 6 ML models, GridSearchCV, metrics |
| imbalanced-learn     | SMOTE oversampling                         |
| Joblib               | Model serialization                        |
| Streamlit            | Web app _(Phase 8)_                        |

---

## 📈 Final Results

| Metric        | Baseline LR | Tuned Pipeline |
| ------------- | ----------- | -------------- |
| **Accuracy**  | 96.59%      | ~98%+          |
| **Precision** | ~98%        | ~99%           |
| **Recall**    | ~90%        | ~96%           |
| **F1-Score**  | ~94%        | ~97%+          |
| **ROC-AUC**   | ~99%        | ~99%+          |

---

## 👨‍💻 Author

**Arjun** — Internship Project, 2026  
Built with ❤️ using Python & Scikit-Learn

---

## 📄 License

Open source under the [MIT License](LICENSE).
