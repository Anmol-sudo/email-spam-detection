# 📧 Spam Mail Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter%20%7C%20Google%20Colab-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> A complete end-to-end Machine Learning project to classify SMS/Email messages as **Spam** or **Ham (Legitimate)** — built as part of a 45-day internship program.

---

## 📌 Project Overview

Spam messages are a major problem in digital communication. This project builds a machine learning pipeline that automatically detects whether a message is spam or not, using Natural Language Processing (NLP) techniques.

The project goes beyond a basic tutorial by covering:

- Deep Exploratory Data Analysis (EDA)
- Advanced text preprocessing (NLTK)
- Multiple model comparison
- Hyperparameter tuning
- Class imbalance handling
- An interactive web application (Streamlit)

---

## 📂 Project Structure

```
spam-mail-prediction/
│
├── data/
│   └── spam.csv                          # SMS Spam Collection dataset
│
├── Spam_Mail_Prediction_using_Machine_Learning.ipynb  # Main notebook
│
├── app.py                                # Streamlit web app (Phase 8)
├── model.pkl                             # Saved trained model (Phase 7)
├── vectorizer.pkl                        # Saved TF-IDF vectorizer (Phase 7)
│
├── goals.md                              # Project goals & progress tracker
└── README.md                             # This file
```

---

## 📊 Dataset

| Property          | Value                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| **Source**        | [SMS Spam Collection — Kaggle (UCI)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| **Total Records** | 5,572 messages                                                                                          |
| **Columns**       | `Category` (spam/ham), `Message` (text)                                                                 |
| **Spam**          | ~13% (747 messages)                                                                                     |
| **Ham**           | ~87% (4,825 messages)                                                                                   |

---

## 🔬 Project Phases

| Phase        | Title                               | Status         |
| ------------ | ----------------------------------- | -------------- |
| **Phase 1**  | Exploratory Data Analysis (EDA)     | ✅ Complete    |
| **Phase 2**  | Advanced Text Preprocessing (NLTK)  | 🔄 In Progress |
| **Phase 3**  | Multi-Model Comparison              | ⬜ Pending     |
| **Phase 4**  | Comprehensive Evaluation Metrics    | ⬜ Pending     |
| **Phase 5**  | Hyperparameter Tuning               | ⬜ Pending     |
| **Phase 6**  | Class Imbalance Handling (SMOTE)    | ⬜ Pending     |
| **Phase 7**  | Model Saving & Deployment Pipeline  | ⬜ Pending     |
| **Phase 8**  | Streamlit Web Application           | ⬜ Pending     |
| **Phase 9**  | Feature Importance & Explainability | ⬜ Pending     |
| **Phase 10** | Final Documentation & Report        | ⬜ Pending     |

---

## ✅ Phase 1 — EDA Highlights

| Finding                  | Detail                              |
| ------------------------ | ----------------------------------- |
| Class Imbalance          | 87% Ham, 13% Spam                   |
| Spam messages are longer | Avg ~139 chars vs ~72 chars for ham |
| Spam uses more uppercase | Aggressive marketing tone           |
| Spam uses more digits    | Phone numbers, prize amounts        |
| Spam uses more `!`       | Urgency-driven language             |
| Key spam words           | FREE, call, win, prize, text, claim |
| Key ham words            | ok, come, get, going, know, like    |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook: `Spam_Mail_Prediction_using_Machine_Learning.ipynb`
3. Upload the `data/` folder to `/content/data/`
4. Click **Runtime → Run All**

### Option 2: Local (Jupyter)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/spam-mail-prediction.git
cd spam-mail-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Option 3: Streamlit Web App _(Phase 8 — Coming Soon)_

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool                 | Purpose                      |
| -------------------- | ---------------------------- |
| Python 3.10+         | Core language                |
| Pandas & NumPy       | Data manipulation            |
| Matplotlib & Seaborn | Data visualization           |
| WordCloud            | Word frequency visualization |
| NLTK                 | Text preprocessing           |
| Scikit-learn         | ML models & evaluation       |
| XGBoost              | Gradient boosting model      |
| imbalanced-learn     | SMOTE for class imbalance    |
| Streamlit            | Web application              |
| Joblib               | Model serialization          |

---

## 📈 Results (Phase 1 — Baseline Model)

| Metric       | Training | Test   |
| ------------ | -------- | ------ |
| **Accuracy** | 96.70%   | 96.59% |

> More metrics (Precision, Recall, F1, ROC-AUC) will be added in Phase 4.

---

## 👨‍💻 Author

**Arjun**

- Internship Project — 2026
- Built with ❤️ using Python & Scikit-Learn

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
