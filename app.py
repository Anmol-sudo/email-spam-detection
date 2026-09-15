import streamlit as st
import joblib
import re
import string
import numpy as np

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("saved_model/spam_model_pipeline.pkl")
    return pipeline

pipeline = load_model()

# ── NLTK setup ────────────────────────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

def get_top_features(text, n=10):
    """Extract top contributing words from model coefficients."""
    try:
        tfidf = pipeline.named_steps['tfidf']
        clf   = pipeline.named_steps['clf']
        vec   = tfidf.transform([clean_text(text)])
        coef  = clf.coef_[0]
        feature_names = np.array(tfidf.get_feature_names_out())
        nonzero_idx   = vec.nonzero()[1]
        if len(nonzero_idx) == 0:
            return [], []
        scores = coef[nonzero_idx]
        words  = feature_names[nonzero_idx]
        sorted_idx = np.argsort(np.abs(scores))[::-1][:n]
        return words[sorted_idx].tolist(), scores[sorted_idx].tolist()
    except Exception:
        return [], []

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .spam-box {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        border-radius: 12px; padding: 24px; text-align: center;
        color: white; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 20px rgba(255,71,87,0.5);
    }
    .ham-box {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        border-radius: 12px; padding: 24px; text-align: center;
        color: white; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 20px rgba(46,204,113,0.5);
    }
    .confidence-bar { border-radius: 8px; height: 20px; margin-top: 10px; }
    .feature-tag {
        display: inline-block; margin: 4px;
        padding: 4px 10px; border-radius: 20px;
        font-size: 13px; font-weight: 600;
    }
    .stat-card {
        background-color: #1e1e2e; border-radius: 10px;
        padding: 16px; text-align: center; color: white;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("# 📧 Spam Mail Detector")
st.markdown("**AI-powered SMS & Email spam classifier** | Logistic Regression + TF-IDF Bigrams")
st.divider()

# ── Input ─────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area(
        "Paste your message here:",
        height=150,
        placeholder="e.g.  FREE entry! Win £1000 cash prize. Text WIN to 87121 now...",
    )

with col2:
    st.markdown("#### Quick Tests")
    if st.button("🚨 Spam example"):
        st.session_state["sample"] = "FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121 to receive entry now!"
    if st.button("✅ Ham example"):
        st.session_state["sample"] = "Hey, are you coming to the party tonight? Let me know when you're free!"
    if st.button("⚠️ Phishing example"):
        st.session_state["sample"] = "URGENT: Your account has been suspended. Click http://verify-now.com to restore access immediately."

if "sample" in st.session_state:
    user_input = st.session_state["sample"]
    del st.session_state["sample"]
    st.rerun()

# ── Classify button ───────────────────────────────────────────────
classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)

if classify_btn and user_input.strip():
    cleaned = clean_text(user_input)
    pred    = pipeline.predict([cleaned])[0]
    proba   = pipeline.predict_proba([cleaned])[0]
    conf    = proba[pred] * 100
    spam_prob = proba[0] * 100
    ham_prob  = proba[1] * 100

    st.divider()

    # ── Result box ───────────────────────────────────────────────
    if pred == 0:  # Spam
        st.markdown(f'<div class="spam-box">🚨 SPAM DETECTED<br><small style="font-size:16px">Confidence: {conf:.1f}%</small></div>', unsafe_allow_html=True)
    else:          # Ham
        st.markdown(f'<div class="ham-box">✅ Legitimate Message (Ham)<br><small style="font-size:16px">Confidence: {conf:.1f}%</small></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability bars ─────────────────────────────────────────
    st.markdown("#### 📊 Probability Breakdown")
    col_s, col_h = st.columns(2)
    with col_s:
        st.metric("🚨 Spam Probability", f"{spam_prob:.1f}%")
        st.progress(spam_prob / 100)
    with col_h:
        st.metric("✅ Ham Probability", f"{ham_prob:.1f}%")
        st.progress(ham_prob / 100)

    # ── Top features ─────────────────────────────────────────────
    st.markdown("#### 🔑 Key Words Driving This Prediction")
    words, scores = get_top_features(user_input, n=12)
    if words:
        tags_html = ""
        for w, s in zip(words, scores):
            color = "#e74c3c" if s < 0 else "#2ecc71"
            tags_html += f'<span class="feature-tag" style="background:{color};color:white">{w}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)
        st.caption("🔴 Red = spam indicator  |  🟢 Green = ham indicator")
    else:
        st.info("Not enough recognized tokens to show feature contributions.")

    # ── Message stats ────────────────────────────────────────────
    st.markdown("#### 📋 Message Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters", len(user_input))
    c2.metric("Words", len(user_input.split()))
    c3.metric("UPPER ratio", f"{sum(c.isupper() for c in user_input)/(len(user_input)+1)*100:.1f}%")
    has_url = "Yes 🔗" if re.search(r'http|www|\.com', user_input, re.I) else "No"
    c4.metric("Has URL", has_url)

elif classify_btn and not user_input.strip():
    st.warning("Please enter a message to classify.")

# ── Sidebar: About ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
This app uses a **Logistic Regression** model trained on the SMS Spam Collection dataset (5,572 messages).

**Pipeline:**
1. NLTK text cleaning
2. TF-IDF Bigrams (top 20k features)
3. Logistic Regression (tuned)

**Performance:**
- Accuracy: ~98%
- F1-Score: ~97%
- ROC-AUC: ~99%

**Built by:** Arjun | Internship 2026
    """)
    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](#) | [Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("📧 Spam Mail Detector | Built with Scikit-Learn & Streamlit | Internship Project 2026")

