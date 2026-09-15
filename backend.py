"""
Backend / inference layer for the Spam Mail Detector.

Everything that is *not* user-interface lives here: locating and loading the
trained pipeline, reproducing the exact Phase-2 NLTK cleaning used at training
time, running predictions, and explaining them.

`app.py` (the Streamlit front-end) imports this module and never touches the
model directly. Keeping the two apart means:

  * the same inference code can be reused by a CLI, a FastAPI service or a test
    suite without dragging Streamlit in,
  * the UI can be rewritten without any risk of changing model behaviour,
  * `pytest tests/` can verify the model end-to-end with no browser involved.

IMPORTANT — label convention (must match the notebook, Section 4):
    spam -> 0      ham -> 1
so `predict_proba(...)[0]` is P(spam) and `[1]` is P(ham).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import string
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np

# ── Label constants ───────────────────────────────────────────────────────────
SPAM, HAM = 0, 1
LABEL_NAMES = {SPAM: "Spam", HAM: "Ham"}

# Default decision threshold on P(spam). 0.50 is the plain argmax boundary the
# model was scored with; see docs for the precision/recall trade-off table.
DEFAULT_THRESHOLD = 0.50

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = "spam_model_pipeline.pkl"

# Where to look for the pickle, in priority order. `saved_model/` is what the
# notebook writes (Phase 7); the project root is kept as a fallback so an older
# checkout, or a Colab download dropped next to app.py, still works.
MODEL_SEARCH_DIRS = (
    PROJECT_ROOT / "saved_model",
    PROJECT_ROOT,
    PROJECT_ROOT / "model",
    Path.cwd() / "saved_model",
    Path.cwd(),
)

# scikit-learn version the pickle was created with. Loading under a different
# version still usually works but scikit-learn emits InconsistentVersionWarning
# and silently-wrong results are possible, so we surface it in the UI instead.
TRAINED_SKLEARN_VERSION = "1.6.1"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained pipeline cannot be located on disk."""


# ══════════════════════════════════════════════════════════════════════════════
# 1. Text cleaning — must mirror `clean_text()` from Phase 2 of the notebook
# ══════════════════════════════════════════════════════════════════════════════

# Frozen copy of NLTK's English stopword list. Used only when the NLTK download
# fails at runtime (no network on the host, corpora not yet cached, ...), so the
# app degrades gracefully instead of crashing on start-up.
_FALLBACK_STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren as at be because been before
being below between both but by can couldn d did didn do does doesn doing don down during each
few for from further had hadn has hasn have haven having he her here hers herself him himself
his how i if in into is isn it its itself just ll m ma me mightn more most mustn my myself needn
no nor not now o of off on once only or other our ours ourselves out over own re s same shan she
should shouldn so some such t than that the their theirs them themselves then there these they
this those through to too under until up ve very was wasn we were weren what when where which
while who whom why will with won wouldn y you your yours yourself yourselves
""".split())

# Plural rules approximating WordNet's `morphy`. Real morphy only applies a rule
# when the result exists in WordNet; without the corpus we cannot check that, so
# the ordering below encodes the common cases instead (boxes -> box, prizes ->
# prize, lives -> life) and leaves words like "miss" or "bus" untouched.
_SIBILANT_PLURALS = ("ches", "shes", "sses", "xes")
_KEEP_AS_IS_ENDINGS = ("ss", "us", "is", "ous", "s'")
_IRREGULAR_PLURALS = {
    "lives": "life", "wives": "wife", "knives": "knife", "leaves": "leaf",
    "men": "man", "women": "woman", "children": "child", "people": "person",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "wolves": "wolf",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_URL_RE = re.compile(r"http\S+|www\S+")
_DIGIT_RE = re.compile(r"\d+")


def _fallback_lemmatize(token: str) -> str:
    """Approximate WordNetLemmatizer(pos='n') without the WordNet corpus."""
    if token in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[token]
    if len(token) <= 3 or not token.endswith("s") or token.endswith(_KEEP_AS_IS_ENDINGS):
        return token
    if token.endswith("ies") and len(token) > 4:   # entries -> entry
        return token[:-3] + "y"
    if token.endswith(_SIBILANT_PLURALS):
        return token[:-2]
    return token[:-1]


def _corpus_available(nltk, corpus: str) -> bool:
    """True if an NLTK corpus is already on disk, extracted or still zipped."""
    for probe in (f"corpora/{corpus}", f"corpora/{corpus}.zip"):
        try:
            nltk.data.find(probe)
            return True
        except LookupError:
            continue
    return False


def _download_quietly(nltk, corpus: str, download_dir: str) -> None:
    """Fetch a corpus, swallowing both the exception and NLTK's stderr chatter."""
    logger = logging.getLogger("nltk")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            nltk.download(corpus, quiet=True, download_dir=download_dir)
    except Exception:  # network blocked / mirror down — the fallback takes over
        pass
    finally:
        logger.setLevel(previous_level)


@lru_cache(maxsize=1)
def _load_nlp_tools() -> tuple[frozenset, object, str]:
    """
    Return (stopwords, lemmatize_fn, backend_name).

    Tries NLTK first (what the model was trained with); falls back to the frozen
    stopword list plus rule-based lemmatizer if the corpora cannot be obtained.
    """
    try:
        import nltk

        # Keep downloads inside a writable dir — Streamlit Cloud allows $HOME,
        # but honour NLTK_DATA when the host sets it.
        download_dir = os.environ.get("NLTK_DATA") or str(Path.home() / "nltk_data")
        os.makedirs(download_dir, exist_ok=True)
        if download_dir not in nltk.data.path:
            nltk.data.path.insert(0, download_dir)

        # Only hit the network for corpora that are genuinely missing: on a warm
        # Streamlit Cloud container they are already cached, and a blocked
        # download should stay silent rather than spraying the logs. Corpora may
        # sit on disk either extracted or still zipped, so probe for both.
        for corpus in ("stopwords", "wordnet", "omw-1.4"):
            if _corpus_available(nltk, corpus):
                continue
            _download_quietly(nltk, corpus, download_dir)

        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer

        stop = frozenset(nltk_stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tickets")  # force WordNet load, fails loudly here
        return stop, lemmatizer.lemmatize, "nltk"
    except Exception:
        return _FALLBACK_STOPWORDS, _fallback_lemmatize, "fallback"


def nlp_backend() -> str:
    """'nltk' when the real corpora loaded, 'fallback' when approximating."""
    return _load_nlp_tools()[2]


def clean_text(text: str) -> str:
    """
    Phase-2 cleaning pipeline, identical to the notebook:

        lowercase -> strip URLs -> strip digits -> strip punctuation
        -> tokenize on whitespace -> drop stopwords and 1-char tokens
        -> lemmatize

    The TF-IDF vocabulary inside the pickle was fitted on text produced by this
    exact function, so any drift here silently degrades accuracy.
    """
    stop, lemmatize, _ = _load_nlp_tools()
    text = str(text).lower()
    text = _URL_RE.sub("", text)
    text = _DIGIT_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    return " ".join(
        lemmatize(token)
        for token in text.split()
        if token not in stop and len(token) > 1
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Rule-based message features (the Phase-1 EDA signals, shown in the UI)
# ══════════════════════════════════════════════════════════════════════════════

_CURRENCY_RE = re.compile(r"[\$£€]|\b(free|win|prize|cash|claim|winner)\b", re.I)
_URL_HINT_RE = re.compile(r"http|www|\.com|\.net|bit\.ly", re.I)
_PHONE_RE = re.compile(r"\b\d{5,}\b")


def message_features(text: str) -> dict:
    """Surface-level statistics — the same features explored in Phase 1 EDA."""
    length = len(text)
    words = text.split()
    return {
        "char_count": length,
        "word_count": len(words),
        "uppercase_ratio": sum(c.isupper() for c in text) / (length + 1),
        "digit_count": sum(c.isdigit() for c in text),
        "exclamation_count": text.count("!"),
        "has_url": bool(_URL_HINT_RE.search(text)),
        "has_currency_or_prize": bool(_CURRENCY_RE.search(text)),
        "has_long_number": bool(_PHONE_RE.search(text)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Prediction result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Prediction:
    """One classified message, with everything the UI needs to render it."""

    text: str
    cleaned: str
    label: int                    # SPAM (0) or HAM (1)
    spam_prob: float              # P(spam), 0-1
    ham_prob: float               # P(ham), 0-1
    threshold: float              # decision threshold applied to spam_prob
    features: dict = field(default_factory=dict)

    @property
    def is_spam(self) -> bool:
        return self.label == SPAM

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]

    @property
    def confidence(self) -> float:
        """Probability assigned to the label that was actually returned."""
        return self.spam_prob if self.is_spam else self.ham_prob

    @property
    def is_borderline(self) -> bool:
        """True when the model is within 15 points of the decision boundary."""
        return abs(self.spam_prob - self.threshold) < 0.15

    def as_row(self) -> dict:
        """Flat dict for building a DataFrame in batch mode."""
        return {
            "message": self.text,
            "prediction": self.label_name,
            "spam_probability": round(self.spam_prob, 4),
            "ham_probability": round(self.ham_prob, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class TokenContribution:
    """One TF-IDF term and how much it pushed the decision."""

    term: str
    tfidf_value: float
    coefficient: float
    contribution: float           # tfidf_value * coefficient (logit units)

    @property
    def pushes_toward(self) -> str:
        # Coefficients are positive toward ham (class 1), negative toward spam.
        return "ham" if self.contribution > 0 else "spam"


# ══════════════════════════════════════════════════════════════════════════════
# 4. The predictor
# ══════════════════════════════════════════════════════════════════════════════

def resolve_model_path(explicit: str | os.PathLike | None = None) -> Path:
    """Find `spam_model_pipeline.pkl`, checking the usual locations."""
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise ModelNotFoundError(f"No model file at {path}")

    env_path = os.environ.get("SPAM_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    for directory in MODEL_SEARCH_DIRS:
        candidate = directory / MODEL_FILENAME
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(d / MODEL_FILENAME) for d in MODEL_SEARCH_DIRS)
    raise ModelNotFoundError(
        f"Could not find '{MODEL_FILENAME}'. Looked in:\n  - {searched}\n"
        "Run Phase 7 of the notebook to regenerate it, or set SPAM_MODEL_PATH."
    )


class SpamPredictor:
    """Thin, stateless wrapper around the trained TF-IDF -> LogisticRegression pipeline."""

    def __init__(self, model_path: str | os.PathLike | None = None):
        self.model_path = resolve_model_path(model_path)
        self.pipeline = joblib.load(self.model_path)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.classifier = self.pipeline.named_steps["clf"]
        self._feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        self._coef = self.classifier.coef_[0]

    # ── inference ────────────────────────────────────────────────────────────
    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Prediction:
        """Classify a single raw message."""
        return self.predict_many([text], threshold)[0]

    def predict_many(
        self, texts: Sequence[str], threshold: float = DEFAULT_THRESHOLD
    ) -> list[Prediction]:
        """Classify a batch of raw messages in one vectorizer pass."""
        texts = [str(t) for t in texts]
        cleaned = [clean_text(t) for t in texts]
        probabilities = self.pipeline.predict_proba(cleaned)

        results = []
        for raw, clean, proba in zip(texts, cleaned, probabilities):
            spam_prob = float(proba[SPAM])
            results.append(
                Prediction(
                    text=raw,
                    cleaned=clean,
                    label=SPAM if spam_prob >= threshold else HAM,
                    spam_prob=spam_prob,
                    ham_prob=float(proba[HAM]),
                    threshold=threshold,
                    features=message_features(raw),
                )
            )
        return results

    # ── explanation ──────────────────────────────────────────────────────────
    def explain(self, text: str, top_n: int = 12) -> list[TokenContribution]:
        """
        Explain a prediction from the model's own arithmetic.

        Logistic regression scores a message as  intercept + Σ (tfidf_i * coef_i).
        Ranking terms by that product — not by the raw coefficient — shows what
        actually moved *this* message, because a strong word that barely appears
        contributes little.
        """
        vector = self.vectorizer.transform([clean_text(text)])
        indices = vector.nonzero()[1]
        if indices.size == 0:
            return []

        values = np.asarray(vector[0, indices].todense()).ravel()
        coefficients = self._coef[indices]
        contributions = values * coefficients

        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        return [
            TokenContribution(
                term=str(self._feature_names[indices[i]]),
                tfidf_value=float(values[i]),
                coefficient=float(coefficients[i]),
                contribution=float(contributions[i]),
            )
            for i in order
        ]

    def global_top_terms(self, n: int = 20) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Strongest spam and ham indicators the model learned overall."""
        order = self._coef.argsort()
        spam_terms = [(str(self._feature_names[i]), float(self._coef[i])) for i in order[:n]]
        ham_terms = [(str(self._feature_names[i]), float(self._coef[i])) for i in order[-n:][::-1]]
        return spam_terms, ham_terms

    def unknown_tokens(self, text: str) -> list[str]:
        """Cleaned tokens the TF-IDF vocabulary has never seen (carry zero weight)."""
        vocabulary = self.vectorizer.vocabulary_
        return [t for t in clean_text(text).split() if t not in vocabulary]

    # ── metadata for the UI ──────────────────────────────────────────────────
    def info(self) -> dict:
        import sklearn

        return {
            "model_path": str(self.model_path),
            "estimator": type(self.classifier).__name__,
            "C": getattr(self.classifier, "C", None),
            "penalty": getattr(self.classifier, "penalty", None),
            "vocabulary_size": len(self._feature_names),
            "ngram_range": self.vectorizer.ngram_range,
            "max_features": self.vectorizer.max_features,
            "min_df": self.vectorizer.min_df,
            "sklearn_running": sklearn.__version__,
            "sklearn_trained": TRAINED_SKLEARN_VERSION,
            "version_mismatch": sklearn.__version__ != TRAINED_SKLEARN_VERSION,
            "nlp_backend": nlp_backend(),
        }


@lru_cache(maxsize=1)
def get_predictor(model_path: str | None = None) -> SpamPredictor:
    """Process-wide singleton — the pickle is loaded at most once."""
    return SpamPredictor(model_path)


# ── Sample messages used by the UI's quick-test buttons ──────────────────────
SAMPLE_MESSAGES = {
    "Spam — prize draw": (
        "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
        "Text FA to 87121 to receive entry question(std txt rate)"
    ),
    "Spam — fake winner": (
        "Congratulations! You have WON a £1000 Walmart gift card. "
        "Call 0800-123456 NOW to claim your prize before it expires!"
    ),
    "Phishing — account alert": (
        "URGENT: Your account has been suspended. Click http://verify-now.com "
        "to restore access immediately or it will be deleted."
    ),
    "Ham — friendly chat": (
        "Hey, are you coming to the party tonight? Let me know when you're free!"
    ),
    "Ham — everyday note": "Mom says dinner is ready, come home now.",
    "Ham — tricky (sounds salesy)": (
        "Don't forget the free coffee vouchers from work are in my bag, "
        "grab one before the meeting."
    ),
}


if __name__ == "__main__":  # quick smoke test:  python backend.py
    predictor = get_predictor()
    print(f"Loaded {predictor.model_path}  (NLP backend: {nlp_backend()})")
    for name, message in SAMPLE_MESSAGES.items():
        result = predictor.predict(message)
        print(f"  {name:<32} -> {result.label_name:<5} "
              f"P(spam)={result.spam_prob:.3f}")
