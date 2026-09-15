"""
Sanity tests for the inference layer.

These run without Streamlit and without a browser, so they can gate a deploy:

    pip install pytest
    pytest -q

They check the things that would silently break the app — the label convention,
the cleaning pipeline, threshold behaviour and the explanation arithmetic —
rather than re-measuring model accuracy, which the notebook already does.
"""

import math

import pytest

import backend
from backend import HAM, SPAM

SPAM_EXAMPLES = [
    "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121!",
    "Congratulations! You WON a £1000 prize. Call 0800-123456 NOW to claim.",
    "URGENT! Your mobile number has won a guaranteed cash award. Reply WIN now.",
]

HAM_EXAMPLES = [
    "Hey, are you coming to the party tonight? Let me know when you're free!",
    "Mom says dinner is ready, come home now.",
    "I'll be there in ten minutes, just parking the car.",
]


@pytest.fixture(scope="module")
def predictor():
    return backend.get_predictor()


# ── Text cleaning ────────────────────────────────────────────────────────────

def test_clean_text_strips_urls_digits_and_punctuation():
    cleaned = backend.clean_text("Visit http://spam.com NOW!!! Call 87121.")
    assert "http" not in cleaned
    assert not any(character.isdigit() for character in cleaned)
    assert "!" not in cleaned and "." not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_drops_stopwords_and_single_characters():
    assert backend.clean_text("I am a the and of it") == ""


def test_clean_text_handles_empty_and_symbol_only_input():
    assert backend.clean_text("") == ""
    assert backend.clean_text("!!! ??? ...") == ""


def test_fallback_lemmatizer_handles_common_plurals():
    cases = {
        "prizes": "prize", "tickets": "ticket", "boxes": "box",
        "entries": "entry", "lives": "life", "women": "woman",
        "miss": "miss", "bus": "bus", "cash": "cash",
    }
    for word, expected in cases.items():
        assert backend._fallback_lemmatize(word) == expected, word


# ── Label convention ─────────────────────────────────────────────────────────

def test_label_convention_matches_the_notebook():
    """spam=0 / ham=1 is baked into the pickle; flipping it inverts the app."""
    assert SPAM == 0 and HAM == 1
    assert backend.LABEL_NAMES[SPAM] == "Spam"


def test_classifier_classes_are_ordered_spam_then_ham(predictor):
    assert list(predictor.classifier.classes_) == [SPAM, HAM]


# ── Predictions ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("message", SPAM_EXAMPLES)
def test_known_spam_is_flagged(predictor, message):
    result = predictor.predict(message)
    assert result.is_spam, f"{message!r} -> P(spam)={result.spam_prob:.3f}"


@pytest.mark.parametrize("message", HAM_EXAMPLES)
def test_known_ham_is_not_flagged(predictor, message):
    result = predictor.predict(message)
    assert not result.is_spam, f"{message!r} -> P(spam)={result.spam_prob:.3f}"


def test_probabilities_are_valid_and_sum_to_one(predictor):
    result = predictor.predict(SPAM_EXAMPLES[0])
    assert 0.0 <= result.spam_prob <= 1.0
    assert math.isclose(result.spam_prob + result.ham_prob, 1.0, abs_tol=1e-9)


def test_confidence_reports_the_winning_class(predictor):
    spam = predictor.predict(SPAM_EXAMPLES[0])
    ham = predictor.predict(HAM_EXAMPLES[0])
    assert spam.confidence == spam.spam_prob
    assert ham.confidence == ham.ham_prob
    assert spam.confidence >= 0.5 and ham.confidence >= 0.5


def test_threshold_shifts_the_decision_boundary(predictor):
    message = SPAM_EXAMPLES[0]
    assert predictor.predict(message, threshold=0.05).is_spam
    assert not predictor.predict(message, threshold=0.99).is_spam


def test_threshold_does_not_change_probabilities(predictor):
    """Only the verdict moves with the threshold — the model output is fixed."""
    low = predictor.predict(SPAM_EXAMPLES[0], threshold=0.10)
    high = predictor.predict(SPAM_EXAMPLES[0], threshold=0.90)
    assert low.spam_prob == high.spam_prob


def test_batch_matches_single_predictions(predictor):
    messages = SPAM_EXAMPLES + HAM_EXAMPLES
    batch = predictor.predict_many(messages)
    assert len(batch) == len(messages)
    for message, batched in zip(messages, batch):
        assert batched.spam_prob == predictor.predict(message).spam_prob


def test_empty_message_does_not_crash(predictor):
    result = predictor.predict("")
    assert result.cleaned == ""
    assert 0.0 <= result.spam_prob <= 1.0


# ── Explanations ─────────────────────────────────────────────────────────────

def test_explanation_terms_come_from_the_message(predictor):
    contributions = predictor.explain("FREE cash prize, claim now!", top_n=10)
    assert contributions
    cleaned_tokens = set(backend.clean_text("FREE cash prize, claim now!").split())
    for contribution in contributions:
        # Unigrams must be present; bigrams are built from adjacent tokens.
        assert set(contribution.term.split()) <= cleaned_tokens


def test_explanation_is_ranked_by_absolute_contribution(predictor):
    contributions = predictor.explain(SPAM_EXAMPLES[0], top_n=8)
    magnitudes = [abs(c.contribution) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_agrees_with_the_models_own_score(predictor):
    """Σ(tfidf × coef) over ALL terms + intercept must equal decision_function."""
    message = SPAM_EXAMPLES[0]
    vector = predictor.vectorizer.transform([backend.clean_text(message)])
    reconstructed = float(
        vector.dot(predictor.classifier.coef_[0])[0] + predictor.classifier.intercept_[0]
    )
    actual = float(predictor.pipeline.decision_function([backend.clean_text(message)])[0])
    assert math.isclose(reconstructed, actual, rel_tol=1e-6, abs_tol=1e-9)


def test_spam_words_push_toward_spam(predictor):
    contributions = predictor.explain("free cash prize claim urgent", top_n=5)
    assert any(c.pushes_toward == "spam" for c in contributions)


def test_unknown_message_has_no_explanation(predictor):
    assert predictor.explain("zzzqqq wwwxxx vvvyyy") == []


def test_unknown_tokens_are_reported(predictor):
    unknown = predictor.unknown_tokens("zzzqqq free offer")
    assert "zzzqqq" in unknown
    assert "free" not in unknown


# ── Message features and metadata ────────────────────────────────────────────

def test_message_features_detect_spam_signals():
    features = backend.message_features("WIN £1000 CASH NOW!! http://bit.ly/x 87121")
    assert features["has_url"]
    assert features["has_currency_or_prize"]
    assert features["has_long_number"]
    assert features["exclamation_count"] == 2
    assert features["uppercase_ratio"] > 0.2


def test_message_features_are_quiet_on_normal_text():
    features = backend.message_features("see you at lunch tomorrow")
    assert not features["has_url"]
    assert not features["has_currency_or_prize"]
    assert features["exclamation_count"] == 0


def test_info_exposes_what_the_sidebar_renders(predictor):
    info = predictor.info()
    for key in ("estimator", "vocabulary_size", "ngram_range",
                "sklearn_running", "version_mismatch", "nlp_backend"):
        assert key in info
    assert info["vocabulary_size"] > 0
    assert info["nlp_backend"] in {"nltk", "fallback"}


def test_global_top_terms_are_signed_correctly(predictor):
    spam_terms, ham_terms = predictor.global_top_terms(n=10)
    assert all(weight < 0 for _, weight in spam_terms)   # negative => spam
    assert all(weight > 0 for _, weight in ham_terms)    # positive => ham


def test_missing_model_path_raises_a_helpful_error():
    with pytest.raises(backend.ModelNotFoundError):
        backend.resolve_model_path("/nonexistent/spam_model_pipeline.pkl")


def test_every_sample_message_lands_on_its_advertised_label(predictor):
    """The UI's quick-test buttons must actually demo what they claim."""
    for name, message in backend.SAMPLE_MESSAGES.items():
        result = predictor.predict(message)
        if name.startswith("Spam"):
            assert result.is_spam, f"{name} -> P(spam)={result.spam_prob:.3f}"
        elif name.startswith("Ham"):
            assert not result.is_spam, f"{name} -> P(spam)={result.spam_prob:.3f}"
