import re
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from algorithms.preprocessor import get_sentences
from algorithms.keyword_extractor import extract_keywords


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_quiz(transcript: str, num_questions: int = 8) -> list[dict]:
    """
    Generate fill-in-the-blank multiple-choice questions (MCQs).

    Algorithm
    ---------
    1. Extract the top keywords (TF-IDF unigrams + bigrams).
    2. For each keyword, locate the first sentence that contains it.
    3. Blank the keyword in that sentence → the question stem.
    4. Generate 3 plausible distractors via cosine similarity between
       per-term TF-IDF vectors (similar terms = realistic wrong answers).
    5. Shuffle answer options; record the correct answer.

    Returns
    -------
    List of {"question": str, "options": list[str], "answer": str} dicts.
    """
    sentences = get_sentences(transcript)
    keywords  = extract_keywords(transcript, n=30)
    kw_list   = [kw for kw, _ in keywords]

    if len(kw_list) < 4:
        return []   # not enough vocabulary for meaningful distractors

    # Fit a single vectorizer over the full corpus (used for cosine similarity)
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit([transcript])

    quiz: list[dict] = []
    used_sentences: set[str] = set()

    for keyword, _score in keywords[:num_questions]:
        target_sentence = _find_sentence(keyword, sentences, used_sentences)
        if not target_sentence:
            continue

        blanked = re.sub(
            re.escape(keyword), '_____',
            target_sentence, count=1, flags=re.IGNORECASE
        )

        distractors = _get_distractors(keyword, kw_list, vectorizer, transcript)
        if len(distractors) < 3:
            continue

        options = distractors[:3] + [keyword]
        random.shuffle(options)

        quiz.append({
            "question": blanked,
            "options":  options,
            "answer":   keyword,
        })

    return quiz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sentence(
    keyword: str,
    sentences: list[str],
    used: set[str],
) -> str | None:
    """Return the first unused sentence that contains *keyword* (case-insensitive)."""
    for sent in sentences:
        if keyword.lower() in sent.lower() and sent not in used:
            used.add(sent)
            return sent
    return None


def _get_distractors(
    keyword: str,
    all_keywords: list[str],
    vectorizer: TfidfVectorizer,
    corpus: str,
    n: int = 3,
) -> list[str]:
    """
    Return up to *n* distractors for *keyword* by ranking candidate keywords
    via cosine similarity of their TF-IDF column vectors.

    High similarity → plausible (but wrong) answer; low-similarity outliers are
    discarded so options are not obviously nonsensical.

    Falls back to a random sample if the vocabulary lookup fails.
    """
    try:
        tfidf_matrix = vectorizer.transform([corpus])
        vocab        = vectorizer.vocabulary_

        if keyword not in vocab:
            raise KeyError(keyword)

        kw_col  = tfidf_matrix[:, vocab[keyword]]
        scores: dict[str, float] = {}

        for word in all_keywords:
            if word == keyword or word not in vocab:
                continue
            w_col = tfidf_matrix[:, vocab[word]]
            sim   = cosine_similarity(kw_col.T.toarray(), w_col.T.toarray())[0][0]
            scores[word] = sim

        sorted_words = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return sorted_words[:n]

    except Exception:
        others = [k for k in all_keywords if k != keyword]
        return random.sample(others, min(n, len(others)))
