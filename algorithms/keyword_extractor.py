import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(transcript: str, n: int = 12) -> list[tuple[str, float]]:
    """
    Extract the top-N keywords (and bigrams) from a transcript using TF-IDF.

    The whole transcript is treated as a single document.  Term scores are the
    mean TF-IDF weight across all rows (here just one row), so they reflect how
    distinctive each term is relative to the fitted vocabulary.

    Parameters
    ----------
    transcript : str
        Raw lecture text.
    n : int
        Number of top keywords to return.

    Returns
    -------
    List of (keyword, score) tuples, sorted by score descending.
    """
    if not transcript.strip():
        return []

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=500,
        ngram_range=(1, 2),   # unigrams + bigrams ("machine learning", etc.)
    )
    tfidf_matrix = vectorizer.fit_transform([transcript])
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    top_indices = np.argsort(scores)[-n:][::-1]
    return [(feature_names[i], round(float(scores[i]), 4)) for i in top_indices]
