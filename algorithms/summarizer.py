import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from algorithms.preprocessor import get_sentences


def summarize(transcript: str, num_sentences: int = 5) -> tuple[list[str], float]:
    """
    Extractive summarization using TF-IDF sentence scoring.

    Algorithm
    ---------
    1. Tokenize the transcript into sentences.
    2. Vectorize every sentence with TF-IDF (sklearn, English stop-words removed).
    3. Score each sentence by the mean of its non-zero TF-IDF weights.
    4. Select the top-N sentences and return them in their original order so the
       summary reads coherently.

    Returns
    -------
    (summary_sentences, compression_ratio)
        compression_ratio = 1 − (selected / total).  0 means no compression.
    """
    sentences = get_sentences(transcript)

    # Edge-case: fewer sentences than requested → return all
    if len(sentences) <= num_sentences:
        return sentences, 0.0

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Row-wise mean of the TF-IDF matrix  (shape: n_sentences × vocab)
    scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()

    top_indices = np.argsort(scores)[-num_sentences:]
    top_indices = np.sort(top_indices)   # preserve original order

    summary = [sentences[i] for i in top_indices]
    compression = round(1.0 - len(summary) / len(sentences), 2)
    return summary, compression
