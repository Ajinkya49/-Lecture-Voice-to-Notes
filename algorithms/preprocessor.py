import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def clean_text(text: str) -> str:
    """Remove redundant whitespace and non-standard punctuation."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    return text.strip()


def get_sentences(text: str) -> list[str]:
    """Tokenize text into sentences after cleaning."""
    return sent_tokenize(clean_text(text))


def get_words(text: str) -> list[str]:
    """
    Return cleaned, lemmatized, non-stopword alpha tokens longer than 3 chars.
    Used downstream by keyword extractor and summarizer helpers.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w.isalpha() and w not in stop_words and len(w) > 3
    ]
