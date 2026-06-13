import re
import nltk

from algorithms.preprocessor import get_sentences

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Each entry: (regex_pattern, question_template)
# Patterns ordered from most specific to most general.
DEFINITION_PATTERNS: list[tuple[str, str]] = [
    (r'(.+?)\s+is\s+defined\s+as\s+(.+)',  "What is the definition of {subject}?"),
    (r'(.+?)\s+are\s+defined\s+as\s+(.+)', "How are {subject} defined?"),
    (r'(.+?)\s+refers?\s+to\s+(.+)',        "What does {subject} refer to?"),
    (r'(.+?)\s+means\s+(.+)',               "What does {subject} mean?"),
    (r'(.+?)\s+is\s+a\s+type\s+of\s+(.+)', "What type of thing is {subject}?"),
    (r'(.+?)\s+is\s+a\s+(.+)',              "What is {subject}?"),
    (r'(.+?)\s+are\s+(.+)',                 "What are {subject}?"),
    (r'(.+?)\s+is\s+(.+)',                  "What is {subject}?"),
]


def generate_flashcards(transcript: str) -> list[dict]:
    """
    Generate Q&A flashcards from definitional sentences in the transcript.

    Strategy
    --------
    1. Split transcript into sentences (NLTK sent_tokenize).
    2. For each sentence, test every regex pattern in priority order.
    3. On a match, extract (subject, definition) and format into a card dict.
    4. Skip duplicates (same subject, case-insensitive) and overly long subjects.

    Returns
    -------
    List of {"question": str, "answer": str} dicts.
    """
    sentences = get_sentences(transcript)
    flashcards: list[dict] = []
    seen: set[str] = set()

    for sentence in sentences:
        for pattern, q_template in DEFINITION_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                answer  = match.group(2).strip()

                # Filter out noise: very long subjects or already processed ones
                if len(subject) > 80 or not subject:
                    break
                if subject.lower() in seen:
                    break

                seen.add(subject.lower())
                question = q_template.format(subject=subject)
                flashcards.append({"question": question, "answer": answer})
                break   # one card per sentence; stop trying other patterns

    return flashcards
