"""
Export utilities for Lecture Voice-to-Notes.

Formats supported
-----------------
PDF   — lecture summary + keywords (fpdf2)
TXT   — raw transcript
CSV   — flashcard deck (pandas)
JSON  — quiz questions
"""

import json
import io

import pandas as pd
from fpdf import FPDF


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

class _LectureNotePDF(FPDF):
    """Custom FPDF subclass with a branded header and footer."""

    def header(self):
        self.set_font("Helvetica", 'B', 10)
        self.set_text_color(108, 99, 255)   # #6C63FF brand accent
        self.cell(0, 8, "Lecture Voice-to-Notes | Generated Summary", ln=True)
        self.set_draw_color(108, 99, 255)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')


def export_notes_pdf(
    summary_sentences: list[str],
    keywords: list[tuple[str, float]],
) -> bytes:
    """
    Render summary sentences and keyword table to a PDF byte string.

    Returns raw bytes suitable for st.download_button(data=...).
    """
    pdf = _LectureNotePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", 'B', 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "Lecture Summary Notes", ln=True)
    pdf.ln(2)

    # ── Summary body ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(40, 40, 40)
    for i, sentence in enumerate(summary_sentences, 1):
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(8, 8, f"{i}.", ln=False)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, sentence)
        pdf.ln(1)

    pdf.ln(6)

    # ── Keywords section ─────────────────────────────────────────────────
    if keywords:
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 10, "Top Keywords", ln=True)
        pdf.set_draw_color(108, 99, 255)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Helvetica", size=11)
        pdf.set_text_color(40, 40, 40)
        for word, score in keywords:
            pdf.cell(0, 7, f"-  {word}  (TF-IDF score: {score})", ln=True)

    # fpdf2 outputs str; encode to bytes for Streamlit
    return pdf.output()


# ---------------------------------------------------------------------------
# Other export helpers
# ---------------------------------------------------------------------------

def export_flashcards_csv(flashcards: list[dict]) -> bytes:
    """Return UTF-8 encoded CSV bytes for a list of {question, answer} dicts."""
    df = pd.DataFrame(flashcards)
    return df.to_csv(index=False).encode('utf-8')


def export_quiz_json(quiz: list[dict]) -> bytes:
    """Return pretty-printed UTF-8 JSON bytes for the quiz question list."""
    return json.dumps(quiz, indent=2, ensure_ascii=False).encode('utf-8')


def export_transcript_txt(transcript: str) -> bytes:
    """Return UTF-8 encoded bytes of the raw transcript string."""
    return transcript.encode('utf-8')
