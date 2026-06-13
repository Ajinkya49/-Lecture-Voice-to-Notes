"""
Lecture Voice-to-Notes — Main Streamlit Application
====================================================
Fully offline · No external APIs · Pure ML pipeline

Pages
-----
1. Upload & Transcribe   — audio file → PocketSphinx offline STT
2. Smart Notes           — TF-IDF extractive summarisation + keyword chart
3. Flashcards            — regex + NLTK POS definition extraction
4. Quiz Mode             — MCQ with cosine-similarity distractors
5. Export                — PDF / TXT / CSV / JSON downloads
"""

import os
import tempfile

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Lecture Voice to Notes",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Algorithm & utility imports ────────────────────────────────────────────
from algorithms.speech_to_text      import transcribe_audio
from algorithms.summarizer          import summarize
from algorithms.keyword_extractor   import extract_keywords
from algorithms.flashcard_generator import generate_flashcards
from algorithms.quiz_generator      import generate_quiz
from utils.exporter                 import (
    export_notes_pdf,
    export_flashcards_csv,
    export_quiz_json,
    export_transcript_txt,
)
from utils.visualizer import plot_keyword_chart, plot_summary_stats


# ══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════

_DEFAULT_STATE: dict = {
    'transcript':    None,
    'summary':       None,
    'compression':   None,
    'keywords':      None,
    'flashcards':    None,
    'quiz':          None,
    'quiz_score':    0,
    'quiz_index':    0,
    'quiz_answers':  {},
}

for _key, _val in _DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ══════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 16px;">
            <span style="font-size:2.4rem;">🎙️</span><br>
            <strong style="font-size:1.1rem; color:#6C63FF;">
                Lecture Voice to Notes
            </strong><br>
            <small style="color:#888;">100 % offline · No API · Pure ML</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📤 Upload & Transcribe",
            "📝 Smart Notes",
            "🃏 Flashcards",
            "❓ Quiz Mode",
            "📥 Export",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick status indicators in sidebar
    st.markdown("**Pipeline status**")
    st.markdown(
        f"{'✅' if st.session_state.transcript else '⬜'} Transcript ready"
    )
    st.markdown(
        f"{'✅' if st.session_state.summary    else '⬜'} Notes generated"
    )
    st.markdown(
        f"{'✅' if st.session_state.flashcards else '⬜'} Flashcards built"
    )
    st.markdown(
        f"{'✅' if st.session_state.quiz       else '⬜'} Quiz ready"
    )

    st.divider()
    st.caption(
        "Algorithms: TF-IDF · Cosine Similarity · "
        "Extractive Summarisation · NLTK POS · Regex"
    )


# ══════════════════════════════════════════════════════════════════════════
# Helper: styled section header
# ══════════════════════════════════════════════════════════════════════════

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
            <h2 style="margin:0; color:#FAFAFA;">{icon} {title}</h2>
            {"<p style='color:#888;margin:0;'>" + subtitle + "</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Transcribe
# ══════════════════════════════════════════════════════════════════════════

if page == "📤 Upload & Transcribe":
    section_header(
        "📤", "Upload & Transcribe",
        "Convert lecture audio to text — 100 % offline via PocketSphinx"
    )

    tab_audio, tab_manual = st.tabs(["🎵 Upload Audio File", "✍️ Paste Text"])

    # ── Audio upload tab ──────────────────────────────────────────────────
    with tab_audio:
        st.info(
            "**Supported formats:** .wav · .mp3 · .m4a  \n"
            "Processing happens entirely on this machine — no data leaves your device.",
            icon="ℹ️",
        )

        audio_file = st.file_uploader(
            "Drop your lecture recording here",
            type=['wav', 'mp3', 'm4a'],
            help="Max 200 MB per the server config.",
        )

        if audio_file is not None:
            st.audio(audio_file)   # playback preview

        transcribe_btn = st.button(
            "🚀 Transcribe Audio",
            disabled=(audio_file is None),
            type="primary",
        )

        if transcribe_btn and audio_file is not None:
            suffix = '.' + audio_file.name.rsplit('.', 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            with st.spinner("Running PocketSphinx offline STT — this may take a moment…"):
                transcript, error = transcribe_audio(tmp_path)

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            if error:
                st.error(f"**Transcription failed:** {error}")
                st.info(
                    "Tip: PocketSphinx works best with clear English speech "
                    "recorded in a quiet environment.  "
                    "For noisy recordings, paste the text manually in the "
                    "**Paste Text** tab.",
                    icon="💡",
                )
            else:
                st.session_state.transcript = transcript
                # Reset downstream outputs when a new transcript is loaded
                for k in ('summary', 'compression', 'keywords',
                          'flashcards', 'quiz'):
                    st.session_state[k] = None
                st.success("✅ Transcription complete!")
                st.balloons()

    # ── Manual text tab ───────────────────────────────────────────────────
    with tab_manual:
        st.markdown(
            "Paste or type your lecture transcript below.  "
            "This bypasses the STT step and feeds the text directly "
            "into the ML pipeline."
        )
        manual_text = st.text_area(
            "Lecture transcript",
            placeholder="Paste lecture notes or transcript here…",
            height=280,
        )
        if st.button("✅ Use This Text", type="primary",
                     disabled=not manual_text.strip()):
            st.session_state.transcript = manual_text.strip()
            for k in ('summary', 'compression', 'keywords',
                      'flashcards', 'quiz'):
                st.session_state[k] = None
            st.success("Transcript loaded from manual input!")

    # ── Transcript preview ────────────────────────────────────────────────
    if st.session_state.transcript:
        st.divider()
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown("### 📄 Current Transcript")
            st.text_area(
                "Raw transcript",
                st.session_state.transcript,
                height=300,
                key="transcript_display",
            )
        with col_right:
            words = len(st.session_state.transcript.split())
            chars = len(st.session_state.transcript)
            sents = st.session_state.transcript.count('.') \
                  + st.session_state.transcript.count('!') \
                  + st.session_state.transcript.count('?')
            st.metric("Words",      words)
            st.metric("Characters", chars)
            st.metric("~Sentences", sents)
            st.markdown("")
            if st.button("🗑️ Clear transcript"):
                for k in _DEFAULT_STATE:
                    st.session_state[k] = _DEFAULT_STATE[k]
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Smart Notes
# ══════════════════════════════════════════════════════════════════════════

elif page == "📝 Smart Notes":
    section_header(
        "📝", "Smart Notes Generator",
        "TF-IDF extractive summarisation + keyword analysis"
    )

    if not st.session_state.transcript:
        st.warning("⚠️ No transcript found. Go to **Upload & Transcribe** first.")
        st.stop()

    col_ctrl, _ = st.columns([2, 3])
    with col_ctrl:
        num_sentences = st.slider(
            "Summary length (sentences)", min_value=3, max_value=15, value=5,
            help="How many key sentences to extract from the transcript."
        )

    if st.button("⚡ Generate Notes", type="primary"):
        with st.spinner("Running TF-IDF summarisation…"):
            summary, compression = summarize(
                st.session_state.transcript, num_sentences
            )
        with st.spinner("Extracting keywords…"):
            keywords = extract_keywords(st.session_state.transcript)

        st.session_state.summary     = summary
        st.session_state.compression = compression
        st.session_state.keywords    = keywords
        st.success(
            f"Notes generated — {len(summary)} sentences, "
            f"{round(compression * 100)}% compression."
        )

    if st.session_state.summary:
        st.divider()
        col_notes, col_kw = st.columns([3, 2])

        # ── Summary notes ─────────────────────────────────────────────────
        with col_notes:
            st.markdown("### 🗒️ Summary Notes")
            for i, sentence in enumerate(st.session_state.summary, 1):
                st.markdown(
                    f"""
                    <div style="
                        background:#1A1C24;
                        border-left: 3px solid #6C63FF;
                        padding: 10px 14px;
                        border-radius: 4px;
                        margin-bottom: 10px;
                        color: #FAFAFA;
                        font-size: 0.95rem;
                        line-height: 1.55;
                    ">
                        <strong style="color:#6C63FF;">{i}.</strong> {sentence}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Stats chart
            total_sents = len(st.session_state.transcript.split('.'))
            word_count  = len(st.session_state.transcript.split())
            stats_buf = plot_summary_stats(
                total_sents,
                len(st.session_state.summary),
                word_count,
            )
            st.image(stats_buf, use_column_width=True)

        # ── Keywords panel ────────────────────────────────────────────────
        with col_kw:
            st.markdown("### 🔑 Top Keywords")
            chart_buf = plot_keyword_chart(st.session_state.keywords)
            st.image(chart_buf, use_column_width=True)

            st.markdown("**Ranked keyword list**")
            for rank, (word, score) in enumerate(
                st.session_state.keywords[:12], 1
            ):
                st.markdown(
                    f"<span style='color:#6C63FF;font-weight:600;'>"
                    f"#{rank}</span>  `{word}` — {score}",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Flashcards
# ══════════════════════════════════════════════════════════════════════════

elif page == "🃏 Flashcards":
    section_header(
        "🃏", "Flashcard Generator",
        "Regex pattern matching + NLTK POS tagging extracts definitions automatically"
    )

    if not st.session_state.transcript:
        st.warning("⚠️ No transcript found. Go to **Upload & Transcribe** first.")
        st.stop()

    if st.button("🃏 Generate Flashcards", type="primary"):
        with st.spinner("Scanning for definition patterns…"):
            st.session_state.flashcards = generate_flashcards(
                st.session_state.transcript
            )

    if st.session_state.flashcards is not None:
        cards = st.session_state.flashcards

        if not cards:
            st.info(
                "No definition-style sentences found in this transcript.  \n"
                "The extractor looks for patterns like:  \n"
                "- **X is defined as Y**  \n"
                "- **X refers to Y**  \n"
                "- **X means Y**  \n"
                "- **X is a Y**  \n"
                "\nTry a transcript with more formal, concept-defining language.",
                icon="💡",
            )
        else:
            st.success(f"✅ Generated **{len(cards)}** flashcards!")
            st.divider()

            # Card grid — two columns
            col_a, col_b = st.columns(2)
            for i, card in enumerate(cards):
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    with st.expander(
                        f"Card {i + 1} — {card['question'][:55]}…"
                        if len(card['question']) > 55
                        else f"Card {i + 1} — {card['question']}"
                    ):
                        st.markdown(
                            f"<div style='color:#6C63FF;font-weight:600;"
                            f"margin-bottom:6px;'>❓ Question</div>"
                            f"<div style='margin-bottom:12px;'>"
                            f"{card['question']}</div>"
                            f"<div style='color:#6C63FF;font-weight:600;"
                            f"margin-bottom:6px;'>✅ Answer</div>"
                            f"<div>{card['answer']}</div>",
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Quiz Mode
# ══════════════════════════════════════════════════════════════════════════

elif page == "❓ Quiz Mode":
    section_header(
        "❓", "Interactive Quiz",
        "Fill-in-the-blank MCQs with cosine-similarity-based distractors"
    )

    if not st.session_state.transcript:
        st.warning("⚠️ No transcript found. Go to **Upload & Transcribe** first.")
        st.stop()

    col_gen, col_n = st.columns([2, 1])
    with col_n:
        num_q = st.number_input(
            "Questions to generate", min_value=3, max_value=20, value=8, step=1
        )
    with col_gen:
        if st.button("🎲 Generate Quiz", type="primary"):
            with st.spinner("Building MCQs with cosine similarity distractors…"):
                quiz = generate_quiz(st.session_state.transcript, int(num_q))
            if not quiz:
                st.error(
                    "Could not generate enough questions — the transcript may be "
                    "too short or lack sufficient vocabulary variety."
                )
            else:
                st.session_state.quiz         = quiz
                st.session_state.quiz_index   = 0
                st.session_state.quiz_score   = 0
                st.session_state.quiz_answers = {}
                st.success(f"✅ {len(quiz)} questions ready!")

    if st.session_state.quiz:
        quiz  = st.session_state.quiz
        idx   = st.session_state.quiz_index
        total = len(quiz)

        st.divider()

        # ── Quiz in progress ──────────────────────────────────────────────
        if idx < total:
            # Progress bar
            st.progress((idx) / total, text=f"Question {idx + 1} of {total}")
            st.markdown("")

            q = quiz[idx]

            # Question card
            st.markdown(
                f"""
                <div style="
                    background: #1A1C24;
                    border: 1px solid #6C63FF44;
                    border-radius: 8px;
                    padding: 20px 24px;
                    margin-bottom: 20px;
                ">
                    <div style="color:#888;font-size:0.8rem;margin-bottom:8px;">
                        FILL IN THE BLANK
                    </div>
                    <div style="font-size:1.1rem;color:#FAFAFA;line-height:1.7;">
                        {q['question']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            selected = st.radio(
                "Select the correct word:",
                q['options'],
                key=f"quiz_radio_{idx}",
                index=None,
            )

            col_submit, col_next = st.columns([1, 1])

            with col_submit:
                submit_disabled = (selected is None) or (idx in st.session_state.quiz_answers)
                if st.button("✔ Submit Answer", disabled=submit_disabled):
                    correct = (selected == q['answer'])
                    st.session_state.quiz_answers[idx] = {
                        "selected": selected,
                        "correct":  q['answer'],
                        "is_right": correct,
                    }
                    if correct:
                        st.session_state.quiz_score += 1
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong — the answer is **{q['answer']}**")

            with col_next:
                if st.button("Next →", type="primary"):
                    st.session_state.quiz_index += 1
                    st.rerun()

            # Show feedback if already answered
            if idx in st.session_state.quiz_answers:
                ans = st.session_state.quiz_answers[idx]
                if ans['is_right']:
                    st.success("✅ Correct!")
                else:
                    st.error(
                        f"❌ You answered **{ans['selected']}** — "
                        f"correct answer: **{ans['correct']}**"
                    )

        # ── Quiz complete ─────────────────────────────────────────────────
        else:
            score = st.session_state.quiz_score
            pct   = round(score / total * 100) if total else 0

            st.balloons()

            grade_msg = (
                "🏆 Excellent!" if pct >= 80 else
                "👍 Good effort!" if pct >= 50 else
                "📖 Keep studying!"
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    background:#1A1C24;
                    border-radius:12px;
                    padding:36px;
                    border: 1px solid #6C63FF66;
                    margin-bottom:20px;
                ">
                    <div style="font-size:3rem;">{grade_msg}</div>
                    <div style="font-size:2rem;color:#6C63FF;font-weight:700;margin-top:12px;">
                        {score} / {total}
                    </div>
                    <div style="color:#AAAACC;margin-top:4px;">{pct}% correct</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(pct / 100)

            # Answer review
            with st.expander("📋 Review all answers"):
                for i in range(total):
                    ans = st.session_state.quiz_answers.get(i)
                    if ans:
                        icon = "✅" if ans['is_right'] else "❌"
                        st.markdown(
                            f"{icon} **Q{i + 1}** — "
                            f"You chose: `{ans['selected']}` · "
                            f"Correct: `{ans['correct']}`"
                        )
                    else:
                        st.markdown(f"⬜ **Q{i + 1}** — not answered")

            if st.button("🔄 Retake Quiz", type="primary"):
                st.session_state.quiz_index   = 0
                st.session_state.quiz_score   = 0
                st.session_state.quiz_answers = {}
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Export
# ══════════════════════════════════════════════════════════════════════════

elif page == "📥 Export":
    section_header(
        "📥", "Export Study Material",
        "Download your notes, flashcards, and quiz in multiple formats"
    )

    if not st.session_state.transcript:
        st.warning("⚠️ No transcript found. Go to **Upload & Transcribe** first.")
        st.stop()

    st.markdown(
        "Everything is generated on-device.  "
        "Files are created fresh on each click — no server storage."
    )
    st.divider()

    col_left, col_right = st.columns(2)

    # ── Left column ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### 📄 Notes (PDF)")
        if st.session_state.summary:
            pdf_bytes = export_notes_pdf(
                st.session_state.summary,
                st.session_state.keywords or [],
            )
            st.download_button(
                label="⬇️ Download Notes as PDF",
                data=pdf_bytes,
                file_name="lecture_notes.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Generate notes on the **Smart Notes** page first.", icon="ℹ️")

        st.markdown("#### 📝 Transcript (TXT)")
        txt_bytes = export_transcript_txt(st.session_state.transcript)
        st.download_button(
            label="⬇️ Download Transcript as TXT",
            data=txt_bytes,
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Right column ──────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### 🃏 Flashcards (CSV)")
        if st.session_state.flashcards:
            csv_bytes = export_flashcards_csv(st.session_state.flashcards)
            st.download_button(
                label="⬇️ Download Flashcards as CSV",
                data=csv_bytes,
                file_name="flashcards.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Generate flashcards on the **Flashcards** page first.", icon="ℹ️")

        st.markdown("#### ❓ Quiz (JSON)")
        if st.session_state.quiz:
            json_bytes = export_quiz_json(st.session_state.quiz)
            st.download_button(
                label="⬇️ Download Quiz as JSON",
                data=json_bytes,
                file_name="quiz.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Generate a quiz on the **Quiz Mode** page first.", icon="ℹ️")

    # ── Format guide ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ Format guide"):
        st.markdown(
            """
| Format | Contents | Best for |
|--------|----------|----------|
| **PDF** | Summary sentences + keyword table | Printing, sharing |
| **TXT** | Raw transcript text | Further editing |
| **CSV** | question, answer columns | Anki, Quizlet import |
| **JSON** | Full MCQ objects with options + answer | LMS / custom apps |
            """
        )
