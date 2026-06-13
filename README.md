<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=🎙️+Lecture+Voice-to-Notes;100%25+Offline+%C2%B7+Pure+ML;No+API+Keys+Required" alt="Typing SVG" />

<br/>

<p align="center">
  <strong>Convert lecture audio into smart study material — instantly, privately, offline.</strong>
</p>

<p align="center">
  <a href="https://your-app.streamlit.app">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  &nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  &nbsp;
  <img src="https://img.shields.io/badge/NLTK-NLP-green?style=flat" alt="NLTK"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=flat" alt="License"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Deployment-Streamlit Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Deployment"/>
</p>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

</div>

<br/>

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [ML Algorithms](#-ml-algorithms)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How to Use](#-how-to-use)
- [Deploy to Streamlit Cloud](#-deploy-to-streamlit-cloud)
- [Sample Test Text](#-sample-test-text)
- [Future Scope](#-future-scope)
- [Author](#-author)
- [License](#-license)

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🧭 Overview

**Lecture Voice-to-Notes** is a fully offline, AI-powered web application that transforms raw lecture audio or text transcripts into structured, exam-ready study material — with zero internet dependency and zero API costs.

Built entirely on classical ML algorithms (TF-IDF, Cosine Similarity, NLTK POS Tagging), the app delivers:

> 🎧 Audio → 📝 Transcript → 🗒️ Summary → 🔑 Keywords → 🃏 Flashcards → ❓ Quiz → 📥 Export

Everything runs **locally on your machine or on Streamlit Cloud's free tier** — your lecture content never leaves your device.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🌐 Live Demo

<div align="center">

| Platform | Link |
|----------|------|
| 🚀 Streamlit Cloud | https://ajinkya-kamble-lecture-voice-to-notes-app.streamlit.app/ |
| 💻 GitHub Repo | [https://github.com/Ajinkya49/-Lecture-Voice-to-Notes](https://github.com/Ajinkya49/-Lecture-Voice-to-Notes) |

> 💡 Replace the Streamlit link above with your actual deployed URL.

</div>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## ✨ Key Features

<table>
  <tr>
    <td>🎙️ <strong>Offline Speech-to-Text</strong></td>
    <td>Transcribes <code>.wav</code>, <code>.mp3</code>, <code>.m4a</code> using PocketSphinx — no internet required</td>
  </tr>
  <tr>
    <td>📝 <strong>Smart Notes</strong></td>
    <td>TF-IDF extractive summarisation — picks the most information-dense sentences</td>
  </tr>
  <tr>
    <td>🔑 <strong>Keyword Extraction</strong></td>
    <td>Ranks top unigrams & bigrams by TF-IDF score with a visual bar chart</td>
  </tr>
  <tr>
    <td>🃏 <strong>Flashcard Generator</strong></td>
    <td>Auto-creates Q&A pairs from definition sentences using regex + NLTK POS tagging</td>
  </tr>
  <tr>
    <td>❓ <strong>Interactive Quiz</strong></td>
    <td>Fill-in-the-blank MCQs with cosine-similarity-based wrong answers (plausible distractors)</td>
  </tr>
  <tr>
    <td>📊 <strong>Visual Analytics</strong></td>
    <td>Keyword frequency bar chart + compression ratio donut chart (matplotlib)</td>
  </tr>
  <tr>
    <td>📥 <strong>Multi-Format Export</strong></td>
    <td>Download as PDF notes, TXT transcript, CSV flashcards (Anki-ready), JSON quiz</td>
  </tr>
  <tr>
    <td>🔒 <strong>100% Private</strong></td>
    <td>No data sent to any server — all processing happens on-device</td>
  </tr>
</table>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🧠 ML Algorithms

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE OVERVIEW                            │
├──────┬────────────────────────────┬──────────────┬─────────────────┤
│  #   │  Algorithm                 │  Library     │  Powers         │
├──────┼────────────────────────────┼──────────────┼─────────────────┤
│  1   │  TF-IDF Vectorization      │ scikit-learn │ Summarisation   │
│      │                            │              │ Keyword Ranking │
├──────┼────────────────────────────┼──────────────┼─────────────────┤
│  2   │  Extractive Summarisation  │ scikit-learn │ Smart Notes     │
│      │  (mean TF-IDF sentence     │ + numpy      │ Generation      │
│      │   scoring)                 │              │                 │
├──────┼────────────────────────────┼──────────────┼─────────────────┤
│  3   │  Cosine Similarity         │ sklearn      │ Quiz Distractor │
│      │                            │ .pairwise    │ Generation      │
├──────┼────────────────────────────┼──────────────┼─────────────────┤
│  4   │  POS Tagging               │ NLTK         │ Flashcard       │
│      │  (Part-of-Speech)          │              │ Subject Extract │
├──────┼────────────────────────────┼──────────────┼─────────────────┤
│  5   │  Regex Pattern Matching    │ Python re    │ Definition      │
│      │  ("X is Y", "X means Y")   │              │ Extraction      │
└──────┴────────────────────────────┴──────────────┴─────────────────┘
```

### How Each Algorithm Works

<details>
<summary><strong>Algorithm 1 — TF-IDF Vectorization</strong></summary>
<br/>

**Term Frequency–Inverse Document Frequency** measures how important a word is to a document relative to a corpus.

- **TF**: how often a term appears in the sentence
- **IDF**: penalises terms that appear everywhere (stop words)
- **Result**: rare but meaningful terms score highest

```python
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(sentences)
```
</details>

<details>
<summary><strong>Algorithm 2 — Extractive Summarisation</strong></summary>
<br/>

Each sentence is scored by the **mean of its TF-IDF weights**. Top-N sentences are returned in their original order so the summary reads naturally.

```
Score(sentence) = mean(TF-IDF weights of all terms in sentence)
Summary = Top-N sentences sorted by original position
```
</details>

<details>
<summary><strong>Algorithm 3 — Cosine Similarity for Quiz Distractors</strong></summary>
<br/>

For each quiz keyword, the 3 most **semantically similar** (but different) keywords become wrong answer options. Similar terms = plausible distractors.

```
similarity(A, B) = (A · B) / (|A| × |B|)
Distractors = top-3 keywords by cosine similarity to the answer
```
</details>

<details>
<summary><strong>Algorithm 4 & 5 — POS Tagging + Regex for Flashcards</strong></summary>
<br/>

Sentences matching definition patterns are extracted and converted to Q&A pairs:

| Pattern | Question Template |
|---------|------------------|
| `X is defined as Y` | What is the definition of X? |
| `X refers to Y` | What does X refer to? |
| `X means Y` | What does X mean? |
| `X is a Y` | What is X? |
</details>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | 5-page dark-themed web app |
| **Speech-to-Text** | PocketSphinx + SpeechRecognition | 100% offline transcription |
| **NLP** | ![NLTK](https://img.shields.io/badge/NLTK-green?style=flat) | Tokenization, POS, Lemmatization |
| **ML Core** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | TF-IDF, Cosine Similarity |
| **Numerics** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Matrix operations |
| **Audio** | pydub + ffmpeg | MP3/M4A → WAV conversion |
| **Visualisation** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-blue?style=flat) | Keyword & stats charts |
| **PDF Export** | fpdf2 | Branded PDF notes |
| **Data Export** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | CSV flashcard export |
| **Deployment** | Streamlit Cloud | Free, auto-redeploy on push |

</div>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 📁 Project Structure

```
📦 lecture-voice-to-notes/
│
├── 📄 app.py                        ← Main Streamlit entry point (5 pages)
├── 📄 requirements.txt              ← Python dependencies
├── 📄 packages.txt                  ← Linux system packages (Streamlit Cloud)
│
├── ⚙️  .streamlit/
│   └── config.toml                  ← Dark theme (#6C63FF) + layout config
│
├── 🧠 algorithms/
│   ├── __init__.py
│   ├── speech_to_text.py            ← Offline STT (PocketSphinx)
│   ├── preprocessor.py              ← NLTK text cleaning & lemmatization
│   ├── summarizer.py                ← TF-IDF extractive summarisation
│   ├── keyword_extractor.py         ← TF-IDF unigram + bigram ranking
│   ├── flashcard_generator.py       ← Regex + POS definition extraction
│   └── quiz_generator.py            ← MCQ + cosine similarity distractors
│
└── 🔧 utils/
    ├── __init__.py
    ├── exporter.py                  ← PDF / TXT / CSV / JSON export
    └── visualizer.py                ← matplotlib keyword & stats charts
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- ffmpeg (for audio conversion)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Ajinkya49/-Lecture-Voice-to-Notes.git
cd -Lecture-Voice-to-Notes
```

**2. Create a virtual environment** *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Install system dependencies**

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg swig libpocketsphinx-dev

# macOS (Homebrew)
brew install ffmpeg swig

# Windows
# Download ffmpeg from https://ffmpeg.org/download.html and add to PATH
```

**5. Run the application**
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501** 🎉

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 📱 How to Use

```
┌─────────────────────────────────────────────────┐
│                 APP WORKFLOW                     │
│                                                 │
│  1. Upload Audio ──► or ──► Paste Text          │
│           │                      │              │
│           └──────────┬───────────┘              │
│                      ▼                          │
│              Transcript Ready ✅                │
│                      │                          │
│          ┌───────────┼───────────┐              │
│          ▼           ▼           ▼              │
│     Smart Notes  Flashcards  Quiz Mode          │
│          │           │           │              │
│          └───────────┼───────────┘              │
│                      ▼                          │
│                   Export 📥                     │
│           PDF · TXT · CSV · JSON                │
└─────────────────────────────────────────────────┘
```

### Page-by-Page Guide

| Page | Steps |
|------|-------|
| 📤 **Upload & Transcribe** | Upload audio file OR paste transcript → Click transcribe/use |
| 📝 **Smart Notes** | Adjust summary length slider (3–15 sentences) → Click ⚡ Generate Notes |
| 🃏 **Flashcards** | Click 🃏 Generate Flashcards → Expand cards to review Q&A |
| ❓ **Quiz Mode** | Click 🎲 Generate Quiz → Answer MCQs → See final score |
| 📥 **Export** | Download any combination of PDF / TXT / CSV / JSON |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## ☁️ Deploy to Streamlit Cloud

Deploy your own instance in **under 2 minutes** — completely free.

```bash
# Step 1: Push to GitHub
git add .
git commit -m "Initial commit"
git push origin main
```

```
Step 2: Go to https://share.streamlit.io
Step 3: Sign in with GitHub
Step 4: Click "New App" → Select your repo
Step 5: Main file path → app.py
Step 6: Click Deploy ✅
```

Your app will be live at:
```
https://YOUR_USERNAME-lecture-voice-to-notes-app-XXXX.streamlit.app
```

> Auto-redeploys on every `git push` to `main`.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🧪 Sample Test Text

Copy and paste this into the **Paste Text** tab to instantly test all features:

```
Machine learning is a branch of artificial intelligence that enables systems to learn
from data without being explicitly programmed. Supervised learning refers to a paradigm
where models are trained on labelled examples. A neural network is a computational model
inspired by the human brain. Backpropagation is defined as the algorithm used to compute
gradients in deep networks. Gradient descent means an optimisation method that minimises
a loss function by iteratively adjusting model parameters. Overfitting is a phenomenon
where a model performs well on training data but poorly on unseen examples.
Regularisation refers to techniques that prevent overfitting by penalising model
complexity. Cross-validation is a technique used to estimate generalisation performance.
Convolutional neural networks are deep learning architectures designed for image
processing. Transformers are attention-based models that have revolutionised natural
language processing.
```

**Expected output:**
- ✅ ~4 summary sentences
- ✅ 8–12 keywords (e.g. `machine learning`, `neural network`, `gradient descent`)
- ✅ 8+ flashcards (definitions auto-extracted)
- ✅ 6–8 MCQ questions

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🔮 Future Scope

- [ ] 🎤 **Whisper STT** — local Whisper model for far more accurate transcription
- [ ] 🤖 **Abstractive Summarisation** — local LLM (Ollama + Mistral 7B) for human-readable notes
- [ ] 🌐 **Multi-language Support** — Hindi, Marathi, and other regional languages via NLTK
- [ ] 🔊 **Speaker Diarisation** — identify and label different speakers (student vs. professor)
- [ ] 🃏 **Anki Export** — direct `.apkg` deck export for spaced repetition
- [ ] ☁️ **User Accounts** — Supabase auth + cloud sync to save notes across sessions
- [ ] 📱 **Mobile App** — React Native frontend with Python backend
- [ ] 📈 **Progress Tracker** — quiz history and performance analytics dashboard

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 👤 Author

<div align="center">

| | |
|---|---|
| **Name** | Ajinkya |
| **GitHub** | [@Ajinkya49](https://github.com/Ajinkya49) |
| **Project** | Capstone Project — CSE |

</div>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io) — for the incredible free deployment platform
- [NLTK](https://www.nltk.org) — for the NLP toolkit
- [scikit-learn](https://scikit-learn.org) — for TF-IDF and Cosine Similarity
- [PocketSphinx](https://github.com/cmusphinx/pocketsphinx) — for offline speech recognition
- [fpdf2](https://py-fpdf2.readthedocs.io) — for PDF generation

<br/>

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line"/>

### ⭐ If this project helped you, please give it a star!

[![GitHub stars](https://img.shields.io/github/stars/Ajinkya49/-Lecture-Voice-to-Notes?style=social)](https://github.com/Ajinkya49/-Lecture-Voice-to-Notes/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Ajinkya49/-Lecture-Voice-to-Notes?style=social)](https://github.com/Ajinkya49/-Lecture-Voice-to-Notes/network/members)

<br/>

*Made with ❤️ by Ajinkya*

</div>
