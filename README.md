<div align="center">

# 📄 DocMind
### Chat with your PDFs using AI — 100% Free & Open Source

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Upload any PDF → Ask questions → Get accurate answers with page references**

[🚀 Live Demo](#-live-demo) · [⚡ Quick Start](#-quick-start) · [🏗️ Architecture](#-architecture) · [📊 Evaluation](#-evaluation)

</div>

---

## 📌 What is DocMind?

DocMind is a **Retrieval-Augmented Generation (RAG)** chatbot built as a Final Year B.Tech Data Science project. It lets you upload any PDF document — research papers, textbooks, legal documents, manuals — and have a natural conversation with it.

Instead of reading the entire document yourself, just ask questions and DocMind finds the exact answer along with the page number it came from.

### Why RAG?

Normal chatbots (like ChatGPT) answer from memory — they can hallucinate and make up facts. RAG is different:

```
Your Question
     │
     ▼
Search your PDF for relevant sections   ← This is the "Retrieval" part
     │
     ▼
Send those sections + your question to the AI   ← This is the "Augmented" part
     │
     ▼
AI answers ONLY from your document   ← No hallucination, cites page numbers
```

This makes DocMind trustworthy for real documents — it will say "I don't have enough information" rather than making something up.

---

## ✨ Features

- 📄 **Multi-PDF support** — Upload and query multiple PDFs at once
- 🔍 **Source citations** — Every answer shows which page it came from
- 💬 **Conversational** — Ask follow-up questions naturally
- ⚡ **Fast** — Groq's LLaMA 3.1 runs at 750+ tokens/second
- 🔒 **No hallucination** — Answers strictly from your document
- 🆓 **Completely free** — Groq API free tier + Streamlit Cloud free hosting
- 📊 **Evaluated** — RAGAS metrics to measure answer quality

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              DocMind System              │
                        └─────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
  │  PDF     │───▶│  PyMuPDF     │───▶│  LangChain   │───▶│  ChromaDB   │
  │  Upload  │    │  (Extract    │    │  Text        │    │  (Vector    │
  │          │    │   Text)      │    │  Splitter    │    │   Store)    │
  └──────────┘    └──────────────┘    └──────────────┘    └─────────────┘
                                                                  │
                                                    Embed with    │
                                                    MiniLM-L6-v2  │
                                                                  ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
  │  Answer  │◀───│  Groq API    │◀───│  RAG Prompt  │◀───│  Similarity │
  │  + Page  │    │  LLaMA 3.1  │    │  Template    │    │  Search     │
  │  Numbers │    │  8B          │    │              │    │  (Top 4)    │
  └──────────┘    └──────────────┘    └──────────────┘    └─────────────┘
```

### How it works step by step

**Step 1 — Indexing (happens when you upload a PDF)**

1. PyMuPDF reads the PDF page by page and extracts all text
2. Text is split into 500-character overlapping chunks (so context isn't lost at boundaries)
3. Each chunk is converted into a 384-dimension vector using `all-MiniLM-L6-v2`
4. All vectors are stored in ChromaDB on disk — this is your searchable index

**Step 2 — Querying (happens when you ask a question)**

1. Your question is also converted into a vector using the same embedding model
2. ChromaDB finds the 4 most similar chunks from your PDF using cosine similarity
3. Those chunks + your question are sent to LLaMA 3.1 on Groq with a strict prompt
4. LLaMA answers using ONLY those chunks — no outside knowledge
5. The answer + source page numbers are shown in the UI

---

## 🛠️ Tech Stack

| Layer | Technology | Why we chose it |
|---|---|---|
| **LLM** | LLaMA 3.1 8B via Groq | Free, fast (750 tok/s), same quality as GPT-3.5 |
| **Embeddings** | all-MiniLM-L6-v2 | Free, lightweight, great semantic search |
| **Vector DB** | ChromaDB | Local, no setup needed, persists to disk |
| **PDF Parsing** | PyMuPDF (fitz) | Fast, accurate, handles multi-page PDFs |
| **RAG Framework** | LangChain | Industry standard, LCEL pipeline |
| **UI** | Streamlit | Python-native, easy to deploy |
| **Hosting** | Streamlit Cloud | Free, auto-deploys from GitHub |
| **Evaluation** | RAGAS | Measures faithfulness, relevancy, recall |

---

## 📁 Project Structure

```
docmind/
│
├── app.py                        # Main Streamlit app — UI and session management
│
├── rag/                          # Core RAG logic
│   ├── __init__.py
│   ├── loader.py                 # PDF reading + text chunking (PyMuPDF)
│   ├── embedder.py               # HuggingFace embeddings + ChromaDB operations
│   ├── retriever.py              # Similarity search — fetches top-k relevant chunks
│   └── chain.py                  # LangChain LCEL pipeline — connects retriever to LLM
│
├── evaluate.py                   # RAGAS evaluation — auto-generates Q&A and scores
├── requirements.txt              # All Python dependencies
├── .env.example                  # Local environment variable template
├── .gitignore                    # Prevents secrets and data from being committed
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Option A — Run Locally (Windows)

**Prerequisites:** Python 3.11+, Git

```bash
# 1. Clone the repo
git clone https://github.com/Codex610/docmind.git
cd docmind

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
copy .env.example .env
# Open .env in Notepad and add: GROQ_API_KEY=gsk_your_key_here

# 5. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

### Option B — Deploy on Streamlit Cloud (Free Public URL)

**Step 1 — Get a free Groq API key**

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google or GitHub (no credit card needed)
3. Click **API Keys** → **Create API Key**
4. Copy the key starting with `gsk_...`

**Step 2 — Push your code to GitHub**

```bash
git init
git add .
git commit -m "Initial commit — DocMind RAG chatbot"
git branch -M main
git remote add origin https://github.com/Codex610/docmind.git
git push -u origin main
```

**Step 3 — Deploy**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repository
5. Set **Main file path** to `app.py`
6. Click **Advanced settings** → **Secrets** → paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
7. Click **Deploy**

✅ You get a live public URL: `https://rag-docmind.streamlit.app`

---

## 📊 Evaluation

DocMind includes a built-in evaluation pipeline using [RAGAS](https://docs.ragas.io) — an industry-standard framework for measuring RAG quality.

```bash
# Run evaluation on any PDF (generates 10 test questions automatically)
python evaluate.py uploads/your_document.pdf 10
```

### Metrics explained

| Metric | What it measures | Good score |
|---|---|---|
| **Faithfulness** | Are answers grounded in the document? (no hallucination) | > 0.8 |
| **Answer Relevancy** | Does the answer actually address the question? | > 0.8 |
| **Context Recall** | Did the retriever find the right chunks? | > 0.7 |
| **Context Precision** | Were retrieved chunks relevant (no noise)? | > 0.7 |

### Sample evaluation output

```
=== RAGAS Scores ===
  faithfulness:       0.91   ✅ Good
  answer_relevancy:   0.87   ✅ Good
  context_recall:     0.83   ✅ Good
  context_precision:  0.79   ✅ Good

Report saved to eval_report.md
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |

**Local:** Add to `.env` file
**Streamlit Cloud:** Add to App Settings → Secrets

---

## ⚠️ Common Issues & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: fitz` | Change `pymupdf` to `PyMuPDF` in requirements.txt |
| `GROQ_API_KEY not found` | Add the key in Streamlit Cloud Secrets or your .env file |
| `No text extracted from PDF` | Your PDF might be scanned/image-based — try a text-based PDF |
| App slow on first load | Normal — embedding model downloads once (~90MB) then caches |
| `chromadb` version conflict | Pin to `chromadb==0.5.5` in requirements.txt |

---

## 🔮 Future Improvements

- [ ] Add support for scanned PDFs using OCR (Tesseract)
- [ ] Chat history export as PDF/text
- [ ] Support for `.docx` and `.txt` files
- [ ] Re-ranking retrieved chunks for better precision
- [ ] User authentication for private document storage

---

## 📚 References

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAGAS Evaluation Framework](https://docs.ragas.io)
- [Groq API Documentation](https://console.groq.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## 👨‍💻 Author

Built as a Final Year B.Tech Data Science Project.

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

<div align="center">
  <b>If this project helped you, give it a ⭐ on GitHub!</b>
</div>