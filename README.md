# DocMind — Chat with your PDFs

A local + cloud RAG chatbot. Upload PDFs and ask questions about them.
Powered by LLaMA 3.1 via Groq, ChromaDB, LangChain, and Streamlit.

## Live Demo
Deploy your own → see deployment guide below.

## Tech Stack
| Layer | Tool |
|---|---|
| LLM | LLaMA 3.1 8B via Groq API (free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (local) |
| PDF Parsing | PyMuPDF |
| RAG Framework | LangChain |
| UI | Streamlit |

---

## Deploy on Streamlit Cloud (Free — Recommended)

### Step 1 — Get a free Groq API key
Sign up at https://console.groq.com → Create API Key

### Step 2 — Push to GitHub
```bash
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/Codex610/docmind.git
git push -u origin main
```

### Step 3 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → Main file: `app.py`
5. Click **Advanced settings** → **Secrets** → add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
6. Click **Deploy**

Done! You get a public URL like `https://yourname-docmind.streamlit.app`

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/Codex610/docmind.git
cd docmind

# 2. Install dependencies
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 3. Set your API key
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Run
streamlit run app.py
```

---

## Project Structure
```
docmind/
├── app.py                        # Streamlit UI
├── evaluate.py                   # RAGAS evaluation
├── requirements.txt
├── .env.example                  # Local env template
├── .gitignore
└── rag/
    ├── loader.py                 # PDF parsing + chunking
    ├── embedder.py               # ChromaDB + HuggingFace embeddings
    ├── retriever.py              # Similarity search
    └── chain.py                  # LangChain RAG pipeline (Groq LLM)
```

---

## Evaluate

```bash
python evaluate.py uploads/your_doc.pdf 10
# Generates 10 Q&A pairs and scores with RAGAS metrics
# Output saved to eval_report.md
```

---

## License
MIT