# RAG Librarian

A hands-on, day-by-day project to understand and build a **Retrieval Augmented Generation (RAG)** system from scratch using Python, ChromaDB, and Claude.

---

## What is RAG?

Instead of sending an entire document to an AI (expensive and slow), RAG works in 3 steps:

1. **Retrieve** — find the most relevant chunks from your document
2. **Augment** — add those chunks as context to your question
3. **Generate** — the AI answers using only that context

Think of it like an open-book exam: you don't read the whole textbook, you flip to the relevant pages and answer from there.

---

## Project Structure

```
level1-rag-librarian/
├── src/
│   ├── day1_embeddings.py        # Text → numbers (embeddings)
│   ├── day2_loader_and_chunker.py # PDF/text loading & chunking
│   ├── day3_vector_store.py      # ChromaDB storage & semantic search
│   ├── day4_rag_qa.py            # Full RAG pipeline with Claude
│   └── day5_app.py               # Streamlit web app
├── data/
│   └── sample_ml_basics.txt      # Sample document for testing
├── .env                          # API keys (not committed)
└── requirements.txt
```

---

## Day-by-Day Breakdown

### Day 1 — Embeddings
Understand how text gets converted into numbers (vectors) that capture meaning.
- "dog" and "cat" have similar vectors; "dog" and "car" don't
- Uses `sentence-transformers` — runs 100% locally, no API key needed

```bash
python src/day1_embeddings.py
```

### Day 2 — Document Loading & Chunking
Load PDFs and text files, then split them into overlapping chunks.
- Overlapping chunks prevent sentences from being cut in half
- Supports `.pdf`, `.txt`, `.md`

```bash
python src/day2_loader_and_chunker.py
```

### Day 3 — Vector Store with ChromaDB
Store chunks as embeddings in ChromaDB and search by meaning.
- Semantic search finds relevant chunks even without exact keyword matches
- ChromaDB persists to disk so data survives between runs

```bash
python src/day3_vector_store.py
```

### Day 4 — Full RAG Pipeline
Connect ChromaDB retrieval to Claude for a complete Q&A system.
- Retrieve top N relevant chunks for a question
- Claude answers using only those chunks as context
- Out-of-scope questions get "I don't have enough information" — no hallucination

```bash
python src/day4_rag_qa.py
```

### Day 5 — Streamlit Web App
A browser-based UI to upload documents and ask questions interactively.

```bash
streamlit run src/day5_app.py --server.fileWatcherType none
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/boinepallycharantej/RAG_Librarian.git
cd RAG_Librarian/level1-rag-librarian
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in `level1-rag-librarian/`:
```
ANTHROPIC_API_KEY=your_key_here
```
Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).
> Days 1–3 don't need an API key. Only Days 4 and 5 require Claude.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Local embeddings (free, no API needed) |
| `ChromaDB` | Vector database for storing and searching chunks |
| `LangChain` | Document loading and text splitting |
| `Anthropic Claude` | Answering questions from retrieved context |
| `Streamlit` | Web UI |
