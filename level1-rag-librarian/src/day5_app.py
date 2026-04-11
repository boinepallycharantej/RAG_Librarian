"""
DAY 5: Streamlit Web UI — Your RAG Librarian App
==================================================

THE PROBLEM:
    Running scripts in the terminal is great for learning,
    but nobody wants to use a command line to ask questions.

THE SOLUTION:
    Streamlit turns your Python script into a web app with
    just a few lines of code. No HTML, no CSS, no JavaScript needed.

WHAT YOU'LL BUILD:
    A web app where you can:
    1. Upload any PDF or text file
    2. Ask questions about it
    3. See Claude's answer AND the source chunks it used

    That's a real, working RAG application!

HOW TO RUN:
    streamlit run src/day5_app.py
    (from the level1-rag-librarian folder)
"""

import os
import tempfile
import streamlit as st
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Librarian",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# HELPER FUNCTIONS (same logic as Days 2-4, wrapped for Streamlit)
# ============================================================

def load_and_chunk(file_path, chunk_size=500, chunk_overlap=50):
    """Load a file and split into chunks."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


@st.cache_resource
def get_embedding_fn():
    """Load embedding model once and reuse (cached by Streamlit)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def build_vector_store(chunks):
    """Store chunks in an in-memory ChromaDB collection."""
    client = chromadb.EphemeralClient()  # in-memory, no disk writes needed
    embedding_fn = get_embedding_fn()

    collection = client.get_or_create_collection(
        name="uploaded_doc",
        embedding_function=embedding_fn,
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )

    return collection


def ask_claude(collection, question, n_chunks=3):
    """Retrieve relevant chunks and ask Claude."""
    results = collection.query(query_texts=[question], n_results=n_chunks)
    chunks = results["documents"][0]

    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text, chunks


# ============================================================
# UI
# ============================================================

st.title("📚 RAG Librarian")
st.caption("Upload a document, ask questions, get answers — powered by Claude.")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("1. Upload your document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or text file",
        type=["pdf", "txt", "md"],
    )

    st.divider()
    st.header("2. Settings")
    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 200, 50, 10)
    n_results = st.slider("Chunks to retrieve", 1, 5, 3)

# --- Main area ---
if uploaded_file is None:
    st.info("Upload a PDF or text file in the sidebar to get started.")

    st.subheader("How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Upload**\nDrop any PDF or text file")
    with col2:
        st.markdown("**2. Chunk**\nDocument split into pieces")
    with col3:
        st.markdown("**3. Search**\nRelevant pieces retrieved")
    with col4:
        st.markdown("**4. Answer**\nClaude answers from context")

else:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load, chunk, store
    with st.spinner("Processing document..."):
        try:
            chunks = load_and_chunk(tmp_path, chunk_size, chunk_overlap)
            collection = build_vector_store(chunks)
            st.sidebar.success(f"Ready! {len(chunks)} chunks indexed.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    # --- Q&A ---
    st.subheader(f"Ask a question about: *{uploaded_file.name}*")
    question = st.text_input("Your question", placeholder="What is this document about?")

    if question:
        with st.spinner("Thinking..."):
            try:
                answer, source_chunks = ask_claude(collection, question, n_results)
            except Exception as e:
                st.error(f"Error calling Claude: {e}")
                st.stop()

        st.markdown("### Answer")
        st.markdown(answer)

        with st.expander("View source chunks used"):
            for i, chunk in enumerate(source_chunks):
                st.markdown(f"**Chunk {i + 1}**")
                st.text(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                st.divider()
                