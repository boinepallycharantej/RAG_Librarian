"""
DAY 4: Full RAG Pipeline — Ask Questions, Get Answers
=======================================================

THE PROBLEM:
    You have a document, and you want to ask questions about it.
    But sending the WHOLE document to Claude is expensive and slow.

THE SOLUTION — RAG (Retrieval Augmented Generation):
    1. RETRIEVE  — Find the most relevant chunks from ChromaDB (Day 3)
    2. AUGMENT   — Add those chunks as context to your question
    3. GENERATE  — Claude reads the context and answers your question

HOW IT ALL CONNECTS:
    Day 1: Embeddings  →  Turn text into numbers
    Day 2: Chunking    →  Break documents into pieces
    Day 3: ChromaDB    →  Store and search those pieces
    Day 4: Claude      →  Answer questions using the found pieces

ANALOGY:
    Imagine an open-book exam:
    - You have a 500-page textbook (your document)
    - A question comes in: "What is reinforcement learning?"
    - Instead of reading all 500 pages, you flip to the 3 most relevant pages
    - Then you answer using ONLY those pages

    That's exactly what RAG does — but in milliseconds.
"""

import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from day2_loader_and_chunker import load_document, chunk_documents
from day3_vector_store import create_vector_store, search

# Load API key from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def ask_question(collection, question, n_chunks=3):
    """
    The full RAG pipeline in one function.

    Step 1: Retrieve — find relevant chunks from ChromaDB
    Step 2: Augment  — build a prompt with those chunks as context
    Step 3: Generate — send to Claude and get an answer
    """

    # --- STEP 1: RETRIEVE ---
    # Search ChromaDB for the most relevant chunks
    results = search(collection, question, n_results=n_chunks)
    relevant_chunks = results["documents"][0]

    # --- STEP 2: AUGMENT ---
    # Build the context from retrieved chunks
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # --- STEP 3: GENERATE ---
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast and cheap — perfect for RAG
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    return answer, relevant_chunks


def display_rag_result(question, answer, chunks):
    """Pretty-print the question, retrieved chunks, and final answer."""
    print(f"\n{'=' * 60}")
    print(f"  QUESTION: {question}")
    print(f"{'=' * 60}")

    print(f"\n  [Retrieval] Found {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i + 1}: \"{chunk[:100]}...\"")

    print(f"\n  [Answer from Claude]:")
    print(f"  {answer}")
    print()


# ============================================================
# LET'S EXPERIMENT!
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 4: Full RAG Pipeline — Q&A with Claude")
    print("=" * 60)

    # --- Step 1: Load, chunk, and store document ---
    print("\n--- Step 1: Setting up the knowledge base ---\n")

    sample_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_ml_basics.txt"
    )
    docs = load_document(sample_path)
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    collection = create_vector_store(chunks)

    # --- Step 2: Ask questions! ---
    print("\n--- Step 2: Asking questions (full RAG pipeline) ---")

    questions = [
        "What is supervised learning and give an example?",
        "What are the steps in a machine learning workflow?",
        "How is deep learning different from regular machine learning?",
    ]

    for question in questions:
        answer, retrieved_chunks = ask_question(collection, question)
        display_rag_result(question, answer, retrieved_chunks)

    # --- Step 3: Show what happens with an out-of-context question ---
    print("--- Step 3: What happens when the answer isn't in the document? ---")
    out_of_scope = "What is the capital of France?"
    answer, retrieved_chunks = ask_question(collection, out_of_scope)
    display_rag_result(out_of_scope, answer, retrieved_chunks)

    print("=" * 60)
    print("You just built a complete RAG pipeline!")
    print("Next: Wrap this in a web UI with Streamlit (Day 5).")
    print("=" * 60)
