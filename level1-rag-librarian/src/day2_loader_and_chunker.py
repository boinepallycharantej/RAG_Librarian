"""
DAY 2: Document Loading & Chunking
====================================

THE PROBLEM:
    You have a 100-page PDF. You can't send ALL of it to the AI — it's too big
    and costs too much money. Plus, most of it isn't relevant to the question.

THE SOLUTION:
    1. LOAD the document (PDF, Word, or text file)
    2. CHOP it into small pieces called "chunks"
    3. Later (Day 3), we'll store these chunks in our vector database

WHAT IS CHUNKING?
    Imagine you have a textbook. Instead of searching the WHOLE book,
    you tear out individual paragraphs and put each one on an index card.

    Now when someone asks a question, you just flip through the cards
    and find the 3-4 most relevant ones. Much faster!

WHY OVERLAPPING CHUNKS?
    If we cut exactly at every 500 characters, we might cut a sentence in half:

    Chunk 1: "...machine learning is a method that"
    Chunk 2: "uses data to learn patterns..."

    The meaning is LOST! So we overlap chunks by ~50 characters:

    Chunk 1: "...machine learning is a method that uses data to"
    Chunk 2: "...a method that uses data to learn patterns..."

    Now the full sentence exists in at least one chunk. Smart, right?
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path):
    """
    Load a document (PDF or text) and return its content.

    Think of this as: Opening a book and reading all the pages.
    """
    # Figure out what type of file it is
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {extension}. Use .pdf, .txt, or .md")

    # Load returns a list of "Document" objects (one per page for PDFs)
    documents = loader.load()

    print(f"  Loaded '{os.path.basename(file_path)}'")
    print(f"  Number of pages/sections: {len(documents)}")
    print(f"  Total characters: {sum(len(doc.page_content) for doc in documents)}")

    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks.

    Think of this as: Tearing a book into index cards.

    Parameters:
        chunk_size:    How many characters per chunk (like card size)
        chunk_overlap: How many characters to repeat between chunks
                       (so we don't cut sentences in half)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These are the "cut points" — it tries to split at paragraphs first,
        # then sentences, then words. It's smart about where it cuts!
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"\n  Chunking complete!")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    return chunks


# ============================================================
# LET'S EXPERIMENT!
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 2: Document Loading & Chunking")
    print("=" * 60)

    # --- Step 1: Create a sample document to play with ---
    sample_text = """Machine Learning Fundamentals

Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed to perform a task, these systems use algorithms to identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning:

Supervised Learning: The algorithm learns from labeled training data. For example, given thousands of emails labeled as "spam" or "not spam", the algorithm learns to classify new emails. Common algorithms include Linear Regression, Decision Trees, and Support Vector Machines.

Unsupervised Learning: The algorithm finds hidden patterns in unlabeled data. For example, grouping customers into segments based on their purchasing behavior without knowing the groups in advance. K-Means Clustering and Principal Component Analysis (PCA) are popular unsupervised methods.

Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties. This is how game-playing AIs like AlphaGo were trained. The agent tries different strategies and learns which actions lead to the best outcomes.

Deep Learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks are particularly good at processing unstructured data like images, text, and audio. Convolutional Neural Networks (CNNs) excel at image recognition, while Recurrent Neural Networks (RNNs) and Transformers are used for natural language processing.

The machine learning workflow typically involves: collecting data, cleaning and preprocessing it, splitting into training and test sets, choosing a model, training it, evaluating performance, and deploying it to production. Feature engineering and hyperparameter tuning are crucial steps that can significantly impact model performance.
"""

    # Path to your PDF
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "Applied MultiVar Stat Analysis 6th Edition copy.pdf")

    # --- Step 2: Load the document ---
    print("\n--- Step 1: Loading the document ---\n")
    docs = load_document(sample_path)

    # --- Step 3: Show what a raw document looks like ---
    print("\n--- Step 2: What does a loaded document look like? ---\n")
    print(f"  First 200 characters:")
    print(f"  '{docs[0].page_content[:200]}...'")
    print(f"\n  Metadata (info about the document):")
    print(f"  {docs[0].metadata}")

    # --- Step 4: Chunk it! ---
    print("\n--- Step 3: Chunking the document ---\n")
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    # --- Step 5: Show the chunks ---
    print("\n--- Step 4: Let's look at each chunk ---\n")
    for i, chunk in enumerate(chunks):
        print(f"  CHUNK {i + 1} ({len(chunk.page_content)} chars):")
        print(f"  '{chunk.page_content[:100]}...'")
        print()

    # --- Step 6: Show overlap ---
    if len(chunks) >= 2:
        print("--- Step 5: See the overlap between chunks ---\n")
        end_of_chunk1 = chunks[0].page_content[-60:]
        start_of_chunk2 = chunks[1].page_content[:60]
        print(f"  End of Chunk 1:   '...{end_of_chunk1}'")
        print(f"  Start of Chunk 2: '{start_of_chunk2}...'")
        print(f"\n  See how they share some text? That's the overlap!")
        print(f"  This ensures no sentence gets cut in half.")

    print("\n" + "=" * 60)
    print("GREAT! Now you can load and chunk any document!")
    print("Tomorrow we'll store these chunks in ChromaDB (vector database).")
    print("=" * 60)

    # --- BONUS: Try with a PDF! ---
    print("\n--- BONUS: Want to try with your own PDF? ---\n")
    print("  Put any PDF in the 'data/' folder and change sample_path to:")
    print("  sample_path = 'data/your_file.pdf'")
    print("  Then run this script again!")
