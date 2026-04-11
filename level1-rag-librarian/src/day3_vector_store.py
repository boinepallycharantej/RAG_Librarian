"""
DAY 3: Vector Store with ChromaDB
====================================

THE PROBLEM:
    You have chunks of text from Day 2. But how do you FIND the right chunk
    when someone asks a question? You can't just Ctrl+F — you need to search
    by MEANING, not exact words.

THE SOLUTION:
    A vector database! It stores each chunk as an embedding (Day 1) and lets
    you search by similarity. Think of it as a librarian who understands
    what you MEAN, not just what you SAY.

HOW IT WORKS:
    1. Take each chunk from Day 2
    2. Convert it to an embedding (a list of numbers)
    3. Store it in ChromaDB
    4. When someone asks a question, convert the question to an embedding
    5. Find the chunks whose embeddings are CLOSEST to the question's embedding

    "What is supervised learning?" → finds chunks about supervised learning, 
    even if those chunks never use the exact phrase!
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from day2_loader_and_chunker import load_document, chunk_documents


def create_vector_store(chunks, collection_name="my_documents"):
    """
    Store document chunks in ChromaDB.

    Think of this as: Filing index cards into a smart cabinet
    that remembers what each card is ABOUT.
    """
    # Initialize ChromaDB — it saves to disk so your data persists
    db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    client = chromadb.PersistentClient(path=db_path)

    # Use the same embedding model from Day 1 (runs locally, free!)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete old collection if it exists (fresh start each run)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create a collection — like a folder in the filing cabinet
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    # Add chunks in batches (ChromaDB has a limit per call)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=[chunk.page_content for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )

    print(f"  Stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'")
    return collection


def search(collection, query, n_results=3):
    """
    Search for chunks most similar to the query.

    Think of this as: Asking the librarian a question and getting
    the 3 most relevant index cards back.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    return results


def display_results(query, results):
    """Pretty-print search results."""
    print(f"\n  Query: \"{query}\"")
    print(f"  Top {len(results['documents'][0])} results:\n")

    for i, (doc, distance) in enumerate(
        zip(results["documents"][0], results["distances"][0])
    ):
        # ChromaDB returns distances — lower = more similar
        similarity = 1 - distance  # rough conversion to similarity
        print(f"  Result {i + 1} (similarity: {similarity:.4f}):")
        print(f"  \"{doc[:150]}...\"")
        print()


# ============================================================
# LET'S EXPERIMENT!
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 3: Vector Store with ChromaDB")
    print("=" * 60)

    # --- Step 1: Load and chunk the document (reusing Day 2!) ---
    print("\n--- Step 1: Loading and chunking document ---\n")
    sample_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_ml_basics.txt"
    )

    # Create sample file if it doesn't exist
    if not os.path.exists(sample_path):
        sample_text = """Machine Learning Fundamentals

Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed to perform a task, these systems use algorithms to identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning:

Supervised Learning: The algorithm learns from labeled training data. For example, given thousands of emails labeled as "spam" or "not spam", the algorithm learns to classify new emails. Common algorithms include Linear Regression, Decision Trees, and Support Vector Machines.

Unsupervised Learning: The algorithm finds hidden patterns in unlabeled data. For example, grouping customers into segments based on their purchasing behavior without knowing the groups in advance. K-Means Clustering and Principal Component Analysis (PCA) are popular unsupervised methods.

Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties. This is how game-playing AIs like AlphaGo were trained. The agent tries different strategies and learns which actions lead to the best outcomes.

Deep Learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks are particularly good at processing unstructured data like images, text, and audio. Convolutional Neural Networks (CNNs) excel at image recognition, while Recurrent Neural Networks (RNNs) and Transformers are used for natural language processing.

The machine learning workflow typically involves: collecting data, cleaning and preprocessing it, splitting into training and test sets, choosing a model, training it, evaluating performance, and deploying it to production. Feature engineering and hyperparameter tuning are crucial steps that can significantly impact model performance.
"""
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        with open(sample_path, "w") as f:
            f.write(sample_text)

    docs = load_document(sample_path)
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    # --- Step 2: Store in ChromaDB ---
    print("\n--- Step 2: Storing chunks in ChromaDB ---\n")
    collection = create_vector_store(chunks)

    # --- Step 3: Search! ---
    print("\n--- Step 3: Let's search by meaning! ---\n")

    queries = [
        "What is supervised learning?",
        "How do neural networks work?",
        "What are the steps to build a model?",
    ]

    for query in queries:
        results = search(collection, query, n_results=2)
        display_results(query, results)

    # --- Step 4: Show WHY this is powerful ---
    print("--- Step 4: Why is this better than keyword search? ---\n")
    tricky_query = "How do machines learn from examples?"
    print(f"  Notice: \"{tricky_query}\"")
    print(f"  This doesn't contain 'supervised' or 'training data',")
    print(f"  but semantic search STILL finds relevant chunks!\n")

    results = search(collection, tricky_query, n_results=2)
    display_results(tricky_query, results)

    print("=" * 60)
    print("Your chunks are now searchable by MEANING!")
    print("Tomorrow we'll connect this to Claude to build a Q&A system.")
    print("=" * 60)
