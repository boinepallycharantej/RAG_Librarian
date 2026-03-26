"""
DAY 1: Understanding Embeddings
================================

WHAT IS AN EMBEDDING?
Think of it like this:

    Every word or sentence can be turned into a LIST OF NUMBERS.
    These numbers capture the MEANING of the text.

    "dog" → [0.2, 0.8, 0.1, 0.9, ...]    (384 numbers!)
    "cat" → [0.3, 0.7, 0.1, 0.8, ...]    (very similar to dog!)
    "car" → [0.9, 0.1, 0.7, 0.2, ...]    (very different from dog!)

WHY DOES THIS MATTER?
    Because computers can't understand words — they only understand numbers.
    By converting text to numbers, we can ask the computer:
    "Hey, which of these 1000 paragraphs is most SIMILAR to this question?"

    The computer compares the numbers and finds the closest match.
    That's the magic behind RAG!

BEST PART: This runs 100% FREE on your computer. No API key needed!
We use a model called "all-MiniLM-L6-v2" — it's small, fast, and free.

Let's see this in action!
"""

from sentence_transformers import SentenceTransformer


# Load the embedding model — runs locally on YOUR computer, totally FREE!
# First time: downloads ~80MB model. After that, it's instant.
print("Loading embedding model (first time takes ~30 seconds to download)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")


def get_embedding(text):
    """
    Turn a piece of text into a list of numbers (embedding).
    This runs on YOUR computer — no internet needed after first download!
    """
    vector = model.encode(text)
    return vector


def cosine_similarity(vec1, vec2):
    """
    Measure how similar two embeddings are.
    Returns a number between -1 and 1:
        1  = exactly the same meaning
        0  = completely unrelated
       -1  = opposite meaning
    """
    # Dot product: multiply matching numbers and add them up
    dot_product = sum(float(a) * float(b) for a, b in zip(vec1, vec2))

    # Magnitude: how "long" each vector is
    magnitude1 = sum(float(a) ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(float(b) ** 2 for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# ============================================================
# LET'S EXPERIMENT!
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 1: Understanding Embeddings")
    print("(Running 100% locally — no API key needed!)")
    print("=" * 60)

    # --- Experiment 1: Similar words ---
    print("\n--- Experiment 1: Are 'dog' and 'cat' similar? ---\n")

    dog_vec = get_embedding("dog")
    cat_vec = get_embedding("cat")
    car_vec = get_embedding("car")

    print(f"  'dog' embedding: [{dog_vec[0]:.4f}, {dog_vec[1]:.4f}, {dog_vec[2]:.4f}, ...] ({len(dog_vec)} numbers!)")
    print(f"  'cat' embedding: [{cat_vec[0]:.4f}, {cat_vec[1]:.4f}, {cat_vec[2]:.4f}, ...]")
    print(f"  'car' embedding: [{car_vec[0]:.4f}, {car_vec[1]:.4f}, {car_vec[2]:.4f}, ...]")

    sim_dog_cat = cosine_similarity(dog_vec, cat_vec)
    sim_dog_car = cosine_similarity(dog_vec, car_vec)

    print(f"\n  Similarity between 'dog' and 'cat': {sim_dog_cat:.4f}")
    print(f"  Similarity between 'dog' and 'car': {sim_dog_car:.4f}")
    print(f"\n  --> 'dog' is MORE similar to 'cat' than to 'car'! Makes sense, right?")

    # --- Experiment 2: Sentences ---
    print("\n\n--- Experiment 2: Sentence similarity ---\n")

    sentences = [
        "I love playing football on weekends",
        "Soccer is my favorite weekend sport",
        "The stock market crashed yesterday",
    ]

    vecs = [get_embedding(s) for s in sentences]

    sim_1_2 = cosine_similarity(vecs[0], vecs[1])
    sim_1_3 = cosine_similarity(vecs[0], vecs[2])

    print(f"  Sentence 1: '{sentences[0]}'")
    print(f"  Sentence 2: '{sentences[1]}'")
    print(f"  Sentence 3: '{sentences[2]}'")
    print(f"\n  Similarity (1 vs 2): {sim_1_2:.4f}  <-- Both about weekend sports!")
    print(f"  Similarity (1 vs 3): {sim_1_3:.4f}  <-- Completely different topics!")

    # --- Experiment 3: Question matching (THIS IS RAG!) ---
    print("\n\n--- Experiment 3: Finding the best answer for a question ---")
    print("    (This is exactly how RAG works!)\n")

    question = "What is machine learning?"
    paragraphs = [
        "Machine learning is a type of AI where computers learn patterns from data without being explicitly programmed.",
        "The weather in San Francisco is usually foggy in the summer months.",
        "Python is a popular programming language used in many fields.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    ]

    question_vec = get_embedding(question)
    paragraph_vecs = [get_embedding(p) for p in paragraphs]

    print(f"  Question: '{question}'\n")
    print("  Searching through 4 paragraphs...\n")

    scores = []
    for i, (para, pvec) in enumerate(zip(paragraphs, paragraph_vecs)):
        score = cosine_similarity(question_vec, pvec)
        scores.append((score, i, para))
        print(f"  [{i+1}] Score: {score:.4f} | '{para[:60]}...'")

    # Sort by score (highest first)
    scores.sort(reverse=True)
    best = scores[0]

    print(f"\n  BEST MATCH (score {best[0]:.4f}):")
    print(f"  '{best[2]}'")
    print(f"\n  THIS IS EXACTLY HOW RAG WORKS!")
    print(f"  Step 1: Turn question into numbers        (you just saw this)")
    print(f"  Step 2: Turn all documents into numbers    (you just saw this)")
    print(f"  Step 3: Find the highest similarity score  (you just saw this)")
    print(f"  Step 4: Give that document to Claude and say 'answer using this'")

    print("\n" + "=" * 60)
    print("CONGRATS! You now understand the foundation of RAG!")
    print("Tomorrow we'll load real documents and chunk them up.")
    print("=" * 60)
