# ============================================================
# MODULE 6 — retriever.py
# Owner: Person 2 (extension of qa_engine role)
# Responsibility: Find the most relevant chunk for a question
#                 using TF-IDF similarity (no GPU needed)
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_best_chunk(question: str, chunks: list[str], top_k: int = 1) -> list[str]:
    """
    Uses TF-IDF + cosine similarity to find the most relevant
    chunk(s) from a document for a given question.

    How it works:
        1. Vectorize all chunks + the question using TF-IDF
        2. Compute cosine similarity between question and each chunk
        3. Return the top_k most similar chunks

    Args:
        question : the user's question string
        chunks   : list of text chunks from the document
        top_k    : number of top chunks to return (default 1)

    Returns:
        A list of the top_k most relevant chunk strings.
    """
    if not chunks:
        return []

    # Combine question + all chunks for vectorization
    corpus = [question] + chunks

    # Fit TF-IDF vectorizer on corpus
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Question vector is index 0, chunk vectors start at index 1
    question_vector = tfidf_matrix[0]
    chunk_vectors   = tfidf_matrix[1:]

    # Compute cosine similarity between question and all chunks
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()

    # Get indices of top_k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


def find_best_chunk_with_scores(question: str, chunks: list[str], top_k: int = 3) -> list[dict]:
    """
    Same as find_best_chunk but also returns similarity scores.
    Useful for debugging and displaying retrieval confidence in the UI.

    Returns:
        A list of dicts: [{"chunk": str, "score": float}, ...]
    """
    if not chunks:
        return []

    corpus = [question] + chunks
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    question_vector = tfidf_matrix[0]
    chunk_vectors   = tfidf_matrix[1:]

    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    top_indices  = np.argsort(similarities)[::-1][:top_k]

    return [
        {"chunk": chunks[i], "score": round(float(similarities[i]), 4)}
        for i in top_indices
    ]