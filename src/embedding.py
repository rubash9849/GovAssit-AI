import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a sentence transformer embedding model.
    """
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print(
        f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}"
    )
    return model


def generate_embeddings(
    texts: List[str], model: SentenceTransformer, show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.
    Returns a NumPy array of shape (n_texts, embedding_dim).
    """
    if not texts:
        raise ValueError("No texts provided for embedding generation.")
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def get_embedding_dimension(model: SentenceTransformer) -> int:
    """
    Return the dimensionality of the embedding vectors.
    """
    return model.get_sentence_embedding_dimension()