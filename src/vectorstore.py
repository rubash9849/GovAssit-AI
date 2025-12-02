import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from tqdm import tqdm


def init_vectorstore(
    persist_directory: str = "data/vector_store",
    embedding_dim: int = 384,
):
    """
    Initialize or load a FAISS vector store.
    FAISS is much faster than ChromaDB for small-medium datasets.
    """
    os.makedirs(persist_directory, exist_ok=True)
    
    index_path = os.path.join(persist_directory, "faiss.index")
    metadata_path = os.path.join(persist_directory, "metadata.pkl")
    
    # Try to load existing index
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print(f"Loading existing FAISS index from {persist_directory}")
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded index with {index.ntotal} documents")
    else:
        print(f"Creating new FAISS index (dim={embedding_dim})")
        # Use IndexFlatL2 for exact search (best for < 1M vectors)
        index = faiss.IndexFlatL2(embedding_dim)
        metadata = {"documents": [], "metadatas": [], "ids": []}
    
    return index, metadata, persist_directory


def add_documents_to_store(
    index, 
    metadata: Dict, 
    persist_directory: str,
    documents: List[Any], 
    embeddings: np.ndarray
):
    """
    Add documents and embeddings to FAISS index.
    This is MUCH faster than ChromaDB!
    """
    if len(documents) != len(embeddings):
        raise ValueError("Number of documents and embeddings must match.")

    print(f"Adding {len(documents)} documents to FAISS index...")
    
    # Prepare data
    texts = []
    metadatas = []
    ids = []
    
    for i, doc in enumerate(tqdm(documents, desc="Preparing documents")):
        doc_id = f"doc_{i}"
        ids.append(doc_id)
        
        # Prepare metadata
        meta = dict(doc.metadata)
        meta["length"] = len(doc.page_content)
        meta["index"] = i
        metadatas.append(meta)
        
        # Store content
        texts.append(doc.page_content)
    
    # Add to FAISS (this is super fast!)
    print("Adding to FAISS index...")
    index.add(embeddings.astype('float32'))
    
    # Store metadata
    metadata["documents"].extend(texts)
    metadata["metadatas"].extend(metadatas)
    metadata["ids"].extend(ids)
    
    # Save to disk
    print("Saving index to disk...")
    index_path = os.path.join(persist_directory, "faiss.index")
    metadata_path = os.path.join(persist_directory, "metadata.pkl")
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"âœ… Successfully added {len(documents)} documents.")
    print(f"Total documents in index: {index.ntotal}")
    
    return index, metadata


def query_vectorstore(
    index,
    metadata: Dict,
    query_embedding: np.ndarray, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Query FAISS index - super fast!
    """
    if index.ntotal == 0:
        print("âš ï¸  Index is empty!")
        return []
    
    # Search (this is lightning fast with FAISS!)
    query_vector = query_embedding.astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, min(top_k, index.ntotal))
    
    # Format results
    formatted_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata["documents"]):
            # Convert L2 distance to similarity score (0-1, higher is better)
            similarity = 1 / (1 + dist)
            
            formatted_results.append({
                "content": metadata["documents"][idx],
                "metadata": metadata["metadatas"][idx],
                "similarity": float(similarity)
            })
    
    print(f"Retrieved {len(formatted_results)} documents.")
    return formatted_results

def count_documents(index):
    """Get total number of documents in index"""
    return index.ntotal