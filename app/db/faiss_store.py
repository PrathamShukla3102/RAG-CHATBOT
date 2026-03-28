import os
import faiss
import numpy as np
import pickle
from typing import List, Dict


class FAISSStore:
    def __init__(self, dim: int, index_path: str = "vector_store/faiss.index", metadata_path: str = "vector_store/metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        # Load if exists
        if os.path.exists(self.index_path):
            self._load()

    # ---------- Add ----------
    def add(self, embeddings: List[List[float]], chunks: List[Dict]):
        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)
        self.metadata.extend(chunks)

    # ---------- Search ----------
    def search(self, query_embedding: List[float], top_k: int = 5):
        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "score": float(distances[0][i]),
                    "text": self.metadata[idx]["text"],
                    "source": self.metadata[idx]["source"],
                    "chunk_id": self.metadata[idx]["chunk_id"]
                })

        return results

    # ---------- Save ----------
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    # ---------- Load ----------
    def _load(self):
        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)