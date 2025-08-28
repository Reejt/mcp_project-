import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class VectorIndex:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.text_chunks = []
        self.embeddings = None

    def add_documents(self, docs: List[str]):
        vectors = self.model.encode(docs, show_progress_bar=False)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.index.add(vectors)
        self.text_chunks.extend(docs)

    def search(self, query: str, top_k: int = 5):
        q_vec = self.model.encode([query])
        D, I = self.index.search(q_vec, top_k)
        return [(self.text_chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0]) if i < len(self.text_chunks)]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
