import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MCPWrapperOllama:
    def __init__(self, model_name="llama3.2:3b-8k", chunk_size=300, embedding_model="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.doc_chunks = []  # stores text chunks aligned with FAISS vectors

    # ---------------- Chunking ----------------
    def chunk_text(self, text):
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    # ---------------- Index ----------------
    def _ensure_index(self, dim):
        """Initialize FAISS index with cosine similarity if not exists."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine) index

    def ingest_documents(self, documents, refresh=True):
        """
        Embed and store document chunks in FAISS.
        refresh=True -> clear and rebuild index
        refresh=False -> append new embeddings
        """
        if refresh:
            self.doc_chunks.clear()
            if self.index:
                self.index.reset()

        embeddings = []

        for doc in documents:
            chunks = self.chunk_text(doc)
            self.doc_chunks.extend(chunks)
            chunk_embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(chunk_embeddings)
            embeddings.append(chunk_embeddings)

        if not embeddings:
            return

        all_embeddings = np.vstack(embeddings)
        self._ensure_index(all_embeddings.shape[1])
        self.index.add(all_embeddings)

    # ---------------- Retrieval ----------------
    def retrieve_with_scores(self, query, top_k=5):
        """Return chunks with cosine similarity scores (0–1)."""
        if self.index is None or len(self.doc_chunks) == 0:
            return [], []

        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)

        results = [self.doc_chunks[i] for i in indices[0] if i != -1]
        scores = [float(s) for s in scores[0] if s != -1]

        return results, scores

    # ---------------- Prompt Builder ----------------
    def build_prompt(self, question, context_chunks=None):
        context = "\n\n".join(context_chunks) if context_chunks else ""
        if context:
            return f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
        else:
            return f"Question:\n{question}\nAnswer:"

    # ---------------- Query ----------------
    def query(self, question, strong_threshold=0.7, weak_threshold=0.5, top_k=5):
        """
        Hybrid logic with cosine similarity:
        - Strong match (score >= strong_threshold): only documents
        - Weak match (score >= weak_threshold): documents + model reasoning
        - No match (score < weak_threshold): base model only
        """
        relevant_chunks, scores = self.retrieve_with_scores(question, top_k=top_k)

        print(f"[DEBUG] Cosine Scores: {scores}")
        print(f"[DEBUG] Relevant Chunks: {relevant_chunks}")

        if relevant_chunks and scores[0] >= strong_threshold:
            # Strong match: answer strictly from docs
            prompt = self.build_prompt(question, relevant_chunks)
        elif relevant_chunks and scores[0] >= weak_threshold:
            # Weak match: combine docs with general reasoning
            doc_context = "\n\n".join(relevant_chunks)
            prompt = (
                f"Use the following context if relevant, but also apply your own reasoning:\n"
                f"{doc_context}\n\nQuestion:\n{question}\nAnswer:"
            )
        else:
            # No match: base model only
            prompt = f"Question:\n{question}\nAnswer:"

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"num_ctx": 8192}
        )

        return response['response']
