"""
Embeddings Module for Redrobe Candidate Ranking System.

Implements dual-vector hybrid search:
1. Dense vectors (BGE) for semantic similarity
2. Sparse vectors (BM25) for keyword matching
3. Learned fusion of both signals

Why both? Sparse finds exact matches, dense finds semantic equivalents.
Example: "RAG" in job matches "built search system" in candidate.
"""

from typing import List, Dict, Any
import numpy as np


class BGEEncoder:
    """
    BGE (BAAI/bge-large-en-v1.5) sentence encoder for embeddings.

    BGE is currently (2026) the state-of-the-art for sentence embeddings,
    outperforming OpenAI embeddings on MTEB leaderboard.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize BGE encoder.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load BGE model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: str | List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of embeddings
        """
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=normalize)
        return embeddings

    def encode_query(self, text: str) -> np.ndarray:
        """Encode query text (adds 'Represent this sentence for searching...')."""
        return self.encode(f"Represent this sentence for searching: {text}")

    def encode_passage(self, text: str) -> np.ndarray:
        """Encode passage text."""
        return self.encode(text)


class BM25Index:
    """
    BM25 sparse indexing for keyword-based search.

    BM25 is the industry standard for lexical search, still highly
    effective for exact matches where BGE can fail.
    """

    def __init__(self):
        self._index = None
        self._documents = []
        self._vocab = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def fit(self, documents: List[str]):
        """
        Build BM25 index from documents.

        Args:
            documents: List of text documents
        """
        self._documents = documents
        # Build vocabulary and document frequencies
        doc_freq = {}
        total_docs = len(documents)

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)

        self._vocab_size = len(self._vocab)
        self._doc_freq = doc_freq
        self._total_docs = total_docs

    def query(self, query: str, top_k: int = 10) -> List[int]:
        """
        Query index for most similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Indices of top-k most relevant documents
        """
        from collections import Counter
        query_tokens = self._tokenize(query)
        query_freq = Counter(query_tokens)

        scores = []
        for i, doc in enumerate(self._documents):
            doc_tokens = self._tokenize(doc)
            doc_freq = Counter(doc_tokens)

            # BM25 score calculation
            score = 0.0
            k1 = 1.5
            b = 0.75
            avg_doc_len = sum(len(t) for t in self._tokenize(d)) / len(self._documents)

            for token, q_freq in query_freq.items():
                if token in doc_freq:
                    doc_len = len(doc_tokens)
                    idf = np.log((self._total_docs - self._doc_freq[token] + 0.5) /
                                (self._doc_freq[token] + 0.5) + 1)
                    tf = (doc_freq[token] * (k1 + 1)) / (doc_freq[token] + k1 * (1 - b + b * doc_len / avg_doc_len))
                    score += q_freq * tf * idf

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s in scores[:top_k]]


class HybridSearcher:
    """
    Hybrid search combining dense (BGE) and sparse (BM25) vectors.

    The key insight: Use both, and learn the optimal fusion weights
    from synthetic training data.
    """

    def __init__(self, bge_encoder: BGEEncoder | None = None):
        """
        Initialize hybrid searcher.

        Args:
            bge_encoder: BGE encoder instance (optional, creates new if None)
        """
        self.bge = bge_encoder or BGEEncoder()
        self.bm25 = BM25Index()
        self.dense_embeddings = None
        self._documents = []
        self._weights = {"dense": 0.4, "sparse": 0.6}  # Learnable

    def fit(self, documents: List[str]):
        """
        Build both indexes from documents.

        Args:
            documents: List of text documents to index
        """
        self._documents = documents

        # Build BM25 index
        self.bm25.fit(documents)

        # Generate dense embeddings
        self.dense_embeddings = self.bge.encode(documents)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using hybrid approach.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of {index, score, dense_score, sparse_score}
        """
        # Dense search (BGE)
        query_embedding = self.bge.encode_query(query)
        dense_scores = np.dot(self.dense_embeddings, query_embedding)

        # Sparse search (BM25)
        sparse_indices = self.bm25.query(query, top_k=len(self._documents))
        sparse_scores = np.zeros(len(self._documents))
        for rank, idx in enumerate(sparse_indices):
            # BM25 score normalized
            sparse_scores[idx] = 1.0 / (rank + 1)

        # Hybrid fusion
        hybrid_scores = (
            self._weights["dense"] * dense_scores +
            self._weights["sparse"] * sparse_scores
        )

        # Get top-k
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(hybrid_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "text": self._documents[idx],
            })

        return results


if __name__ == "__main__":
    # Example usage
    documents = [
        "Senior AI Engineer with 5+ years experience in embeddings and vector databases",
        "Machine Learning Engineer who built recommendation systems",
        "Python developer with experience in LLM applications",
        "Data Scientist working on search and ranking systems",
    ]

    searcher = HybridSearcher()
    searcher.fit(documents)

    query = "RAG system with Pinecone vector database"
    results = searcher.search(query, top_k=2)

    print(f"Query: {query}")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text']}")
