"""
Hybrid Searcher for Redrobe AI Candidate Ranking System.

Combines dense (BGE) and sparse (BM25) vector search for superior results.

Why Hybrid?
- Dense (BGE): Finds semantic equivalents - "RAG" matches "search system"
- Sparse (BM25): Finds exact keyword matches
- Combined: Best of both worlds with learned fusion weights
"""

import numpy as np
from typing import List, Dict, Any, Optional

from .encoder import BGEEncoder
from .bm25_index import BM25Index


class HybridSearcher:
    """
    Hybrid search combining dense and sparse vectors.
    """

    def __init__(self, bge_encoder: Optional[BGEEncoder] = None):
        """
        Initialize hybrid searcher.

        Args:
            bge_encoder: BGE encoder instance (creates new if None)
        """
        self.bge = bge_encoder or BGEEncoder()
        self.bm25 = BM25Index()
        self.dense_embeddings = None
        self._documents = []
        self._weights = {"dense": 0.4, "sparse": 0.6}

    def fit(self, documents: List[str]) -> None:
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
            List of {index, score, dense_score, sparse_score, text}
        """
        # Dense search (BGE)
        query_embedding = self.bge.encode_query(query)
        dense_scores = np.dot(self.dense_embeddings, query_embedding)

        # Sparse search (BM25)
        sparse_results = self.bm25.query(query, top_k=len(self._documents))
        sparse_scores = np.zeros(len(self._documents))
        for result in sparse_results:
            sparse_scores[result["index"]] = result["score"]

        # Normalize scores to 0-1 range
        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()
        if sparse_scores.max() > 0:
            sparse_scores = sparse_scores / sparse_scores.max()

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
