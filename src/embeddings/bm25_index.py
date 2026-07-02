"""
BM25 Sparse Index for Redrobe AI Candidate Ranking System.

BM25 is the industry standard for lexical search, still highly
effective for exact matches where dense embeddings can fail.

Why use BM25?
- Excellent for exact keyword matches
- Fast and deterministic
- Complements BGE embeddings (sparse + dense = hybrid search)
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any


class BM25Index:
    """
    BM25 sparse indexing for keyword-based search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.

        Args:
            k1: BM25 term saturation parameter
            b: BM25 length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._index = None
        self._documents = []
        self._vocab = {}
        self._doc_freq = {}
        self._total_docs = 0
        self._avg_doc_len = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split."""
        return text.lower().split()

    def fit(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of text documents to index
        """
        self._documents = documents
        self._total_docs = len(documents)

        # Build vocabulary and document frequencies
        self._doc_freq = {}
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            unique_tokens = set(tokens)

            for token in unique_tokens:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)

        self._vocab_size = len(self._vocab)
        self._avg_doc_len = total_len / self._total_docs if self._total_docs > 0 else 1

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a query against a document.

        Args:
            query_tokens: Tokenized query
            doc_idx: Index of document to score

        Returns:
            BM25 relevance score
        """
        doc = self._documents[doc_idx]
        doc_tokens = self._tokenize(doc)
        doc_freq = Counter(doc_tokens)

        score = 0.0

        for token, q_freq in Counter(query_tokens).items():
            if token in doc_freq:
                # BM25 formula
                doc_len = len(doc_tokens)
                idf = np.log((self._total_docs - self._doc_freq[token] + 0.5) /
                            (self._doc_freq[token] + 0.5) + 1)
                tf = (doc_freq[token] * (self.k1 + 1)) / (
                    doc_freq[token] + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                )
                score += q_freq * tf * idf

        return score

    def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query index for most similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of {index, score} dicts
        """
        query_tokens = self._tokenize(query)

        scores = []
        for i in range(len(self._documents)):
            score = self._bm25_score(query_tokens, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [{"index": int(idx), "score": float(score)} for idx, score in scores[:top_k]]
