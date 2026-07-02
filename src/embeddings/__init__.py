"""
Embeddings Module for Redrobe AI Candidate Ranking System.

Implements dual-vector hybrid search:
1. Dense vectors (BGE) for semantic similarity
2. Sparse vectors (BM25) for keyword matching
3. Combined for best of both worlds

Key Insight: Sparse finds exact matches, dense finds semantic equivalents.
Example: "RAG" in job matches "built search system" in candidate.
"""

from .encoder import BGEEncoder
from .bm25_index import BM25Index
from .hybrid_searcher import HybridSearcher

__all__ = ["BGEEncoder", "BM25Index", "HybridSearcher"]
