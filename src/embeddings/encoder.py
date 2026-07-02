"""
BGE Embedding Encoder for Redrobe AI Candidate Ranking System.

Uses BAAI/bge-large-en-v1.5 - State-of-the-Art sentence embeddings.

Why BGE?
- Outperforms OpenAI embeddings on MTEB leaderboard (2026)
- Handles both queries and passages well
- 1024-dimensional embeddings with good quality
- Open-source and reproducible
"""

import numpy as np
from typing import List, Union


class BGEEncoder:
    """
    BGE (BAAI/bge-large-en-v1.5) sentence encoder for embeddings.
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
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
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
        """
        Encode query text with BGE query prefix.

        BGE uses "Represent this sentence for searching..." prefix for queries.
        """
        return self.encode(f"Represent this sentence for searching: {text}")

    def encode_passage(self, text: str) -> np.ndarray:
        """Encode passage text without query prefix."""
        return self.encode(text)