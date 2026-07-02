# Redrobe LLM Integration Layer

"""
LLM integration for the Redrobe AI Candidate Ranking System.

This module provides:
- GroqClient: Fast Mistral-based LLM for parsing and batch processing
- OpenAIClient: High-quality GPT-4 for explanations and re-ranking
- Consistent interface across providers
"""

from .prompt_templates import PROMPT_TEMPLATES


def _get_groq_client():
    """Lazy import of GroqClient to avoid ImportError if groq not installed."""
    try:
        from .client import GroqClient
        return GroqClient
    except ImportError:
        return None


def _get_openai_client():
    """Lazy import of OpenAIClient."""
    from .client import OpenAIClient
    return OpenAIClient


# Define classes conditionally based on available dependencies
GroqClient = _get_groq_client()
OpenAIClient = _get_openai_client()

__all__ = ["GroqClient", "OpenAIClient", "PROMPT_TEMPLATES"]
