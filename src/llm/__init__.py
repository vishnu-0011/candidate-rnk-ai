# Redrobe LLM Integration Layer

"""
LLM integration for the Redrobe AI Candidate Ranking System.

This module provides:
- GroqClient: Fast Mistral-based LLM for parsing and batch processing
- OpenAIClient: High-quality GPT-4 for explanations and re-ranking
- Consistent interface across providers
"""

from .client import GroqClient, OpenAIClient, get_llm_client
from .prompt_templates import PROMPT_TEMPLATES

__all__ = ["GroqClient", "OpenAIClient", "get_llm_client", "PROMPT_TEMPLATES"]
