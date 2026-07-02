"""
LLM Integration Layer for Redrobe Candidate Ranking System.

Provides Groq (Mistral) for fast job parsing and OpenAI (GPT-4) for
high-quality explanations and re-ranking.
"""

from .groq_client import GroqClient
from .openai_client import OpenAIClient
from .prompt_templates import PROMPT_TEMPLATES

__all__ = ["GroqClient", "OpenAIClient", "PROMPT_TEMPLATES"]
