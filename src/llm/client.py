"""
LLM Client for Redrobe AI Candidate Ranking System.

Unified interface for Groq (Mistral) and OpenAI (GPT-4) providers.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Load environment variables from .env file
# Try multiple paths to find .env file
env_paths = [
    Path(__file__).parent.parent.parent / ".env",
    Path.cwd() / ".env",
    Path(__file__).parent.parent / ".env",
]
for env_path in env_paths:
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        break


@dataclass
class LLMResponse:
    """Response from LLM call."""
    text: str
    model: str
    usage: Optional[Dict[str, int]] = None


class GroqClient:
    """Groq API client using LLM for fast inference."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self._client = None
        self._groq_available = self._check_groq_available()

    def _check_groq_available(self) -> bool:
        """Check if groq module is available."""
        try:
            from groq import Groq
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            if not self._groq_available:
                raise ImportError(
                    "groq module not installed. Install with: pip install groq "
                    "Or use --provider openai instead"
                )
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        return self._client

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text using Groq API."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Provide precise, structured responses in JSON format when requested."},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if response_format == "json" else None,
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            usage=dict(response.usage) if response.usage else None,
        )


class OpenAIClient:
    """OpenAI API client for high-quality reasoning (GPT-4)."""

    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text using OpenAI API."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": "You are an expert recruiter and talent analyst. Provide detailed, structured responses in JSON format when requested."},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if response_format == "json" else None,
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            usage=dict(response.usage) if response.usage else None,
        )


def get_llm_client(provider: str = "groq", model: Optional[str] = None) -> GroqClient | OpenAIClient:
    """Factory function to get LLM client."""
    if provider == "groq":
        return GroqClient(model=model)
    elif provider == "openai":
        return OpenAIClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")
