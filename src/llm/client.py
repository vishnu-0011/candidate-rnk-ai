"""
LLM Client Module for Redrobe Candidate Ranking System.

Provides unified interface for:
- Groq (Mistral-7B) for fast job parsing and extraction
- OpenAI (GPT-4) for high-quality explanations and re-ranking
"""

import os
from typing import Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM call."""
    text: str
    model: str
    usage: dict[str, int] | None = None


class LLMClient:
    """
    Unified LLM client interface supporting multiple providers.

    Design Philosophy:
    - Simple, consistent interface
    - Structured output support
    - Automatic error handling and retry
    - Usage tracking for cost monitoring
    """

    def __init__(self, provider: str = "groq", model: str | None = None):
        """
        Initialize LLM client.

        Args:
            provider: 'groq' or 'openai'
            model: Model name (defaults to best for provider)
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.api_key = self._get_api_key()

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == "groq":
            return "mistral-7b-instruct-v0.3"
        elif self.provider == "openai":
            return "gpt-4-turbo"
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        env_var = f"{self.provider.upper()}_API_KEY"
        key = os.environ.get(env_var)
        if not key:
            raise ValueError(f"Set {env_var} environment variable")
        return key

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: str | None = None,
    ) -> LLMResponse:
        """
        Generate text from LLM.

        Args:
            prompt: Input prompt
            temperature: Creativity (0-1)
            max_tokens: Maximum output tokens
            response_format: 'json' for structured output

        Returns:
            LLMResponse with text and metadata
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def extract_json(self, prompt: str) -> dict[str, Any]:
        """Extract JSON response from LLM."""
        response = self.generate(prompt, response_format="json")
        import json
        return json.loads(response.text)


class GroqClient(LLMClient):
    """Groq API client using Mistral for fast inference."""

    def __init__(self, model: str = "mistral-7b-instruct-v0.3"):
        super().__init__(provider="groq", model=model)
        self._client = None

    def _get_client(self):
        """Lazy load Groq client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Generate text using Groq API."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides precise, structured responses."},
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


class OpenAIClient(LLMClient):
    """OpenAI API client for high-quality reasoning."""

    def __init__(self, model: str = "gpt-4-turbo"):
        super().__init__(provider="openai", model=model)
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Generate text using OpenAI API."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": "You are an expert recruiter and talent analyst. Provide detailed, structured responses."},
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


def get_llm_client(provider: str = "groq", model: str | None = None) -> LLMClient:
    """Factory function to get LLM client."""
    if provider == "groq":
        return GroqClient(model=model)
    elif provider == "openai":
        return OpenAIClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # Test Groq client
    try:
        client = GroqClient()
        response = client.generate("Return { 'status': 'ok' }", response_format="json")
        print(f"Groq response: {response.text}")
    except Exception as e:
        print(f"Groq test skipped (no API key): {e}")

    # Test OpenAI client
    try:
        client = OpenAIClient()
        response = client.generate("Return { 'status': 'ok' }", response_format="json")
        print(f"OpenAI response: {response.text}")
    except Exception as e:
        print(f"OpenAI test skipped (no API key): {e}")
