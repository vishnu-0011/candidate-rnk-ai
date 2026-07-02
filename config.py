# Redrobe AI Candidate Ranking System Configuration

"""
Configuration for the AI-powered candidate ranking system.

This module defines all configurable parameters including:
- LLM provider and model settings
- Scoring weights for each dimension
- Behavioral signal thresholds
- Path configurations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "groq"  # 'groq' (fast) or 'openai' (quality)
    fast_model: str = "mistral-7b-instruct-v0.3"
    quality_model: str = "llama3-70b-8192"
    openai_model: str = "gpt-4-turbo"


@dataclass
class ScoringConfig:
    """Scoring weights for each dimension."""
    skill_match: float = 0.35
    behavioral: float = 0.25
    experience: float = 0.20
    cultural_fit: float = 0.10
    availability: float = 0.10

    @property
    def weights(self) -> Dict[str, float]:
        """Return all weights as dict."""
        return {
            "skill_match": self.skill_match,
            "behavioral": self.behavioral,
            "experience": self.experience,
            "cultural_fit": self.cultural_fit,
            "availability": self.availability,
        }


@dataclass
class PathConfig:
    """Path configuration for data files."""
    data_dir: str = "data"
    output_dir: str = "output"
    candidates_file: str = "candidates.jsonl"
    job_description_file: str = "job_description.txt"
    output_file: str = "ranked_candidates.csv"


# Global configuration
LLM_CONFIG = LLMConfig()
SCORING_CONFIG = ScoringConfig()
PATH_CONFIG = PathConfig()

# AI Core Skills to look for
AI_CORE_SKILLS: List[str] = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Embeddings",
    "Vector Databases",
    "Retrieval",
    "Ranking",
    "LLMs",
    "Fine-tuning",
    "RAG",
    "Transformers",
    "OpenAI API",
    "PyTorch",
    "TensorFlow",
    "Hugging Face",
]

# Behavioral signal thresholds
BEHAVIORAL_THRESHOLDS = {
    "response_rate_high": 0.7,
    "response_rate_medium": 0.4,
    "response_rate_low": 0.2,
    "last_active_days_threshold": 30,
    "interview_completion_high": 0.9,
    "interview_completion_low": 0.5,
}

# Red flag indicators
RED_FLAGS = [
    "pure research",
    "academic lab",
    "research only",
    "consulting firm",
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "computer vision",
    "speech",
    "robotics",
    "closed-source",
]


def get_scoring_weights_for_role(role_level: str) -> Dict[str, float]:
    """
    Get adjusted scoring weights based on role level.

    Args:
        role_level: 'senior', 'staff', or 'principal'

    Returns:
        Dictionary of scoring weights
    """
    weights = SCORING_CONFIG.weights.copy()

    if role_level == "staff" or role_level == "principal":
        # Increase cultural fit weight for senior roles
        weights["cultural_fit"] = 0.20
        weights["skill_match"] = 0.30
        weights["behavioral"] = 0.25

    return weights


if __name__ == "__main__":
    print("Redrobe Ranking Configuration")
    print("=" * 50)
    print(f"\nLLM Provider: {LLM_CONFIG.provider}")
    print(f"Fast Model: {LLM_CONFIG.fast_model}")
    print(f"Quality Model: {LLM_CONFIG.quality_model}")

    print("\nScoring Weights:")
    for dim, weight in SCORING_CONFIG.weights.items():
        print(f"  {dim}: {weight:.2f}")

    print(f"\nData Path: {PATH_CONFIG.data_dir}")
    print(f"Output Path: {PATH_CONFIG.output_dir}")
