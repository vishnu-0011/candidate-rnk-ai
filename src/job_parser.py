"""
Job Description Parser for Redrobe AI Candidate Ranking System.

Uses LLMs to semantically understand job descriptions, extracting:
- Core requirements vs nice-to-haves
- Red flags and disqualifiers
- Cultural signals and vibe check items
- Experience level interpretation

Key Insight: "5-9 years" doesn't mean literal 5-9 years - it means
"senior-level judgment with hands-on coding ability".
"""

import json
from typing import Any, Dict

from .llm import GroqClient, OpenAIClient
from .llm.prompt_templates import PROMPT_TEMPLATES


class JobDescriptionParser:
    """
    Parse job descriptions using LLMs for semantic understanding.
    """

    def __init__(self, provider: str = "groq"):
        """
        Initialize parser with LLM client.

        Args:
            provider: 'groq' for fast parsing, 'openai' for quality
        """
        self.client = GroqClient() if provider == "groq" else OpenAIClient()

    def parse(self, job_description: str) -> Dict[str, Any]:
        """
        Parse job description into structured requirements.

        Args:
            job_description: Raw job description text

        Returns:
            Dictionary with extracted requirements
        """
        prompt = PROMPT_TEMPLATES["parse_job_description"].format(
            job_description=job_description
        )

        response = self.client.generate(prompt, response_format="json")

        return json.loads(response.text)

    def extract_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract streamlined requirements for scoring.

        Returns only what the scorer needs to work with.
        """
        parsed = self.parse(job_description)

        return {
            "role_title": parsed["role_title"],
            "experience_years": (int(parsed["experience_years"]["min"]), int(parsed["experience_years"]["max"])),
            "location": parsed["location"],
            "work_mode": parsed["work_mode"],
            "core_skills": parsed["core_skills"],
            "preferred_skills": parsed["preferred_skills"],
            "must_haves": parsed["must_haves"],
            "red_flags": parsed["red_flags"],
            "cultural_signals": parsed["cultural_signals"],
            "role_level": parsed["role_level"],
        }


def load_job_description(file_path: str) -> str:
    """
    Load job description from file.

    Args:
        file_path: Path to .txt or .docx file

    Returns:
        Job description text
    """
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
