"""
Job Description Parser - LLM-Powered Semantic Understanding.

Uses Groq (Mistral) for fast parsing and OpenAI (GPT-4) for nuanced
understanding of what the role truly requires beyond the keywords.

Key Insight: Job descriptions are often misleading. "5-9 years" doesn't
mean 5-9 years - it means "senior-level judgment with hands-on coding".
"""

import json
import re
from typing import Any, Dict, List

from .llm import GroqClient, OpenAIClient
from .llm.prompt_templates import PROMPT_TEMPLATES


class LLMJobParser:
    """
    Parse job descriptions using LLMs for semantic understanding.

    The key difference from traditional parsers:
    - Extracts requirements (what must-haves really are)
    - Identifies red flags (consulting, pure research, etc.)
    - Detects cultural signals (async-first, product mindset)
    - Interprets experience levels contextually
    """

    def __init__(self, fast_provider: str = "groq", quality_provider: str = "openai"):
        """
        Initialize parser with LLM clients.

        Args:
            fast_provider: 'groq' for fast parsing
            quality_provider: 'openai' for nuanced understanding
        """
        self.fast_client = GroqClient()
        self.quality_client = OpenAIClient()

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

        # Use Groq for fast initial parsing
        response = self.fast_client.generate(prompt, response_format="json")

        return json.loads(response.text)

    def extract_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Alternative: Extract just the key requirements for scoring.

        Returns a streamlined dict with only what the scorer needs.
        """
        parsed = self.parse(job_description)

        return {
            "role_title": parsed["role_title"],
            "experience_years": tuple(parsed["experience_years"].values()),
            "location": parsed["location"],
            "work_mode": parsed["work_mode"],
            "core_skills": parsed["core_skills"],
            "preferred_skills": parsed["preferred_skills"],
            "must_haves": parsed["must_haves"],
            "red_flags": parsed["red_flags"],
            "cultural_signals": parsed["cultural_signals"],
            "role_level": parsed["role_level"],
            "raw_parsing": parsed,  # Keep full parsing for debugging
        }


class HybridJobParser:
    """
    Hybrid approach: LLM for structure + regex for verification.

    Uses LLM to extract structure, then validates/extracts with regex
    patterns for confidence scoring.
    """

    def __init__(self):
        self.llm_parser = LLMJobParser()

        # Fallback regex patterns for validation
        self.experience_pattern = r"(\d+)[-\s]?\+?\s*[-\s]\s*(\d+)[-\s]?\+?\s*years?"
        self.location_pattern = r"(?:pune|noida|hyderabad|mumbai|delhi)[^\n]*"

    def parse(self, job_description: str) -> Dict[str, Any]:
        """
        Parse with LLM, fallback to regex for validation.
        """
        # Primary: LLM parsing
        llm_result = self.llm_parser.parse(job_description)

        # Secondary: Regex validation/enrichment
        experience_match = re.search(self.experience_pattern, job_description.lower())
        if experience_match and llm_result["experience_years"]["min"] == 5:
            # LLM might have defaulted, use regex if different
            llm_result["experience_years"] = {
                "min": int(experience_match.group(1)),
                "max": int(experience_match.group(2)),
            }

        return llm_result


# Backward compatibility
def parse_job_description(job_description: str) -> Dict[str, Any]:
    """
    Convenience function for parsing job descriptions.

    Args:
        job_description: Raw job description text

    Returns:
        Structured requirements dictionary
    """
    parser = LLMJobParser()
    return parser.parse(job_description)


# For docx files
def load_job_description(file_path: str) -> str:
    """
    Load job description from file.

    Supports .txt and .docx files.
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


if __name__ == "__main__":
    # Example usage
    sample_job = """
    Job Description: Senior AI Engineer — Founding Team
    Company: Redrob AI (Series A AI-native talent intelligence platform)
    Location: Pune/Noida, India (Hybrid — flexible cadence)
    Experience Required: 5–9 years

    Required Skills:
    - Production experience with embeddings-based retrieval systems
    - Experience with vector databases (Pinecone, Weaviate, Qdrant, etc.)
    - Strong Python skills
    - Experience designing evaluation frameworks (NDCG, MRR, MAP, A/B testing)

    Red Flags:
    - Pure research without production deployment
    - AI experience only from recent LangChain projects
    - Consulting background (TCS, Infosys, Wipro, etc.)
    """

    parser = LLMJobParser()
    result = parser.parse(sample_job)

    print("Parsed Job Requirements:")
    print(f"Role: {result.get('role_title', 'N/A')}")
    print(f"Experience: {result.get('experience_years', 'N/A')}")
    print(f"Core Skills: {result.get('core_skills', [])}")
