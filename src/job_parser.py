"""
Job Description Parser - Semantic understanding of job requirements.

This module parses job descriptions to extract:
1. Core technical requirements (skills, tools, experience)
2. Behavioral and cultural signals
3. Role-level priorities and trade-offs
4. Red flags and disqualifiers
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobRequirements:
    """Structured representation of job requirements."""
    role_title: str
    experience_years: tuple[int, int]
    location: str
    work_mode: str
    core_skills: list[str]
    preferred_skills: list[str]
    must_haves: list[str]
    red_flags: list[str]
    cultural_signals: list[str]
    priority_weights: dict[str, float]
    role_level: str  # 'senior', 'staff', 'principal'


class JobDescriptionParser:
    """
    Parse job descriptions with semantic understanding.

    Uses pattern matching and rule-based extraction to identify:
    - Required vs preferred skills
    - Cultural fit signals
    - Disqualifying criteria
    - Priority weights for different factors
    """

    # Skill patterns for technical skill extraction
    SKILL_PATTERNS = {
        "embedding_models": [
            r"sentence[-\s]?transformers",
            r"openai.*embed",
            r"bge",
            r"e5",
            r"embeddings",
            r"vector.*embed",
        ],
        "vector_databases": [
            r"pinecone",
            r"weaviate",
            r"qdrant",
            r"milvus",
            r"opensearch",
            r"elasticsearch",
            r"faiss",
            r"vector.*databas",
        ],
        "llm_frameworks": [
            r"langchain",
            r"llama.*ind",
            r"pytorch",
            r"tensorflow",
            r"transformers",
            r"fine[-\s]?tun",
            r"lora",
            r"qlora",
        ],
        "ranking_search": [
            r"rank",
            r"retriev",
            r"search",
            r"recommend",
            r"bm25",
            r"hybrid.*search",
            r"rag",
        ],
    }

    # Cultural and behavioral signal keywords
    CULTURAL_SIGNALS = {
        "async_communication": [
            r"async",
            r"write.*lot",
            r"documentation",
        ],
        "product_mindset": [
            r"product",
            r"ship",
            r"working.*user",
            r"learning.*user",
        ],
        "ambiguity_tolerance": [
            r"unclear",
            r"ambiguous",
            r"changing",
            r"pivot",
        ],
        "seniority_level": [
            r"senior",
            r"staff",
            r"principal",
            r"end[-\s]?to[-\s]?end",
            r"own",
        ],
    }

    # Red flag patterns (disqualifiers)
    RED_FLAGS = [
        r"pure research",
        r"academic lab",
        r"research only",
        r"consulting firm",
        r"tcs",
        r"infosys",
        r"wipro",
        r"accenture",
        r"cognizant",
        r"capgemini",
        r"computer vision",
        r"speech",
        r"robotics",
        r"closed[-\s]?source",
    ]

    def __init__(self, job_description_text: str):
        """Initialize parser with job description text."""
        self.text = job_description_text.lower()
        self.requirements = None

    def parse(self) -> JobRequirements:
        """
        Parse the job description and extract structured requirements.

        Returns:
            JobRequirements object with all extracted information.
        """
        # Extract role title
        role_title = self._extract_role_title()

        # Extract experience range
        experience_years = self._extract_experience()

        # Extract location
        location = self._extract_location()

        # Extract work mode
        work_mode = self._extract_work_mode()

        # Extract skills
        skills = self._extract_skills()

        # Extract red flags
        red_flags = self._extract_red_flags()

        # Extract cultural signals
        cultural_signals = self._extract_cultural_signals()

        # Determine role level and priority weights
        role_level = self._determine_role_level()
        priority_weights = self._calculate_priority_weights(role_level, skills)

        self.requirements = JobRequirements(
            role_title=role_title,
            experience_years=experience_years,
            location=location,
            work_mode=work_mode,
            core_skills=skills["core"],
            preferred_skills=skills["preferred"],
            must_haves=skills["core"],  # For now, core = must have
            red_flags=red_flags,
            cultural_signals=cultural_signals,
            priority_weights=priority_weights,
            role_level=role_level,
        )

        return self.requirements

    def _extract_role_title(self) -> str:
        """Extract the role title from the job description."""
        # Look for "Senior AI Engineer" pattern
        patterns = [
            r"senior\s+(ai|artificial intelligence)\s+engineer",
            r"ai\s+engineer",
            r"(sr|staff|principal)\s+engineer",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.text)
            if match:
                return match.group(0).title()
        return "AI Engineer"

    def _extract_experience(self) -> tuple[int, int]:
        """Extract experience range (min_years, max_years)."""
        # Look for patterns like "5-9 years" or "5 to 9 years"
        patterns = [
            r"(\d+)[-\s]?\+?\s*[-\s]\s*(\d+)[-\s]?\+?\s*years?",
            r"(\d+)\s*-\s*(\d+)\s*years?",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.text)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        return (5, 9)  # Default based on known job description

    def _extract_location(self) -> str:
        """Extract location information."""
        patterns = [
            r"location[:\s]+([^\n]+)",
            r"(?:pune|noida|hyderabad|mumbai|delhi)[^\n]*",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Pune/Noida, India"

    def _extract_work_mode(self) -> str:
        """Extract preferred work mode."""
        if "hybrid" in self.text:
            return "hybrid"
        elif "remote" in self.text:
            return "remote"
        elif "onsite" in self.text or "office" in self.text:
            return "onsite"
        return "flexible"

    def _extract_skills(self) -> dict[str, list[str]]:
        """Extract skills from job description."""
        skills = {"core": [], "preferred": []}

        # Core skills based on job description analysis
        core_patterns = [
            (r"embeddings?", "Embeddings"),
            (r"vector.*databas|pinecone|weaviate|qdrant|milvus|faiss", "Vector Databases"),
            (r"python", "Python"),
            (r"evalu|ndcg|mrr|map|a/b", "Evaluation Frameworks"),
            (r"retrieval|rank|search", "Retrieval/Ranking"),
            (r"llm|large language model", "LLMs"),
            (r"fine[-\s]?tun", "Fine-tuning"),
        ]

        for pattern, skill_name in core_patterns:
            if re.search(pattern, self.text):
                skills["core"].append(skill_name)

        # Preferred skills
        preferred_patterns = [
            (r"lora|qlora", "LoRA/QLoRA"),
            (r"learning[-\s]?to[-\s]?rank|xgboost", "Learning-to-Rank"),
            (r"hr[-\s]?tech|recruit", "HR Tech"),
            (r"marketplace", "Marketplace Products"),
            (r"distribut|scal|inference", "Distributed Systems"),
        ]

        for pattern, skill_name in preferred_patterns:
            if re.search(pattern, self.text):
                skills["preferred"].append(skill_name)

        return skills

    def _extract_red_flags(self) -> list[str]:
        """Extract disqualifying criteria."""
        red_flags = []

        flag_patterns = {
            "Pure Research": r"pure research|academic lab|research only",
            "Consulting Background": r"consulting firm|tcs|infosys|wipro|accenture|cognizant|capgemini",
            "No Production Experience": r"only worked at consulting",
            "Wrong Technical Domain": r"computer vision|speech|robotics",
            "No External Validation": r"closed[-\s]?source.*proprietary.*5",
        }

        for flag_name, pattern in flag_patterns.items():
            if re.search(pattern, self.text):
                red_flags.append(flag_name)

        return red_flags

    def _extract_cultural_signals(self) -> list[str]:
        """Extract cultural fit signals from job description."""
        signals = []

        signal_patterns = {
            "Async Communication": r"async",
            "Strong Writing Skills": r"write.*lot|documentation",
            "Disagree Openly": r"disagree.*open",
            "Move Fast": r"move fast|break things",
            "Product Mindset": r"product[-\s]?engineer|ship.*working",
            "Comfort with Ambiguity": r"unclear|ambiguous|changing|pivot",
        }

        for signal_name, pattern in signal_patterns.items():
            if re.search(pattern, self.text):
                signals.append(signal_name)

        return signals

    def _determine_role_level(self) -> str:
        """Determine the role level."""
        if "staff" in self.text:
            return "staff"
        elif "senior" in self.text:
            return "senior"
        elif "principal" in self.text:
            return "principal"
        return "senior"

    def _calculate_priority_weights(self, role_level: str, skills: dict) -> dict[str, float]:
        """
        Calculate weights for different ranking factors based on job requirements.

        Returns dict mapping factor names to weights (sum = 1.0).
        """
        # Base weights - can be adjusted based on role
        weights = {
            "skill_match": 0.35,
            "experience_relevance": 0.20,
            "behavioral_signals": 0.20,
            "cultural_fit": 0.15,
            "availability": 0.10,
        }

        # Adjust based on role level
        if role_level == "staff" or role_level == "principal":
            weights["cultural_fit"] += 0.10
            weights["skill_match"] -= 0.10

        return weights


def load_job_description(file_path: str) -> str:
    """
    Load job description from various file formats.

    Supports:
    - .txt (plain text)
    - .docx (Word document)
    """
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


# Example usage
if __name__ == "__main__":
    # Example: parse a job description
    sample_text = """
    Job Description: Senior AI Engineer

    Required:
    - 5-9 years of experience
    - Production experience with embeddings-based retrieval systems
    - Experience with vector databases (Pinecone, Weaviate, Qdrant, etc.)
    - Strong Python skills
    - Experience designing evaluation frameworks

    Location: Pune/Noida (Hybrid)
    """

    parser = JobDescriptionParser(sample_text)
    requirements = parser.parse()

    print(f"Role: {requirements.role_title}")
    print(f"Experience: {requirements.experience_years}")
    print(f"Core Skills: {requirements.core_skills}")
    print(f"Cultural Signals: {requirements.cultural_signals}")
