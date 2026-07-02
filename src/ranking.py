"""
Candidate Ranking Module - Final Output Generator.

Orchestrates the full pipeline:
1. Job parsing (LLM for semantic understanding)
2. Multi-dimensional scoring (skill, behavioral, experience, cultural, availability)
3. LLM re-ranking with detailed explanations
4. CSV output generation

The output includes not just ranks and scores, but human-readable
explanations explaining WHY each candidate was ranked where they were.
"""

import json
import csv
from typing import Dict, List, Any
from pathlib import Path

from .llm import GroqClient, OpenAIClient
from .llm.prompt_templates import PROMPT_TEMPLATES
from .job_parser import JobDescriptionParser, load_job_description
from .scorer import CandidateScorer
from .embeddings import BGEEncoder


class CandidateRanker:
    """
    Full pipeline for candidate ranking with explanations.

    The system goes beyond simple scoring to provide:
    - Multi-dimensional evaluation
    - LLM-powered re-ranking for deeper understanding
    - Human-readable explanations for each ranking
    """

    def __init__(
        self,
        job_description: str | None = None,
        job_file_path: str | None = None,
        provider: str = "groq",
    ):
        """
        Initialize ranker.

        Args:
            job_description: Raw job description text
            job_file_path: Path to job description file (.txt or .docx)
            provider: 'groq' for fast processing, 'openai' for quality
        """
        if job_file_path:
            job_description = load_job_description(job_file_path)

        if not job_description:
            raise ValueError("Either job_description or job_file_path required")

        # Parse job description with LLM
        self.job_parser = JobDescriptionParser()
        self.parsed_job = self.job_parser.parse(job_description)

        # Extract requirements in the format scorer expects
        job_requirements = self.job_parser.extract_requirements(job_description)

        # Initialize scorer
        self.scorer = CandidateScorer(job_requirements)

        # LLM clients
        # Use OpenAI as fallback since Groq might not be installed
        try:
            from .llm import GroqClient
            groq_available = True
        except ImportError:
            groq_available = False

        if provider == "groq" and groq_available:
            self.fast_client = GroqClient()
            self.quality_client = GroqClient(model="llama-3.3-70b-versatile")
        else:
            self.fast_client = OpenAIClient()
            self.quality_client = OpenAIClient()

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        include_explanations: bool = True,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank a list of candidates.

        Args:
            candidates: List of candidate profile dictionaries
            include_explanations: Whether to generate LLM explanations
            top_n: Optional limit on top candidates to return

        Returns:
            List of ranked candidates with scores and explanations
        """
        if not candidates:
            return []

        # Step 1: Initial scoring (all dimensions)
        scored_candidates = []
        for candidate in candidates:
            result = self.scorer.score(candidate)
            candidate_with_scores = {
                **candidate,
                **result,
            }
            scored_candidates.append(candidate_with_scores)

        # Step 2: Initial sort by total score
        scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)

        # Apply top_n limit
        if top_n:
            scored_candidates = scored_candidates[:top_n]

        # Step 3: LLM re-ranking with explanations
        if include_explanations:
            ranked = self._llm_rerank(scored_candidates)
        else:
            # Simple ranking without LLM
            ranked = []
            for i, c in enumerate(scored_candidates, 1):
                ranked.append({
                    "candidate_id": c["candidate_id"],
                    "rank": i,
                    "score": c["total_score"],
                    "reasoning": self._generate_simple_reasoning(c),
                })

        return ranked

    def _llm_rerank(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using LLM with detailed analysis.

        The LLM considers deeper factors than the initial scoring.
        """
        # Prepare candidates for LLM - only include essential info to stay under rate limits
        # Limit to 5 candidates to stay under Groq's 12K TPM limit
        minimal_candidates = []
        for c in candidates[:5]:
            minimal_candidates.append({
                "candidate_id": c["candidate_id"],
                "score": c["total_score"],
                "scores": {
                    "skill_match": c.get("scores", {}).get("skill_match", 0),
                    "behavioral": c.get("scores", {}).get("behavioral", 0),
                    "experience": c.get("scores", {}).get("experience", 0),
                    "cultural_fit": c.get("scores", {}).get("cultural_fit", 0),
                    "availability": c.get("scores", {}).get("availability", 0),
                },
                "reasoning": c.get("reasoning", ""),
            })

        candidates_json = json.dumps(minimal_candidates, default=str, indent=2)

        prompt = PROMPT_TEMPLATES["rerank_candidates"].format(
            role_title=self.parsed_job.get("role_title", "this role"),
            candidates_json=candidates_json,
        )

        response = self.quality_client.generate(prompt, response_format="json")

        try:
            result = json.loads(response.text)
            ranked_candidates = result.get("ranked_candidates", [])

            # Normalize field names if LLM uses different names
            normalized = []
            for c in ranked_candidates:
                normalized.append({
                    "candidate_id": c.get("candidate_id", ""),
                    "rank": c.get("new_rank", c.get("rank", 0)),
                    "score": c.get("score", 0),
                    "reasoning": c.get("reason", c.get("reasoning", "")),
                })
            return normalized
        except json.JSONDecodeError:
            # Fallback to simple ranking if LLM parsing fails
            ranked = []
            for i, c in enumerate(candidates, 1):
                ranked.append({
                    "candidate_id": c["candidate_id"],
                    "rank": i,
                    "score": c["total_score"],
                    "reasoning": self._generate_simple_reasoning(c),
                })
            return ranked

    def _generate_simple_reasoning(self, candidate: Dict[str, Any]) -> str:
        """
        Generate simple reasoning for a candidate.
        """
        scores = candidate.get("scores", {})
        profile = candidate.get("profile", {})

        return (
            f"{profile.get('current_title', 'Unknown')} "
            f"with {profile.get('years_of_experience', 0):.1f} years experience. "
            f"Skill match: {scores.get('skill_match', 0):.2f}, "
            f"Behavioral: {scores.get('behavioral', 0):.2f}, "
            f"Experience: {scores.get('experience', 0):.2f}"
        )

    def generate_output(
        self,
        candidates_file: str,
        output_file: str,
        include_explanations: bool = True,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Load candidates from file and generate ranked output.

        Args:
            candidates_file: Path to JSONL file with candidates
            output_file: Path for output CSV
            include_explanations: Whether to generate explanations
            top_n: Optional limit on top candidates

        Returns:
            List of ranked candidates
        """
        # Load candidates
        with open(candidates_file, "r", encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f if line.strip()]

        # Rank them
        ranked = self.rank_candidates(
            candidates,
            include_explanations=include_explanations,
            top_n=top_n,
        )

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["candidate_id", "rank", "score", "reasoning"]
            )
            writer.writeheader()
            writer.writerows(ranked)

        print(f"Ranking complete. Output saved to {output_file}")
        print(f"Total candidates ranked: {len(ranked)}")

        return ranked


def run_ranking_system(
    job_file: str,
    candidates_file: str,
    output_file: str,
    top_n: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to run the complete ranking pipeline.

    Args:
        job_file: Path to job description file
        candidates_file: Path to candidates JSONL file
        output_file: Path for output CSV
        top_n: Optional limit on top candidates

    Returns:
        List of ranked candidates
    """
    ranker = CandidateRanker(job_file_path=job_file)
    return ranker.generate_output(
        candidates_file,
        output_file,
        top_n=top_n,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ranking.py <job_file> <candidates_file> [output_file] [top_n]")
        sys.exit(1)

    job_file = sys.argv[1]
    candidates_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "output/ranked_candidates.csv"
    top_n = int(sys.argv[4]) if len(sys.argv) > 4 else None

    results = run_ranking_system(job_file, candidates_file, output_file, top_n)

    # Print top 5
    print("\n" + "=" * 60)
    print("TOP 5 CANDIDATES")
    print("=" * 60)
    for r in results[:5]:
        print(f"\n#{r['rank']} - {r['candidate_id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Reasoning: {r['reasoning']}")
