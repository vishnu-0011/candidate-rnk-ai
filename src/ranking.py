"""
Candidate Ranking Module

Generates the final ranked list of candidates with detailed reasoning.
Orchestrates job parsing, scoring, and output generation.
"""

import json
from pathlib import Path
from typing import Any

from .job_parser import JobDescriptionParser, load_job_description
from .scorer import CandidateScorer, ScoreBreakdown, load_candidates
from .utils import save_csv, truncate_reasoning, get_candidate_identity


class CandidateRanker:
    """
    Orchestrates the candidate ranking process.

    Combines job requirements parsing, multi-dimensional scoring,
    and generates the final ranked output with explanations.
    """

    def __init__(
        self,
        job_description_path: str | None = None,
        job_description_text: str | None = None,
    ):
        """
        Initialize the ranker.

        Either job_description_path or job_description_text must be provided.
        """
        self.job_requirements = None
        self.scorer = None

        if job_description_text:
            self._parse_job_description(job_description_text)
        elif job_description_path:
            text = load_job_description(job_description_path)
            self._parse_job_description(text)
        else:
            raise ValueError("Either job_description_path or job_description_text required")

    def _parse_job_description(self, text: str) -> None:
        """Parse the job description and initialize the scorer."""
        parser = JobDescriptionParser(text)
        self.job_requirements = parser.parse()

        # Prepare requirements dict for scorer
        self._requirements_dict = {
            "role_title": self.job_requirements.role_title,
            "experience_years": self.job_requirements.experience_years,
            "location": self.job_requirements.location,
            "work_mode": self.job_requirements.work_mode,
            "core_skills": self.job_requirements.core_skills,
            "preferred_skills": self.job_requirements.preferred_skills,
            "must_haves": self.job_requirements.must_haves,
            "red_flags": self.job_requirements.red_flags,
            "cultural_signals": self.job_requirements.cultural_signals,
            "priority_weights": self.job_requirements.priority_weights,
            "role_level": self.job_requirements.role_level,
        }

        self.scorer = CandidateScorer(self._requirements_dict)

    def rank_candidates(
        self, candidates: list[dict[str, Any]], top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rank a list of candidates.

        Args:
            candidates: List of candidate profile dictionaries
            top_n: Optional limit on number of top candidates to return

        Returns:
            List of ranked candidates with scores and reasoning
        """
        if not candidates:
            return []

        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            breakdown = self.scorer.score_candidate(candidate)
            scored_candidates.append({
                "candidate_id": breakdown.candidate_id,
                "score": breakdown.total_score,
                "skill_match": breakdown.skill_match,
                "experience_score": breakdown.experience_score,
                "behavioral_score": breakdown.behavioral_score,
                "cultural_fit": breakdown.cultural_fit,
                "availability_score": breakdown.availability_score,
                "reasoning": truncate_reasoning(breakdown.reasoning),
            })

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Apply top_n limit if specified
        if top_n:
            scored_candidates = scored_candidates[:top_n]

        # Add rank
        for i, candidate in enumerate(scored_candidates, 1):
            candidate["rank"] = i

        return scored_candidates

    def rank_from_file(
        self, candidates_path: str, top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Load candidates from file and rank them.

        Args:
            candidates_path: Path to JSONL file with candidate data
            top_n: Optional limit on number of top candidates to return

        Returns:
            List of ranked candidates
        """
        candidates = load_candidates(candidates_path)
        return self.rank_candidates(candidates, top_n)

    def generate_output(
        self,
        candidates_path: str,
        output_path: str,
        top_n: int | None = None,
    ) -> None:
        """
        Generate the final ranked output CSV file.

        Args:
            candidates_path: Path to JSONL file with candidate data
            output_path: Path for the output CSV file
            top_n: Optional limit on number of top candidates to return
        """
        ranked = self.rank_from_file(candidates_path, top_n)

        # Prepare output rows
        output_rows = []
        for candidate in ranked:
            output_rows.append({
                "candidate_id": candidate["candidate_id"],
                "rank": candidate["rank"],
                "score": f"{candidate['score']:.4f}",
                "reasoning": candidate["reasoning"],
            })

        # Save to CSV
        save_csv(
            output_rows,
            output_path,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )

        print(f"Ranked {len(output_rows)} candidates saved to {output_path}")


def run_ranking_system(
    job_description_path: str | None = None,
    job_description_text: str | None = None,
    candidates_path: str | None = None,
    output_path: str | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to run the complete ranking pipeline.

    Args:
        job_description_path: Path to job description file
        job_description_text: Job description text (alternative to path)
        candidates_path: Path to candidates JSONL file
        output_path: Path for output CSV
        top_n: Optional limit on top candidates

    Returns:
        List of ranked candidates
    """
    ranker = CandidateRanker(
        job_description_path=job_description_path,
        job_description_text=job_description_text,
    )

    if candidates_path:
        ranked = ranker.rank_from_file(candidates_path, top_n)
    else:
        raise ValueError("candidates_path is required")

    if output_path:
        ranker.generate_output(candidates_path, output_path, top_n)

    return ranked


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ranking.py <candidates_path> <output_path> [top_n]")
        sys.exit(1)

    candidates_path = sys.argv[1]
    output_path = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Load and run
    results = run_ranking_system(
        candidates_path=candidates_path,
        output_path=output_path,
        top_n=top_n,
    )

    # Print top 5
    print("\nTop 5 Candidates:")
    print("-" * 60)
    for r in results[:5]:
        print(f"{r['rank']}. {r['candidate_id']} - Score: {r['score']:.4f}")
        print(f"   {r['reasoning'][:100]}...")
