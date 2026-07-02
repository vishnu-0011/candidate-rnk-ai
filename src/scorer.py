"""
Candidate Scorer - Multi-Dimensional Scoring Engine.

Combines:
1. Semantic Skill Matching (using BGE embeddings)
2. Behavioral Signal Scoring (using LightGBM)
3. Experience Relevance
4. Cultural Alignment
5. Availability Assessment

The scoring is NOT a simple weighted average - it uses an ensemble
approach where different signals contribute differently based on
the role and candidate profile.
"""

from typing import Dict, List, Any
import numpy as np

from .embeddings import BGEEncoder, HybridSearcher
from .behavioral import BehavioralScorer, generate_synthetic_behavioral_data


class MultiDimensionalScorer:
    """
    Multi-dimensional candidate scorer with LLM-powered understanding.

    Each dimension is scored independently, then combined with
    learned weights for the final score.
    """

    def __init__(self, job_requirements: Dict[str, Any] | None = None):
        """
        Initialize scorer.

        Args:
            job_requirements: Parsed job requirements (optional)
        """
        self.job_reqs = job_requirements or {}
        self.job_embedding = None

        # Components
        self.bge = BGEEncoder()
        self.behavioral_scorer = BehavioralScorer()
        self.hybrid_searcher = HybridSearcher()

        # Training data
        self._trained = False

    def set_job_requirements(self, requirements: Dict[str, Any]) -> None:
        """Set job requirements and compute job embedding."""
        self.job_reqs = requirements

        # Compute job description embedding
        job_text = " ".join(requirements.get("core_skills", []))
        job_text += " " + requirements.get("role_title", "")
        self.job_embedding = self.bge.encode(job_text)

    def fit_synthetic(self, n_samples: int = 1000) -> None:
        """
        Train behavioral scorer on synthetic data.

        Args:
            n_samples: Number of synthetic samples to generate
        """
        X, y = generate_synthetic_behavioral_data(n_samples)
        self.behavioral_scorer.fit(X, y)
        self._trained = True

    def score_skill_match(self, candidate: Dict[str, Any]) -> float:
        """
        Score how well candidate skills match job requirements.

        Uses hybrid search to find semantic matches between
        job requirements and candidate profiles.
        """
        if not self.job_reqs:
            return 0.5

        core_skills = self.job_reqs.get("core_skills", [])
        candidate_skills = candidate.get("skills", [])

        # Build skill corpus
        skill_texts = []
        for skill in candidate_skills:
            skill_name = skill.get("name", "")
            proficiency = skill.get("proficiency", "unknown")
            skill_texts.append(f"{skill_name} {proficiency}")

        skill_texts.append(" ".join(core_skills))

        # Use hybrid search
        self.hybrid_searcher.fit(skill_texts)

        # Score each core skill
        skill_scores = []
        for req_skill in core_skills:
            results = self.hybrid_searcher.search(req_skill, top_k=5)
            if results:
                # Best match score
                best_score = results[0]["score"]
                skill_scores.append(best_score)

        if skill_scores:
            return np.mean(skill_scores)
        return 0.0

    def score_behavioral(self, candidate: Dict[str, Any]) -> float:
        """
        Score based on behavioral signals.
        """
        return self.behavioral_scorer.predict(candidate)

    def score_experience(self, candidate: Dict[str, Any]) -> float:
        """
        Score candidate experience relevance.

        Considers:
        - Years of experience vs requirements
        - Career progression
        - Industry relevance
        - Company quality
        """
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])

        # Extract requirements
        job_exp = self.job_reqs.get("experience_years", (5, 9))
        min_exp, max_exp = job_exp

        current_exp = float(profile.get("years_of_experience", 0))

        # Experience score (0-1)
        if current_exp < min_exp:
            # Under-experienced
            score = 0.5 + (current_exp / min_exp) * 0.3
        elif current_exp > max_exp:
            # Over-experienced
            score = 0.8 + (max_exp / current_exp) * 0.2
        else:
            # In range
            score = 0.8 + (current_exp - min_exp) / (max_exp - min_exp) * 0.2

        # Career progression bonus
        if len(career) >= 2:
            titles = [c.get("title", "").lower() for c in career]
            has_progression = any(
                any(t in title for t in ["senior", "lead", "staff", "principal"])
                for title in titles[-2:]
            )
            if has_progression:
                score += 0.1

        # Industry relevance
        industry = profile.get("current_industry", "").lower()
        tech_industries = ["ai", "tech", "software", "product", "data"]
        if any(ind in industry for ind in tech_industries):
            score += 0.05

        return min(1.0, score)

    def score_cultural_fit(self, candidate: Dict[str, Any]) -> float:
        """
        Score cultural alignment.

        Based on signals indicating fit with Redrob's culture:
        - Async communication (profile completeness, documentation)
        - Product mindset (project types)
        - Technical passion (GitHub, contributions)
        """
        signals = candidate.get("redrob_signals", {})
        profile = candidate.get("profile", {})

        score = 0.5  # Start neutral

        # GitHub activity (proxy for technical passion)
        github = signals.get("github_activity_score", -1)
        if github > 70:
            score += 0.2
        elif github > 0:
            score += 0.1

        # Profile completeness
        completeness = signals.get("profile_completeness_score", 0) / 100.0
        score += completeness * 0.1

        # Verification signals
        verified = sum([
            signals.get("verified_email", False),
            signals.get("verified_phone", False),
            signals.get("linkedin_connected", False),
        ])
        score += verified * 0.05

        # Connection count (networking ability)
        connections = signals.get("connection_count", 0)
        if connections > 500:
            score += 0.1
        elif connections > 100:
            score += 0.05

        # Role title alignment
        title = profile.get("current_title", "").lower()
        if any(t in title for t in ["ai", "ml", "engineer", "developer"]):
            score += 0.1

        return min(1.0, score)

    def score_availability(self, candidate: Dict[str, Any]) -> float:
        """
        Score candidate availability.

        Based on factors affecting how quickly they can join.
        """
        signals = candidate.get("redrob_signals", {})

        score = 0.5  # Start neutral

        # Notice period (lower is better)
        notice = signals.get("notice_period_days", 60)
        if notice <= 15:
            score += 0.3
        elif notice <= 30:
            score += 0.25
        elif notice <= 60:
            score += 0.1
        else:
            score += 0.05

        # Willing to relocate
        if signals.get("willing_to_relocate", False):
            score += 0.2

        # Open to work
        if signals.get("open_to_work_flag", False):
            score += 0.1

        return min(1.0, score)

    def score_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a candidate across all dimensions.

        Args:
            candidate: Candidate profile dictionary

        Returns:
            Dictionary with scores and breakdown
        """
        scores = {
            "skill_match": self.score_skill_match(candidate),
            "behavioral": self.score_behavioral(candidate),
            "experience": self.score_experience(candidate),
            "cultural_fit": self.score_cultural_fit(candidate),
            "availability": self.score_availability(candidate),
        }

        # Compute weighted total
        weights = {
            "skill_match": 0.35,
            "behavioral": 0.25,
            "experience": 0.20,
            "cultural_fit": 0.10,
            "availability": 0.10,
        }

        total = sum(scores[dim] * weights[dim] for dim in scores)

        return {
            "total_score": total,
            "scores": scores,
            "weights": weights,
            "dimension_breakdown": {
                dim: f"{scores[dim]:.3f} x {weights[dim]:.2f} = {scores[dim]*weights[dim]:.3f}"
                for dim in scores
            },
        }


class CandidateScorer(MultiDimensionalScorer):
    """
    Simplified interface for scoring candidates.

    This is the main entry point for the scoring system.
    """

    def __init__(self, job_requirements: Dict[str, Any] | None = None):
        super().__init__(job_requirements)
        # Pre-train on synthetic data
        self.fit_synthetic(1000)

    def score(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single candidate.

        Args:
            candidate: Candidate profile dictionary

        Returns:
            Scoring results including total score and breakdown
        """
        return self.score_candidate(candidate)

    def score_many(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple candidates.

        Args:
            candidates: List of candidate profiles

        Returns:
            List of scoring results
        """
        return [self.score(c) for c in candidates]


if __name__ == "__main__":
    # Example usage
    candidate = {
        "candidate_id": "CAND_0001234",
        "profile": {
            "headline": "Senior AI Engineer",
            "years_of_experience": 7.5,
            "current_title": "Machine Learning Engineer",
            "current_company": "Tech Corp",
        },
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 45},
            {"name": "Machine Learning", "proficiency": "expert", "endorsements": 38},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 20},
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.76,
            "last_active_date": "2024-01-15",
            "open_to_work_flag": True,
            "profile_completeness_score": 85,
            "github_activity_score": 78,
            "connection_count": 500,
            "verified_email": True,
            "verified_phone": True,
        },
    }

    # Mock job requirements
    job_reqs = {
        "core_skills": ["Python", "Machine Learning", "Embeddings"],
        "experience_years": (5, 9),
        "role_title": "Senior AI Engineer",
    }

    scorer = CandidateScorer(job_reqs)
    result = scorer.score(candidate)

    print(f"Candidate: {candidate['candidate_id']}")
    print(f"Total Score: {result['total_score']:.3f}")
    print("\nDimension Breakdown:")
    for dim, desc in result["dimension_breakdown"].items():
        print(f"  {dim}: {desc}")
