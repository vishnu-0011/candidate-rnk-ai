"""
Candidate Scorer - Multi-dimensional candidate evaluation.

This module implements the core scoring logic that evaluates candidates
based on:
- Skill match with job requirements
- Experience relevance and quality
- Behavioral signals (engagement, responsiveness)
- Cultural fit indicators
- Availability factors
"""

import json
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict
import math


@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown for a candidate."""
    candidate_id: str
    total_score: float
    skill_match: float = 0.0
    experience_score: float = 0.0
    behavioral_score: float = 0.0
    cultural_fit: float = 0.0
    availability_score: float = 0.0
    reasoning: list[str] = field(default_factory=list)


class CandidateScorer:
    """
    Score candidates based on multi-dimensional criteria.

    The scoring system goes beyond simple keyword matching to understand
    what truly makes a candidate a good fit for the role.
    """

    def __init__(
        self,
        job_requirements: dict[str, Any],
        weight_config: dict[str, float] | None = None,
    ):
        """
        Initialize scorer with job requirements.

        Args:
            job_requirements: Parsed job requirements including skills, priorities
            weight_config: Optional override for scoring weights
        """
        self.job_requirements = job_requirements
        self.weights = weight_config or {
            "skill_match": 0.35,
            "experience_relevance": 0.20,
            "behavioral_signals": 0.20,
            "cultural_fit": 0.15,
            "availability": 0.10,
        }

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def score_candidate(self, candidate: dict[str, Any]) -> ScoreBreakdown:
        """
        Score a single candidate against the job requirements.

        Args:
            candidate: Candidate profile dictionary

        Returns:
            ScoreBreakdown with detailed scoring and reasoning
        """
        candidate_id = candidate.get("candidate_id", "UNKNOWN")
        breakdown = ScoreBreakdown(candidate_id=candidate_id, total_score=0.0)

        # Calculate individual scores
        skill_score = self._calculate_skill_score(candidate, breakdown)
        experience_score = self._calculate_experience_score(candidate, breakdown)
        behavioral_score = self._calculate_behavioral_score(candidate, breakdown)
        cultural_score = self._calculate_cultural_fit_score(candidate, breakdown)
        availability_score = self._calculate_availability_score(candidate, breakdown)

        # Calculate weighted total
        breakdown.skill_match = skill_score
        breakdown.experience_score = experience_score
        breakdown.behavioral_score = behavioral_score
        breakdown.cultural_fit = cultural_score
        breakdown.availability_score = availability_score

        breakdown.total_score = (
            skill_score * self.weights["skill_match"]
            + experience_score * self.weights["experience_relevance"]
            + behavioral_score * self.weights["behavioral_signals"]
            + cultural_score * self.weights["cultural_fit"]
            + availability_score * self.weights["availability"]
        )

        # Add overall reasoning
        breakdown.reasoning.append(
            f"Overall score: {breakdown.total_score:.3f} "
            f"(skills: {skill_score:.2f}, exp: {experience_score:.2f}, "
            f"behavioral: {behavioral_score:.2f}, cultural: {cultural_score:.2f}, "
            f"availability: {availability_score:.2f})"
        )

        return breakdown

    def _calculate_skill_score(
        self, candidate: dict[str, Any], breakdown: ScoreBreakdown
    ) -> float:
        """
        Calculate skill match score (0-1).

        Considers:
        - Direct skill matches with job requirements
        - Skill proficiency levels
        - Skill duration/experience
        - Redrob skill assessment scores
        """
        candidate_skills = candidate.get("skills", [])
        candidate_skill_names = {s.get("name", "").lower() for s in candidate_skills}

        job_core_skills = [s.lower() for s in self.job_requirements.get("core_skills", [])]
        job_preferred_skills = [s.lower() for s in self.job_requirements.get("preferred_skills", [])]

        # Calculate matches
        core_matches = []
        preferred_matches = []
        missing_core = []

        for skill in job_core_skills:
            matched = False
            for cs in candidate_skill_names:
                if skill in cs or cs in skill:
                    core_matches.append(skill)
                    matched = True
                    break
            if not matched:
                missing_core.append(skill)

        for skill in job_preferred_skills:
            if any(skill in cs or cs in skill for cs in candidate_skill_names):
                preferred_matches.append(skill)

        # Base skill score from matches
        skill_score = 0.0
        if job_core_skills:
            skill_score = len(core_matches) / len(job_core_skills)
        elif job_preferred_skills:
            skill_score = len(preferred_matches) / len(job_preferred_skills)

        # Bonus for assessment scores
        redrob_signals = candidate.get("redrob_signals", {})
        assessment_scores = redrob_signals.get("skill_assessment_scores", {})

        if assessment_scores:
            # Check for relevant skill assessments
            assessment_boost = 0.0
            for skill in job_core_skills:
                for skill_name, score in assessment_scores.items():
                    if skill in skill_name.lower():
                        assessment_boost += (score / 100) * 0.1
            skill_score = min(1.0, skill_score + assessment_boost)

        # Penalty for significant skill gaps
        if len(missing_core) >= 2:
            skill_score *= 0.9  # 10% penalty for multiple missing core skills

        breakdown.reasoning.append(
            f"Skill match: {len(core_matches)}/{len(job_core_skills)} core skills matched"
        )

        return skill_score

    def _calculate_experience_score(
        self, candidate: dict[str, Any], breakdown: ScoreBreakdown
    ) -> float:
        """
        Calculate experience relevance score (0-1).

        Considers:
        - Total years of experience vs requirements
        - Career progression quality
        - Industry relevance
        - Company size progression
        """
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])

        job_exp_range = self.job_requirements.get("experience_years", (5, 9))
        current_exp = profile.get("years_of_experience", 0)

        # Experience within range score
        min_exp, max_exp = job_exp_range
        if current_exp < min_exp:
            # Under-experienced but might still be good
            exp_score = 0.5 + (current_exp / min_exp) * 0.3
        elif current_exp > max_exp:
            # Over-experienced might be too senior
            exp_score = 0.8 + (max_exp / current_exp) * 0.2
        else:
            exp_score = 0.8 + (current_exp - min_exp) / (max_exp - min_exp) * 0.2

        # Check career progression
        if len(career_history) >= 2:
            # Look for upward progression
            titles = [c.get("title", "").lower() for c in career_history]
            has_progression = any(
                "senior" in t or "lead" in t or "staff" in t
                for t in titles[-2:]
            )
            if has_progression:
                exp_score += 0.1

        # Check industry relevance
        current_industry = profile.get("current_industry", "").lower()
        if any(ind in current_industry for ind in ["ai", "tech", "software", "product"]):
            exp_score += 0.05

        breakdown.reasoning.append(
            f"Experience: {current_exp} years (target: {min_exp}-{max_exp})"
        )

        return min(1.0, exp_score)

    def _calculate_behavioral_score(
        self, candidate: dict[str, Any], breakdown: ScoreBreakdown
    ) -> float:
        """
        Calculate behavioral signal score (0-1).

        This is a KEY differentiator - behavioral signals often predict
        hiring success better than static profile data.
        """
        redrob_signals = candidate.get("redrob_signals", {})

        # Critical signals for this role
        signals = {
            "response_rate": (redrob_signals.get("recruiter_response_rate", 0), 0.3),
            "last_active": (redrob_signals.get("last_active_date", ""), 0.2),
            "open_to_work": (redrob_signals.get("open_to_work_flag", False), 0.15),
            "application_activity": (redrob_signals.get("applications_submitted_30d", 0), 0.1),
            "profile_views": (redrob_signals.get("profile_views_received_30d", 0), 0.1),
            "saved_by_recruiters": (redrob_signals.get("saved_by_recruiters_30d", 0), 0.1),
            "interview_completion": (redrob_signals.get("interview_completion_rate", 0), 0.05),
        }

        # Calculate weighted score
        behavioral_score = 0.0
        signal_reasons = []

        # Response rate (high priority - indicates engagement)
        response_rate = signals["response_rate"][0]
        if response_rate > 0.7:
            behavioral_score += 0.95 * signals["response_rate"][1]
            signal_reasons.append("excellent response rate")
        elif response_rate > 0.4:
            behavioral_score += 0.7 * signals["response_rate"][1]
            signal_reasons.append("moderate response rate")
        elif response_rate < 0.2:
            behavioral_score += 0.3 * signals["response_rate"][1]
            signal_reasons.append("low response rate")
        else:
            behavioral_score += 0.5 * signals["response_rate"][1]

        # Last active date
        last_active = signals["last_active"][0]
        if last_active:
            # Simple check: was active in last 30 days?
            # In real system, would parse dates properly
            if "recent" in last_active.lower() or "today" in last_active.lower():
                behavioral_score += 1.0 * signals["last_active"][1]
                signal_reasons.append("currently active")
            elif "month" in last_active.lower():
                behavioral_score += 0.7 * signals["last_active"][1]
                signal_reasons.append("moderately active")
            else:
                behavioral_score += 0.4 * signals["last_active"][1]
                signal_reasons.append("inactive recently")
        else:
            behavioral_score += 0.3 * signals["last_active"][1]

        # Open to work
        if signals["open_to_work"][0]:
            behavioral_score += 1.0 * signals["open_to_work"][1]
            signal_reasons.append("marked as open to work")

        # Application activity
        apps_30d = signals["application_activity"][0]
        if apps_30d > 10:
            behavioral_score += 0.8 * signals["application_activity"][1]
            signal_reasons.append("high application activity")
        elif apps_30d > 0:
            behavioral_score += 0.5 * signals["application_activity"][1]

        # Profile views from recruiters
        views = signals["profile_views"][0]
        if views > 50:
            behavioral_score += 1.0 * signals["profile_views"][1]
            signal_reasons.append("high recruiter interest")
        elif views > 10:
            behavioral_score += 0.7 * signals["profile_views"][1]
        elif views == 0:
            signal_reasons.append("low recruiter visibility")

        # Saved by recruiters
        saved = signals["saved_by_recruiters"][0]
        if saved > 20:
            behavioral_score += 1.0 * signals["saved_by_recruiters"][1]
            signal_reasons.append("frequently saved by recruiters")
        elif saved > 5:
            behavioral_score += 0.7 * signals["saved_by_recruiters"][1]

        # Interview completion rate
        interview_rate = signals["interview_completion"][0]
        if interview_rate >= 0.9:
            behavioral_score += 1.0 * signals["interview_completion"][1]
            signal_reasons.append("excellent interview completion")
        elif interview_rate >= 0.7:
            behavioral_score += 0.8 * signals["interview_completion"][1]
        elif interview_rate < 0.5:
            behavioral_score += 0.4 * signals["interview_completion"][1]

        breakdown.reasoning.append(
            f"Behavioral signals: {', '.join(signal_reasons)} ({behavioral_score:.2f})"
        )

        return min(1.0, behavioral_score)

    def _calculate_cultural_fit_score(
        self, candidate: dict[str, Any], breakdown: ScoreBreakdown
    ) -> float:
        """
        Calculate cultural fit score (0-1).

        Based on signals that indicate fit with Redrob's culture:
        - Async communication preference
        - Product-engineering mindset
        - Comfort with ambiguity
        """
        redrob_signals = candidate.get("redrob_signals", {})
        profile = candidate.get("profile", {})

        cultural_score = 0.5  # Start neutral

        # Signal indicators
        signals_detected = []

        # GitHub activity (indicates technical passion, documentation)
        github_score = redrob_signals.get("github_activity_score", -1)
        if github_score > 70:
            cultural_score += 0.2
            signals_detected.append("strong GitHub presence")
        elif github_score > 0:
            cultural_score += 0.1
            signals_detected.append("some GitHub activity")

        # Connection count (networking ability)
        connections = redrob_signals.get("connection_count", 0)
        if connections > 500:
            cultural_score += 0.1
            signals_detected.append("strong network")
        elif connections > 100:
            cultural_score += 0.05

        # LinkedIn connected
        if redrob_signals.get("linkedin_connected", False):
            cultural_score += 0.05
            signals_detected.append("LinkedIn connected")

        # Profile completeness
        completeness = redrob_signals.get("profile_completeness_score", 0)
        if completeness > 80:
            cultural_score += 0.1
            signals_detected.append("complete profile")
        elif completeness > 50:
            cultural_score += 0.05

        # Verification signals
        verified_count = sum([
            redrob_signals.get("verified_email", False),
            redrob_signals.get("verified_phone", False),
        ])
        if verified_count >= 2:
            cultural_score += 0.1
            signals_detected.append("verified contact")

        # Role title alignment
        current_title = profile.get("current_title", "").lower()
        if any(t in current_title for t in ["ai", "ml", "engineer", "developer"]):
            cultural_score += 0.1
            signals_detected.append("technical title alignment")

        breakdown.reasoning.append(
            f"Cultural fit signals: {', '.join(signals_detected) if signals_detected else 'baseline'}"
        )

        return min(1.0, cultural_score)

    def _calculate_availability_score(
        self, candidate: dict[str, Any], breakdown: ScoreBreakdown
    ) -> float:
        """
        Calculate availability score (0-1).

        Based on factors that affect how quickly a candidate can join.
        """
        redrob_signals = candidate.get("redrob_signals", {})
        profile = candidate.get("profile", {})

        availability_score = 0.5  # Start neutral

        # Notice period
        notice_period = redrob_signals.get("notice_period_days", 60)
        if notice_period <= 15:
            availability_score += 0.3
            breakdown.reasoning.append("notice period: instant/15 days")
        elif notice_period <= 30:
            availability_score += 0.25
            breakdown.reasoning.append(f"notice period: {notice_period} days")
        elif notice_period <= 60:
            availability_score += 0.1
            breakdown.reasoning.append(f"notice period: {notice_period} days")
        else:
            availability_score += 0.05
            breakdown.reasoning.append(f"notice period: {notice_period} days (long)")

        # Willing to relocate
        if redrob_signals.get("willing_to_relocate", False):
            availability_score += 0.2
            breakdown.reasoning.append("willing to relocate")

        # Preferred work mode matches
        job_work_mode = self.job_requirements.get("work_mode", "flexible").lower()
        candidate_work_mode = redrob_signals.get("preferred_work_mode", "").lower()

        if job_work_mode == "hybrid" and candidate_work_mode in ["hybrid", "flexible"]:
            availability_score += 0.15
        elif job_work_mode == "remote" and candidate_work_mode in ["remote", "flexible"]:
            availability_score += 0.15
        elif job_work_mode == "onsite" and candidate_work_mode in ["onsite", "hybrid"]:
            availability_score += 0.15

        # Expected salary alignment (simplified)
        salary_range = redrob_signals.get("expected_salary_range_inr_lpa", {})
        # Don't penalize for salary - just note it in reasoning

        return min(1.0, availability_score)


def load_candidates(file_path: str) -> list[dict[str, Any]]:
    """Load candidates from JSONL file."""
    candidates = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


if __name__ == "__main__":
    # Example usage
    sample_candidate = {
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
        },
    }

    # Mock job requirements
    job_reqs = {
        "core_skills": ["Python", "Machine Learning", "Embeddings"],
        "experience_years": (5, 9),
        "work_mode": "hybrid",
    }

    scorer = CandidateScorer(job_reqs)
    result = scorer.score_candidate(sample_candidate)

    print(f"Candidate: {result.candidate_id}")
    print(f"Total Score: {result.total_score:.3f}")
    print(f"Skill Match: {result.skill_match:.3f}")
    print(f"Experience: {result.experience_score:.3f}")
    print(f"Behavioral: {result.behavioral_score:.3f}")
    print(f"Reasoning: {result.reasoning}")
