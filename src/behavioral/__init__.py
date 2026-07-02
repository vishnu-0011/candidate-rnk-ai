"""
Behavioral Signals Module for Redrobe Candidate Ranking System.

Uses LightGBM to learn how to weight 23 redrob behavioral signals.

Key Insight: Behavioral signals often predict hiring success better
than static profile data. A perfect-on-paper candidate who hasn't
logged in for 6 months is not actually available.

LightGBM was chosen over XGBoost because:
1. Faster training (critical for synthetic data generation)
2. Better handling of sparse features
3. Often superior performance on tabular data
"""

from typing import Dict, List, Any
import numpy as np


class BehavioralScorer:
    """
    Score candidates based on behavioral signals using LightGBM.

    The model is trained on synthetic data to learn optimal weights
    for each signal, then applied to real candidates.
    """

    def __init__(self):
        """
        Initialize behavioral scorer.
        """
        self.model = None
        self.trained = False

        # Redrob signals to use
        self.feature_names = [
            "recruiter_response_rate",
            "profile_completeness_score",
            "last_active_days_ago",
            "open_to_work_flag",
            "applications_submitted_30d",
            "profile_views_received_30d",
            "connection_count",
            "endorsements_received",
            "notice_period_days",
            "interview_completion_rate",
            "offer_acceptance_rate",
            "saved_by_recruiters_30d",
            "search_appearance_30d",
            "avg_response_time_hours",
            "github_activity_score",
            "verified_email",
            "verified_phone",
            "linkedin_connected",
            "willing_to_relocate",
            "expected_salary_min",
            "expected_salary_max",
            "years_since_last_job_change",
            "skill_assessment_avg",
        ]

    def _extract_features(self, candidate: Dict[str, Any]) -> np.ndarray:
        """
        Extract numeric features from candidate profile.

        Args:
            candidate: Candidate profile dictionary

        Returns:
            numpy array of features
        """
        signals = candidate.get("redrob_signals", {})
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])

        features = []

        # Response rate (0-1)
        response_rate = signals.get("recruiter_response_rate", 0)
        features.append(response_rate)

        # Profile completeness (0-100)
        completeness = signals.get("profile_completeness_score", 0) / 100.0
        features.append(completeness)

        # Days since last active (simplified)
        last_active = signals.get("last_active_date", "")
        if last_active:
            # In real system, parse date properly
            # For now, assume recent if not empty
            last_active_days = 0 if last_active else 180
        else:
            last_active_days = 180
        features.append(last_active_days / 180.0)

        # Open to work
        open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0
        features.append(open_to_work)

        # Application activity
        apps_30d = signals.get("applications_submitted_30d", 0)
        features.append(min(apps_30d / 10.0, 1.0))

        # Profile views from recruiters
        views = signals.get("profile_views_received_30d", 0)
        features.append(min(views / 100.0, 1.0))

        # Connection count
        connections = signals.get("connection_count", 0)
        features.append(min(np.log1p(connections) / 7.0, 1.0))

        # Endorsements
        endorsements = signals.get("endorsements_received", 0)
        features.append(min(np.log1p(endorsements) / 6.0, 1.0))

        # Notice period (lower is better)
        notice = signals.get("notice_period_days", 60)
        features.append(1.0 - min(notice / 60.0, 1.0))

        # Interview completion rate
        interview_rate = signals.get("interview_completion_rate", 0.5)
        features.append(interview_rate)

        # Offer acceptance rate
        offer_rate = signals.get("offer_acceptance_rate", 0.5)
        features.append(offer_rate if offer_rate >= 0 else 0.5)

        # Saved by recruiters
        saved = signals.get("saved_by_recruiters_30d", 0)
        features.append(min(saved / 20.0, 1.0))

        # Search appearances
        search_appearances = signals.get("search_appearance_30d", 0)
        features.append(min(search_appearances / 50.0, 1.0))

        # Average response time (lower is better)
        response_time = signals.get("avg_response_time_hours", 24)
        features.append(1.0 - min(response_time / 72.0, 1.0))

        # GitHub activity
        github = signals.get("github_activity_score", 0)
        features.append(github / 100.0 if github >= 0 else 0.2)

        # Verification flags
        verified = sum([
            signals.get("verified_email", False),
            signals.get("verified_phone", False),
            signals.get("linkedin_connected", False),
        ])
        features.append(verified / 3.0)

        # Willing to relocate
        relocate = 1.0 if signals.get("willing_to_relocate", False) else 0.0
        features.append(relocate)

        # Salary expectations
        salary_range = signals.get("expected_salary_range_inr_lpa", {})
        expected_min = salary_range.get("min", 20)
        expected_max = salary_range.get("max", 50)
        features.append(min(expected_min / 100.0, 1.0))
        features.append(min(expected_max / 200.0, 1.0))

        # Years since last job change
        if career:
            recent_job = career[0] if career else {}
            # Simplified calculation
            years_since_change = 0.5
        else:
            years_since_change = 1.0
        features.append(years_since_change)

        # Average skill assessment score
        assessments = signals.get("skill_assessment_scores", {})
        if assessments:
            avg_assessment = np.mean(list(assessments.values())) / 100.0
        else:
            avg_assessment = 0.5
        features.append(avg_assessment)

        return np.array(features)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the behavioral scorer on data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target scores (n_samples,)
        """
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(X, y)
            self.trained = True
        except ImportError:
            # Fallback: simple weighted sum if LightGBM not installed
            print("LightGBM not installed, using simple weighted fallback")
            self._simple_weights = self._learn_simple_weights(X, y)

    def _learn_simple_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Learn simple weights when LightGBM unavailable.
        """
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        weights = np.abs(lr.coef_)
        weights = weights / weights.sum()
        return weights

    def predict(self, candidate: Dict[str, Any]) -> float:
        """
        Predict behavioral score for a candidate.

        Args:
            candidate: Candidate profile dictionary

        Returns:
            Behavioral score (0-1)
        """
        features = self._extract_features(candidate).reshape(1, -1)

        if self.trained and self.model is not None:
            score = self.model.predict(features)[0]
        else:
            # Use learned weights
            score = np.dot(features[0], self._simple_weights)

        return float(np.clip(score, 0.0, 1.0))


def generate_synthetic_behavioral_data(
    n_samples: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic behavioral data for training.

    Creates realistic candidate profiles with known "quality" scores.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (feature_matrix, target_scores)
    """
    np.random.seed(42)
    n_features = 23

    # Generate feature ranges
    X = np.random.rand(n_samples, n_features)

    # Apply realistic constraints
    X[:, 0] = np.clip(X[:, 0], 0.1, 1.0)  # Response rate
    X[:, 1] = X[:, 1] * 100  # Profile completeness
    X[:, 2] = np.random.randint(1, 180, n_samples)  # Days since active
    X[:, 3] = np.random.randint(0, 2, n_samples)  # Open to work
    X[:, 4] = np.random.randint(0, 20, n_samples)  # Apps submitted
    X[:, 5] = np.random.randint(0, 150, n_samples)  # Profile views
    X[:, 6] = np.random.randint(10, 2000, n_samples)  # Connections
    X[:, 7] = np.random.randint(0, 100, n_samples)  # Endorsements
    X[:, 8] = np.random.randint(0, 120, n_samples)  # Notice period
    X[:, 9] = np.clip(X[:, 9], 0.3, 1.0)  # Interview completion
    X[:, 10] = np.clip(X[:, 10], -1, 1)  # Offer acceptance
    X[:, 11] = np.random.randint(0, 50, n_samples)  # Saved by recruiters
    X[:, 12] = np.random.randint(0, 100, n_samples)  # Search appearances
    X[:, 13] = np.random.randint(1, 168, n_samples)  # Response time
    X[:, 14] = np.random.randint(-1, 100, n_samples)  # GitHub score
    X[:, 15:18] = np.random.randint(0, 2, (n_samples, 3))  # Verifications
    X[:, 18] = np.random.randint(0, 2, n_samples)  # Relocate
    X[:, 19] = np.random.randint(10, 100, n_samples)  # Salary min
    X[:, 20] = np.random.randint(20, 200, n_samples)  # Salary max
    X[:, 21] = np.clip(X[:, 21], 0, 1)  # Years since change
    X[:, 22] = np.clip(X[:, 22], 0, 1)  # Assessment avg

    # Generate target scores based on feature combinations
    # This simulates "what makes a good candidate"
    y = (
        0.25 * X[:, 0] +  # Response rate
        0.15 * X[:, 1] / 100 +  # Completeness
        0.15 * (1 - X[:, 2] / 180) +  # Recent activity
        0.10 * X[:, 3] +  # Open to work
        0.08 * np.clip(X[:, 4] / 10, 0, 1) +  # Activity
        0.07 * np.clip(X[:, 5] / 100, 0, 1) +  # Visibility
        0.06 * np.clip(X[:, 6] / 500, 0, 1) +  # Network
        0.05 * X[:, 9] +  # Interview completion
        0.04 * (1 - X[:, 8] / 60) +  # Notice period
        0.03 * np.clip(X[:, 11] / 20, 0, 1) +  # Saved by recruiters
        0.03 * np.clip(X[:, 14] / 100, 0, 1) +  # GitHub
        0.02 * np.sum(X[:, 15:18], axis=1) / 3 +  # Verification
        0.02 * X[:, 18] +  # Relocate
        0.02 * np.clip(X[:, 19] / 50, 0, 1)  # Salary match
    )

    # Add noise
    y = y + np.random.normal(0, 0.05, n_samples)
    y = np.clip(y, 0.0, 1.0)

    return X, y


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic behavioral data...")
    X, y = generate_synthetic_behavioral_data(100)

    print(f"Generated {len(X)} samples with {len(X[0])} features")
    print(f"Score range: [{y.min():.3f}, {y.max():.3f}]")
