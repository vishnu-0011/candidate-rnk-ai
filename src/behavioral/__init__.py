"""
Behavioral Signals Module for Redrobe AI Candidate Ranking System.

Uses LightGBM to learn how to weight 23 redrob behavioral signals.

Key Insight: Behavioral signals often predict hiring success better
than static profile data. A perfect-on-paper candidate who hasn't
logged in for 6 months is not actually available.

Why LightGBM?
- Faster training than XGBoost
- Better handling of sparse features
- Superior performance on tabular data
- Handles the 23 behavioral signals effectively
"""

from .scorer import BehavioralScorer, generate_synthetic_behavioral_data

__all__ = ["BehavioralScorer", "generate_synthetic_behavioral_data"]
