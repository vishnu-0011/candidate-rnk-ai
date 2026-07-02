"""
Redrobe AI Candidate Ranking System

An AI-powered candidate ranking system that goes beyond keyword matching
to understand what truly makes a candidate a good fit for a role.
"""

__version__ = "1.0.0"
__author__ = "Redrobe AI Challenge"

from .job_parser import JobDescriptionParser
from .scorer import CandidateScorer
from .ranking import CandidateRanker

__all__ = ["JobDescriptionParser", "CandidateScorer", "CandidateRanker"]
