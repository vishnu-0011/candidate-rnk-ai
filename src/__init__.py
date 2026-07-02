# Redrobe AI Candidate Ranking System

"""
An AI-powered candidate ranking system that goes beyond keyword matching
to understand what truly makes a candidate a good fit for a role.

Architecture:
- LLM Integration: Groq (Mistral) + OpenAI (GPT-4) for parsing and explanations
- Embeddings: BAAI/bge-large-en-v1.5 for semantic search
- Hybrid Search: BM25 + Dense vectors with learned fusion
- Behavioral Scoring: LightGBM for signal weighting
- Multi-dimensional Scoring: Skill, Experience, Behavioral, Cultural, Availability
"""

__version__ = "1.0.0"
__author__ = "Redrobe AI Challenge"
