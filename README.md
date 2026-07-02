# Redrobe AI Candidate Ranking System

An AI-powered candidate ranking system that goes beyond keyword matching to understand what truly makes a candidate a good fit for a role.

## Problem Statement

Recruiters go through hundreds of profiles and still often miss the right person. Not because the talent isn't there — but because keyword filters can't see what actually matters.

## Solution

This system ranks candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role through:

- **Semantic Understanding**: LLM-based job description analysis
- **Multi-dimensional Scoring**: Skills, experience, behavioral signals, cultural fit
- **Role-Aware Weighting**: Different factors weighted based on role requirements
- **Explainable Rankings**: Detailed reasoning for each score

## Architecture

```
redrobe-ranking/
├── src/
│   ├── __init__.py
│   ├── job_parser.py          # Semantic job description understanding
│   ├── scorer.py              # Multi-dimensional candidate scoring
│   ├── ranking.py             # Final ranking with reasoning
│   └── utils.py               # Helper functions
├── data/
│   ├── job_description.txt    # Parsed job requirements
│   └── candidates.jsonl       # Input candidate data
├── output/
│   └── ranked_candidates.csv  # Final ranked output
├── config.py                  # Configuration and scoring weights
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd redrobe-ranking

# Install dependencies
pip install -r requirements.txt

# Run the ranking system
python main.py
```

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0001234,1,0.965,"Senior AI Engineer with 7.5 years experience, strong alignment with role requirements, excellent behavioral signals"
CAND_0005678,2,0.942,"Machine Learning Engineer with 6.2 years experience, good technical match"
```

## Key Features

- **Behavioral Signal Analysis**: Weights recruiter response rates, last active dates, and engagement metrics
- **Experience Relevance**: Considers career progression and industry fit, not just years
- **Cultural Fit Scoring**: Evaluates async communication readiness and product mindset
- **Availability Assessment**: Factors in notice period and work mode preferences

## Files

- `src/job_parser.py` - Parses job descriptions to extract requirements
- `src/scorer.py` - Multi-dimensional candidate scoring engine
- `src/ranking.py` - Generates final ranked list with reasoning
- `main.py` - Main entry point for running the system

## License

MIT License
