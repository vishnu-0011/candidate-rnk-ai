# Redrobe AI Candidate Ranking System

An AI-powered candidate ranking system that goes beyond keyword matching to understand what truly makes a candidate a good fit for a role.

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
│   └── candidates.json        # Input candidate data
├── output/
│   └── ranked_candidates.csv  # Final ranked output
├── config.py                  # Configuration and scoring weights
├── main.py                    # Entry point
└── requirements.txt
```

## Key Features

- **Semantic Understanding**: LLM-based job description analysis to extract true requirements
- **Multi-dimensional Scoring**: Combines skills, experience, behavioral signals, and cultural fit
- **Role-Aware Weighting**: Different factors weighted differently based on role requirements
- **Explainable Rankings**: Each candidate gets detailed reasoning for their score

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0001234,1,0.965,Senior AI Engineer with 7.5 years experience, strong alignment with role requirements...
```
