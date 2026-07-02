# Redrobe AI Candidate Ranking System

A Python-based AI system that ranks candidates beyond keyword matching by understanding semantic requirements, behavioral signals, and cultural fit.

## Project Structure

```
redrobe-ranking/
├── src/                # Core implementation
│   ├── __init__.py
│   ├── job_parser.py   # Job description parsing
│   ├── scorer.py       # Multi-dimensional scoring
│   ├── ranking.py      # Final ranking generation
│   └── utils.py        # Utility functions
├── data/               # Input data (job description, candidates)
├── output/             # Generated rankings
├── config.py           # Configuration
├── main.py             # Entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Output

Ranked candidates CSV with scores and reasoning for each selection.
