# Redrobe AI Candidate Ranking System

An AI-powered candidate ranking system that goes beyond keyword matching to understand what truly makes a candidate a good fit for a role.

## Problem Statement

Recruiters go through hundreds of profiles and still often miss the right person. Not because the talent isn't there — but because keyword filters can't see what actually matters.

Most ranking systems use simple keyword matching:
- ✗ Counts how many "AI" skills are listed
- ✗ Looks for exact title matches
- ✗ Ignores behavioral signals entirely

**Our approach:** Understand the *meaning* behind the requirements and evaluate candidates holistically.

## Solution

This system ranks candidates the way a great recruiter would — by understanding:

- **What the role truly needs** (beyond the keyword list)
- **Who can actually deliver** (behavioral signals, engagement)
- **Who will fit** (cultural alignment, availability)

## Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Understanding** | Parses job descriptions to extract core requirements vs nice-to-haves |
| **Behavioral Signal Analysis** | Weights recruiter response rates, last active dates, engagement metrics |
| **Experience Relevance** | Considers career progression and industry fit, not just years |
| **Cultural Fit Scoring** | Evaluates async communication readiness and product mindset |
| **Availability Assessment** | Factors in notice period, work mode, and relocation willingness |
| **Explainable Rankings** | Each candidate gets detailed reasoning for their score |

## Architecture

```
redrobe-ranking/
├── src/
│   ├── __init__.py
│   ├── job_parser.py      # Semantic job description understanding
│   ├── scorer.py          # Multi-dimensional candidate scoring
│   ├── ranking.py         # Final ranking with reasoning
│   └── utils.py           # Helper functions
├── data/
│   ├── job_description.txt    # Parsed job requirements
│   └── candidates.jsonl       # Input candidate data
├── output/
│   └── ranked_candidates.csv  # Final ranked output
├── config.py              # Configuration and scoring weights
├── main.py                # Entry point
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
CAND_0004989,1,0.9920,"HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76"
CAND_0001195,2,0.9840,"HR Manager with 8.7 yrs; 9 AI core skills; response rate 0.20"
CAND_0003114,3,0.9760,"ML Engineer with 6.4 yrs; 4 AI core skills; response rate 0.88"
```

## Scoring Breakdown

Each candidate is scored on five dimensions:

1. **Skill Match (35%)** - Core skills, proficiency, assessment scores
2. **Experience Relevance (20%)** - Years of experience, career progression, industry
3. **Behavioral Signals (25%)** - Response rate, active status, recruitment engagement
4. **Cultural Fit (10%)** - GitHub activity, profile completeness, connections
5. **Availability (10%)** - Notice period, work mode, relocation willingness

## Behavioral Signals Used

The system heavily weights these behavioral indicators:

| Signal | Why It Matters |
|--------|----------------|
| `recruiter_response_rate` | Indicates genuine engagement and communication skills |
| `last_active_date` | Shows current job market activity |
| `open_to_work_flag` | Explicit availability signal |
| `saved_by_recruiters_30d` | Demonstrates recruiter interest |
| `profile_views_received_30d` | Shows search visibility |
| `interview_completion_rate` | Reliability indicator |
| `github_activity_score` | Technical passion and documentation habits |

## Project Structure

### src/job_parser.py
Parses job descriptions to extract:
- Role title and experience range
- Core vs preferred skills
- Location and work mode
- Red flags (consulting background, pure research)
- Cultural signals (async, product mindset)

### src/scorer.py
Multi-dimensional scoring engine that evaluates:
- Skill match with weighted importance
- Experience relevance and progression
- Behavioral signal aggregation
- Cultural fit indicators
- Availability constraints

### src/ranking.py
Orchestrates the pipeline:
- Loads candidates from JSONL
- Scores all candidates
- Generates ranked output with reasoning
- Exports to CSV format

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## License

MIT License
