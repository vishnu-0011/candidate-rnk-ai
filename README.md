# Redrobe AI Candidate Ranking System

> **"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."**

An AI-powered candidate ranking system using **LLM semantic understanding**, **dual-vector hybrid search**, and **behavioral signal fusion**.

## The "Aha!" Architecture

### Why Standard Approaches Fail

| Standard Approach | Problem | Our Solution |
|-------------------|---------|--------------|
| Keyword matching | Misses semantic equivalents ("RAG" ≠ "search system") | **Dual-vector hybrid search** |
| Simple scoring | Ignores engagement signals | **LightGBM behavioral fusion** |
| Rule-based | Hardcoded weights don't adapt | **LLM-weighted hybrid** |
| No explanations | Black box rankings | **LLM-generated explanations** |

### The 5-Layer Scoring Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    FINAL RANKED LIST                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM Re-ranker: Deep understanding + Explanations           │
│  (Groq Mistral-70B or OpenAI GPT-4)                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Hybrid Score = 0.35*Semantic + 0.25*Behavioral + ...       │
│  (Learned fusion weights from synthetic training data)      │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Behavioral Fusion: LightGBM on 23 redrob signals           │
│  (Response rate, active dates, interview completion, etc.)  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Dual-Vector Search: BM25 + BGE-large-en-v1.5 embeddings    │
│  (Find both exact matches AND semantic equivalents)         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM Job Parser: Extract requirements with nuance           │
│  (Understand what "5-9 years" really means)                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd redrobe-ranking

# Install dependencies
pip install -r requirements.txt

# Run with the challenge data files
python main.py data/job_description.docx data/candidates.jsonl output/ranked.csv

# Or for testing with smaller data
python main.py data/job_description.docx data/test_candidates.jsonl output/test_ranked.csv

# Advanced usage
python main.py data/job_description.docx data/candidates.jsonl output/ranked.csv --provider groq --no-explanations
```

## Usage Guide

### Command Format

```
python main.py <job_file> <candidates_file> [output_file] [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `job_file` | Path to job description (.txt or .docx) |
| `candidates_file` | Path to candidates JSONL file |
| `output_file` | Path for output CSV (default: output/ranked_candidates.csv) |

### Options

| Flag | Description |
|------|-------------|
| `--provider {groq,openai}` | LLM provider (default: groq) |
| `--top-n N` | Limit to top N candidates |
| `--no-explanations` | Skip LLM explanations (faster, fewer tokens) |

### Examples

```bash
# Basic run with Groq (fast, cost-effective)
python main.py data/job_description.docx data/candidates.jsonl

# Use OpenAI for higher quality results
python main.py data/job_description.docx data/candidates.jsonl --provider openai

# Only rank top 10 candidates
python main.py data/job_description.docx data/candidates.jsonl --top-n 10

# Skip LLM re-ranking to save tokens (uses rule-based reasoning)
python main.py data/job_description.docx data/candidates.jsonl --no-explanations
```

## Key Components

### 1. LLM Integration (`src/llm/`)
- **GroqClient**: Fast Mistral-based LLM for parsing and batch processing
- **OpenAIClient**: High-quality GPT-4 for explanations and re-ranking
- **Prompt Templates**: 8 structured prompts with few-shot examples for:
  - Job parsing with semantic understanding
  - Candidate evaluation across dimensions
  - Explanations generation
  - Re-ranking with evidence
  - Behavioral signal analysis

### 2. Dual-Vector Hybrid Search (`src/embeddings/`)
- **BGEEncoder**: BAAI/bge-large-en-v1.5 for semantic embeddings
- **BM25Index**: Sparse keyword indexing for exact matches
- **HybridSearcher**: Combines both with learned fusion weights

### 3. Behavioral Scoring (`src/behavioral/`)
- **LightGBM Model**: Learns optimal signal weights from synthetic data
- **23 Redrob Signals**: Response rate, active dates, interview completion, etc.

### 4. Multi-Dimensional Scorer (`src/scorer.py`)
- Skill match (semantic + proficiency)
- Behavioral engagement
- Experience relevance
- Cultural alignment
- Availability assessment

### 5. Ranking Engine (`src/ranking.py`)
- LLM job parsing with semantic understanding
- Full pipeline orchestration
- LLM re-ranking with detailed explanations
- CSV output generation

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| LLM Parsing | Groq + Mistral or OpenAI GPT-4 | Fast inference or high-quality reasoning |
| Embeddings | BAAI/bge-large-en-v1.5 | SOTA sentence embeddings |
| Hybrid Search | BM25 + Cosine | Best of sparse + dense |
| Behavioral Model | LightGBM Ranker | Fast, handles sparse features |
| Vector DB | In-memory Qdrant | Lightweight, fast |

## Why This Approach is Unique

| Feature | Standard System | This System |
|---------|-----------------|-------------|
| Skill Matching | Keyword exact match | Dual-vector semantic + sparse |
| Behavioral Signals | Manual weighting | LightGBM learned weights |
| Job Understanding | Regex patterns | LLM with few-shot examples |
| Final Ranking | Single model | LLM re-ranker with evidence |
| Explanations | None or basic | LLM-generated detailed reasoning |
| Training Data | None / small | Synthetic augmentation (1000+ samples) |

## Data Files

Place your data files in the `data/` folder:

- `job_description.docx` or `job_description.txt` - Job description to parse
- `candidates.jsonl` - Candidates in JSONL format (one per line)

For testing with the challenge data:
- `data/job_description.docx` - Parsed job description from the challenge
- `data/candidates.jsonl` - Full candidate dataset
- `data/test_candidates.jsonl` - First 10 candidates for quick testing

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0004989,1,0.9920,"Senior AI Engineer with 6.1 yrs; 9 AI core skills; response rate 0.76. LLM verified: strong engagement signals align with role's async-first culture"
CAND_0001195,2,0.9840,"Senior AI Engineer with 8.7 yrs; 9 AI core skills; response rate 0.20. Behavioral penalty: low recruiter response rate suggests lower engagement"
```

## Environment Setup

### API Keys

Create a `.env` file in the project root with:

```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Get API keys:
- Groq: https://console.groq.com/
- OpenAI: https://platform.openai.com/api-keys

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `groq` or `openai` - LLM API clients (one required)
- `sentence-transformers` - BGE embeddings
- `lightgbm` - Behavioral scoring model
- `python-dotenv` - Environment variable management

## Rate Limiting & Token Management

The Groq API has daily limits (100K TPD). To stay within limits:

```bash
# Use --no-explanations to skip LLM re-ranking
python main.py data/job_description.docx data/candidates.jsonl --no-explanations

# Use OpenAI instead of Groq
python main.py data/job_description.docx data/candidates.jsonl --provider openai

# Process only top candidates
python main.py data/job_description.docx data/candidates.jsonl --top-n 50
```

## Project Structure

```
redrobe-ranking/
├── src/
│   ├── llm/                    # LLM integration
│   │   ├── __init__.py
│   │   ├── client.py          # Groq/OpenAI clients
│   │   └── prompt_templates.py # Few-shot prompts
│   ├── embeddings/             # Vector system
│   │   ├── __init__.py
│   │   ├── encoder.py         # BGE encoder
│   │   ├── bm25_index.py      # BM25 sparse indexing
│   │   └── hybrid_searcher.py # BM25 + dense combo
│   ├── behavioral/             # Signal fusion
│   │   ├── __init__.py
│   │   └── scorer.py          # LightGBM model
│   ├── job_parser.py           # LLM job parsing
│   ├── scorer.py               # Multi-dimensional
│   └── ranking.py              # Full pipeline
├── data/                       # Input data
├── output/                     # Generated rankings
├── config.py
├── main.py
├── requirements.txt
├── .env (create with API keys)
└── README.md
```

## License

MIT License
