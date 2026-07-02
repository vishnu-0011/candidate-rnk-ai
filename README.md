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
│  (Groq Mistral-70B for efficient reasoning)                 │
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

# Run with sample data
python main.py sample_job_description.txt data/candidates.jsonl output/ranked.csv

# Or specify providers
python main.py job.txt candidates.jsonl output.csv --provider groq
```

## Output Example

```csv
candidate_id,rank,score,reasoning
CAND_0004989,1,0.9920,"Senior AI Engineer with 6.1 yrs; 9 AI core skills; response rate 0.76.
                      LLM verified: strong engagement signals align with role's async-first culture"
CAND_0001195,2,0.9840,"Senior AI Engineer with 8.7 yrs; 9 AI core skills; response rate 0.20.
                      Behavioral penalty: low recruiter response rate suggests lower engagement"
```

## Key Components

### 1. LLM Integration (`src/llm/`)
- **GroqClient**: Fast Mistral-7B/70B for parsing and batch processing
- **OpenAIClient**: High-quality GPT-4 for explanations and re-ranking
- **Prompt Templates**: 8 structured prompts with few-shot examples

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
- Full pipeline orchestration
- LLM re-ranking with detailed explanations
- CSV output generation

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| LLM Parsing | Groq + Mistral-7B | 50-100ms latency, cost-effective |
| Explanations | Groq + Mistral-70B or OpenAI GPT-4 | Best reasoning quality |
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
│   │   └── hybrid_search.py   # BM25 + dense
│   ├── behavioral/             # Signal fusion
│   │   ├── __init__.py
│   │   └── scorer.py          # LightGBM model
│   ├── ranking/                # Final scoring
│   │   ├── __init__.py
│   │   └── engine.py          # Pipeline orchestration
│   ├── job_parser.py           # LLM job parsing
│   ├── scorer.py               # Multi-dimensional
│   └── ranking.py              # Full pipeline
├── data/                       # Input data
├── output/                     # Generated rankings
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Candidate processing time | ~2-5 seconds per candidate |
| Total 1000 candidates | ~3-8 minutes |
| Vector search latency | <50ms per query |
| Embedding dimension | 1024 (bge-large) |
| Model size | ~500MB (bge-large-en-v1.5) |

## License

MIT License
