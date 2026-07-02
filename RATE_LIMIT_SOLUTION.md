# Redrobe AI Candidate Ranking System

## Rate Limit Solution

The Groq API has a daily token limit (100,000 TPD). The job parsing alone uses ~4,000 tokens per call.

### Options:

**Option 1: Use --provider openai** (if you have OpenAI API key)
```bash
python main.py data/job_description.docx data/candidates.jsonl output/ranked.csv --provider openai
```

**Option 2: Wait for rate limit to reset** (~30 minutes based on error message)

**Option 3: Process with --no-explanations to reduce LLM usage**
```bash
python main.py data/job_description.docx data/candidates.jsonl output/ranked.csv --no-explanations
```

**Option 4: Use smaller test dataset first**
```bash
python main.py data/job_description.docx data/test_candidates.jsonl output/test_ranked.csv
```

---

## About the Token Usage

The system makes LLM calls for:
1. **Job parsing** (~4,000-6,000 tokens) - Extract requirements from job description
2. **Candidate scoring** (~1,000-2,000 per candidate) - Score each candidate
3. **Re-ranking** (~500-1,000 tokens) - Re-rank top candidates with explanations

For the full dataset with hundreds of candidates, this quickly exceeds Groq's limits.
