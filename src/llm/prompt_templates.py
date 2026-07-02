"""
Prompt Templates for LLM Integration.

Defines structured prompts for:
- Job description parsing
- Candidate evaluation
- Explanations generation
- Re-ranking

Uses few-shot examples to guide the LLM toward high-quality outputs.
"""

from typing import Dict


PROMPT_TEMPLATES: Dict[str, str] = {
    # Job parsing templates
    "parse_job_description": """You are an expert recruiter and talent analyst. Parse the following job description to extract structured information.

Output as JSON with this structure:
{
    "role_title": "string",
    "experience_years": {"min": int, "max": int},
    "location": "string",
    "work_mode": "remote|hybrid|onsite|flexible",
    "core_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "must_haves": ["requirement1", "requirement2"],
    "red_flags": ["disqualifier1", "disqualifier2"],
    "cultural_signals": ["signal1", "signal2"],
    "role_level": "senior|staff|principal"
}

Job Description:
{job_description}

IMPORTANT: Extract only what's stated or strongly implied. Don't hallucinate requirements.""",

    "parse_job_description_fewshot_input": """
Job Description:
Senior AI Engineer — Founding Team
Company: Redrob AI
Location: Pune/Noida, India (Hybrid — flexible cadence)
Experience Required: 5–9 years

Required Skills:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5)
- Production experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus)
- Strong Python
- Hands-on experience designing evaluation frameworks (NDCG, MRR, MAP, A/B testing)

Red Flags:
- Pure research without production deployment
- AI experience only from recent (under 12 months) LangChain projects
- Senior engineer who hasn't written production code in 18 months
- Consulting firm background (TCS,Infosys,Wipro,Accenture,Cognizant,Capgemini)

Job Description:
""",

    "parse_job_description_fewshot_output": """
{
    "role_title": "Senior AI Engineer",
    "experience_years": {"min": 5, "max": 9},
    "location": "Pune/Noida, India",
    "work_mode": "hybrid",
    "core_skills": ["Embeddings", "Vector Databases", "Python", "Evaluation Frameworks"],
    "preferred_skills": ["RAG", "Retrieval", "Ranking", "LLMs"],
    "must_haves": ["Production embeddings experience", "Vector database experience", "Strong Python", "Evaluation framework design"],
    "red_flags": ["Pure research without production", "Recent LLM-only experience", "No production code in 18 months", "Consulting background"],
    "cultural_signals": ["Async-first communication", "Product-engineering mindset", "Comfort with ambiguity"],
    "role_level": "senior"
}
""",

    # Candidate evaluation templates
    "evaluate_candidate": """Evaluate this candidate for the {role_title} role at {company}.

Job Requirements:
{job_requirements}

Candidate Profile:
{candidate_profile}

Behavioral Signals:
{behavioral_signals}

Rate on these dimensions (0-1 scale):
1. Skill Match: How well do their skills match core requirements?
2. Experience Relevance: Is their experience relevant and progressive?
3. Behavioral Fit: Based on their engagement signals
4. Cultural Alignment: Does this match our async-first, product-driven culture?

Output as JSON:
{
    "skill_match": 0.8,
    "experience_relevance": 0.75,
    "behavioral_fit": 0.9,
    "cultural_alignment": 0.85,
    "overall_score": 0.84,
    "key_strengths": ["strength1", "strength2"],
    "concerns": ["concern1"],
    "red_flag_check": ["verified"]
}
""",

    # Explanation generation
    "generate_explanation": """Generate a human-readable explanation for why candidate {candidate_id} is ranked {rank} for this {role_title} role.

Score: {score}
Dimension Breakdown:
- Skill Match: {skill_match}
- Experience: {experience}
- Behavioral: {behavioral}
- Cultural: {cultural}
- Availability: {availability}

Candidate: {candidate_identity}
Role: {job_title}

The explanation should:
1. Start with a strong summary of why they fit
2. Highlight 2-3 key strengths with evidence
3. Note any concerns or mitigating factors
4. Be concise (under 200 words)
5. Sound like a real recruiter wrote it

Output as JSON:
{
    "explanation": "string"
}
""",

    # Re-ranking
    "rerank_candidates": """Re-rank these candidates for a {role_title} role based on deeper understanding.

Initial scores were computed using keyword matching and basic signals. Now apply deeper analysis:

Candidates:
{candidates_json}

For each candidate, consider:
1. Depth of experience (not just years)
2. Quality of skill application (projects vs. just listing)
3. Behavioral signal coherence
4. Cultural fit indicators
5. True availability (notice period, location)

Output as JSON with re-ranked candidates:
{
    "ranked_candidates": [
        {"candidate_id": "CAND_0001234", "new_rank": 1, "score": 0.95, "reason": "explanation"},
        ...
    ]
}
""",

    # Synthetic data generation
    "generate_synthetic_job": """Generate a realistic job description for a {role_type} role.

Requirements:
- Company: {company_type}
- Location: {location}
- Experience: {experience_range} years
- Must include: {required_skill}
- Nice to have: {preferred_skill}

Return as JSON:
{
    "company": "string",
    "role_title": "string",
    "description": "string",
    "requirements": ["string"],
    "nice_to_haves": ["string"],
    "red_flags": ["string"],
    "cultural_signals": ["string"]
}
""",

    "generate_synthetic_candidate": """Generate a candidate profile for a {role_type} role.

Job Requirements:
{job_requirements}

Generate a {fit_type} candidate (good_fit or bad_fit):
- Name: {name}
- Experience: {years} years
- Skills: relevant to the role
- Behavioral signals: consistent with their profile
- Career history: realistic progression

Return as JSON matching the candidate schema.
""",

    # Behavioral signal analysis
    "analyze_behavioral_signals": """Analyze these behavioral signals and identify any patterns or concerns.

Behavioral Signals:
{behavioral_signals}

Identify:
1. Engagement level (high/medium/low)
2. Reliability indicators
3. Red flags in behavior
4. Market activity level

Output as JSON:
{
    "engagement_level": "high|medium|low",
    "reliability_score": 0.8,
    "concerns": ["concern1", "concern2"],
    "market_activity": "active|passive|inactive",
    "overall_assessment": "string"
}
""",
}