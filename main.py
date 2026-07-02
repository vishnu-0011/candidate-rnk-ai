#!/usr/bin/env python3
"""
Main Entry Point - Redrobe AI Candidate Ranking System.

Run the complete ranking pipeline with a single command:
    python main.py <job_file> <candidates_file> [output_file]

Example:
    python main.py data/job_description.txt data/candidates.jsonl output/ranked.csv
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ranking import CandidateRanker, run_ranking_system
from src.job_parser import load_job_description
from config import LLM_CONFIG, PATH_CONFIG


def print_banner():
    """Print the project banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║        Redrobe AI Candidate Ranking System v1.0               ║
    ║        Semantic Understanding + Behavioral Scoring            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_job_summary(parsed_job: dict):
    """Print a summary of the parsed job requirements."""
    print("\n" + "=" * 60)
    print("JOB REQUIREMENTS")
    print("=" * 60)
    print(f"Role: {parsed_job.get('role_title', 'N/A')}")
    print(f"Experience: {parsed_job.get('experience_years', {})}")
    print(f"Location: {parsed_job.get('location', 'N/A')}")
    print(f"Work Mode: {parsed_job.get('work_mode', 'N/A')}")
    print(f"\nCore Skills ({len(parsed_job.get('core_skills', []))}):")
    for skill in parsed_job.get('core_skills', []):
        print(f"  • {skill}")
    print(f"\nRed Flags Detected ({len(parsed_job.get('red_flags', []))}):")
    for flag in parsed_job.get('red_flags', []):
        print(f"  ⚠ {flag}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-powered candidate ranking system"
    )
    parser.add_argument(
        "job_file",
        help="Path to job description file (.txt or .docx)"
    )
    parser.add_argument(
        "candidates_file",
        help="Path to candidates JSONL file"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=PATH_CONFIG.output_file,
        help="Path for output CSV (default: output/ranked_candidates.csv)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Limit to top N candidates"
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openai"],
        default=LLM_CONFIG.provider,
        help="LLM provider for processing"
    )
    parser.add_argument(
        "--no-explanations",
        action="store_true",
        help="Skip LLM explanations for faster processing"
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check if files exist
    if not Path(args.job_file).exists():
        print(f"Error: Job file not found: {args.job_file}")
        sys.exit(1)

    if not Path(args.candidates_file).exists():
        print(f"Error: Candidates file not found: {args.candidates_file}")
        sys.exit(1)

    # Parse job description
    print(f"\n📄 Parsing job description: {args.job_file}")
    try:
        job_text = load_job_description(args.job_file)
    except Exception as e:
        print(f"Error loading job description: {e}")
        sys.exit(1)

    # Initialize ranker
    print(f"🤖 Using LLM provider: {args.provider}")
    ranker = CandidateRanker(
        job_description=job_text,
        provider=args.provider
    )

    # Print job summary
    print_job_summary(ranker.parsed_job)

    # Run ranking
    print("\n" + "=" * 60)
    print("RANKING CANDIDATES")
    print("=" * 60)

    results = run_ranking_system(
        job_file=args.job_file,
        candidates_file=args.candidates_file,
        output_file=args.output_file,
        top_n=args.top_n
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total candidates processed: {len(results)}")

    if results:
        top_score = results[0]["score"]
        print(f"Top candidate score: {top_score:.4f}")

        print("\nTop 5 Candidates:")
        for r in results[:5]:
            print(f"  #{r['rank']} {r['candidate_id']} - Score: {r['score']:.4f}")

    print(f"\n✅ Output saved to: {args.output_file}")
    print(f"   Full explanations available in the CSV file.")


if __name__ == "__main__":
    main()
