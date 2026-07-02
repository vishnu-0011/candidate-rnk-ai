"""
Utility functions for the Redrobe ranking system.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(file_path: str) -> dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def save_csv(
    records: list[dict[str, Any]], file_path: str, fieldnames: list[str] | None = None
) -> None:
    """Save records to a CSV file."""
    if not records:
        return

    if fieldnames is None:
        fieldnames = list(records[0].keys())

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def ensure_directory(file_path: str) -> None:
    """Ensure the directory for a file path exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def format_score(score: float, decimals: int = 3) -> float:
    """Format a score to specified decimal places."""
    return round(score, decimals)


def get_timestamp() -> str:
    """Get current timestamp as ISO format string."""
    return datetime.now().isoformat()


def merge_rankings(
    rankings: list[dict[str, Any]], weights: dict[str, float]
) -> list[dict[str, Any]]:
    """
    Merge multiple ranking results with specified weights.

    Args:
        rankings: List of ranking dictionaries with 'score' keys
        weights: Dict mapping field names to weights

    Returns:
        Merged rankings sorted by combined score
    """
    if not rankings:
        return []

    merged = []
    for r in rankings:
        combined_score = sum(
            r.get(field, 0) * weight for field, weight in weights.items()
        )
        merged_r = r.copy()
        merged_r["combined_score"] = combined_score
        merged.append(merged_r)

    merged.sort(key=lambda x: x["combined_score"], reverse=True)
    return merged


def truncate_reasoning(reasoning: list[str], max_items: int = 5) -> str:
    """Truncate reasoning list to a readable string."""
    if not reasoning:
        return "No specific reasoning available."

    truncated = reasoning[:max_items]
    if len(reasoning) > max_items:
        truncated.append(f"... and {len(reasoning) - max_items} more items")

    return "; ".join(truncated)


def get_candidate_identity(candidate: dict[str, Any]) -> str:
    """Get a readable identifier for a candidate."""
    profile = candidate.get("profile", {})
    return f"{profile.get('current_title', 'Unknown')} at {profile.get('current_company', 'Unknown')}"
