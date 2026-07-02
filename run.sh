#!/bin/bash

# Redrobe AI Candidate Ranking System - Run Script
# Usage: ./run.sh [job_file] [candidates_file] [output_file]

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Redrobe AI Candidate Ranking System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set up Python environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Parse arguments
JOB_FILE="${1:-data/job_description.docx}"
CANDIDATES_FILE="${2:-data/candidates.jsonl}"
OUTPUT_FILE="${3:-output/ranked_candidates.csv}"

echo -e "${GREEN}Job File:${NC} $JOB_FILE"
echo -e "${GREEN}Candidates File:${NC} $CANDIDATES_FILE"
echo -e "${GREEN}Output File:${NC} $OUTPUT_FILE"
echo ""

# Check if files exist
if [ ! -f "$JOB_FILE" ]; then
    echo -e "${RED}Error: Job file not found: $JOB_FILE${NC}"
    exit 1
fi

if [ ! -f "$CANDIDATES_FILE" ]; then
    echo -e "${RED}Error: Candidates file not found: $CANDIDATES_FILE${NC}"
    exit 1
fi

# Run the ranking system
echo -e "${BLUE}Starting ranking...${NC}"
python main.py "$JOB_FILE" "$CANDIDATES_FILE" "$OUTPUT_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Ranking Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Output saved to: $OUTPUT_FILE"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}   Ranking Failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
