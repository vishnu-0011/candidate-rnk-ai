@echo off
cd /d "%~dp0"
python main.py data\job_description.docx data\candidates.jsonl output\ranked_candidates.csv
