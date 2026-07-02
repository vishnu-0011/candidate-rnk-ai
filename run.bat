@echo off
:: Redrobe AI Candidate Ranking System - Windows Batch Script
:: Usage: run.bat [job_file] [candidates_file] [output_file]

echo ========================================
echo    Redrobe AI Candidate Ranking System
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: Parse arguments
set JOB_FILE=%~1
set CANDIDATES_FILE=%~2
set OUTPUT_FILE=%~3

:: Default values if not provided
if "%JOB_FILE%"=="" set JOB_FILE=data\job_description.docx
if "%CANDIDATES_FILE%"=="" set CANDIDATES_FILE=data\candidates.jsonl
if "%OUTPUT_FILE%"=="" set OUTPUT_FILE=output\ranked_candidates.csv

echo Job File: %JOB_FILE%
echo Candidates File: %CANDIDATES_FILE%
echo Output File: %OUTPUT_FILE%
echo.

:: Check if files exist
if not exist "%JOB_FILE%" (
    echo Error: Job file not found: %JOB_FILE%
    exit /b 1
)

if not exist "%CANDIDATES_FILE%" (
    echo Error: Candidates file not found: %CANDIDATES_FILE%
    exit /b 1
)

:: Run the ranking system
echo Starting ranking...
python main.py "%JOB_FILE%" "%CANDIDATES_FILE%" "%OUTPUT_FILE%"

:: Check exit status
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo    Ranking Complete!
    echo ========================================
    echo.
    echo Output saved to: %OUTPUT_FILE%
) else (
    echo.
    echo ========================================
    echo    Ranking Failed
    echo ========================================
)
