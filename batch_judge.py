#!/usr/bin/env python3
"""
This script processes documents from analysis_180825_v3_clean.csv and rates their relevance
to 7 climate-related questions using the judge.py script. It creates both a rated CSV file
and a separate JSON file containing rationales.

The script automatically detects existing ratings and rationales, allowing you to
stop and restart the process without losing progress. Previously completed evaluations are
skipped, and the script continues from where it left off.

Usage: python batch_judge.py [--test] [--start-row N] [--max-rows N]
"""

import csv
import json
import sys
import subprocess
import os
from typing import Dict, List, Tuple
from datetime import datetime
import argparse

# Define the 7 questions and their corresponding column indices
QUESTIONS = [
    ("What do they say about climate change", 4, 5),
    ("What do they say about threats or pressures related to climate change", 6, 7), 
    ("How does Climate Change impact species, habitats and ecosystems?", 8, 9),
    ("What do they say about gaps in relation to climate change", 10, 11),
    ("What do they recommend in relation to climate change, including issues that require further investigation and/or research?", 12, 13),
    ("What do they say about ocean acidification", 14, 15),
    ("How does Ocean Acidification impact species, habitats and ecosystems?", 16, 17)
]

def load_csv_data(input_file: str) -> Tuple[List[str], List[List[str]]]:
    """Load the CSV data and return headers and rows."""
    with open(input_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows

def call_judge(question: str, summary: str) -> Dict:
    """Call the judge.py script and return the result."""
    try:
        # Call the judge script with the question and summary
        result = subprocess.run([
            sys.executable, 'judge.py', question, summary
        ], capture_output=True, text=True, check=True)
        
        # Parse the JSON response
        response = json.loads(result.stdout.strip())
        return response
        
    except subprocess.CalledProcessError as e:
        print(f"Error calling judge script: {e}")
        print(f"stderr: {e.stderr}")
        return {"rationale": f"ERROR: {e.stderr}", "score": 1, "normalized_score": 0.0}
        
    except json.JSONDecodeError as e:
        print(f"Error parsing judge response: {e}")
        print(f"Raw output: {result.stdout}")
        return {"rationale": f"ERROR: Invalid JSON response", "score": 1, "normalized_score": 0.0}
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"rationale": f"ERROR: {e}", "score": 1, "normalized_score": 0.0}

def save_csv_with_ratings(headers: List[str], rows: List[List[str]], output_file: str):
    """Save the updated CSV with ratings."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def save_rationales(rationales: Dict, output_file: str):
    """Save the rationales to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rationales, f, indent=2, ensure_ascii=False)

def process_documents(input_file: str, output_csv: str, output_json: str, 
                     test_mode: bool = False, start_row: int = 0, max_rows: int = None):
    """Main processing function."""
    
    print(f"Loading data from {input_file}...")
    headers, rows = load_csv_data(input_file)
    
    # Check if output CSV already exists and load existing ratings
    existing_ratings = {}
    if os.path.exists(output_csv):
        print(f"Found existing output file: {output_csv}")
        print("Loading existing ratings to avoid overwriting...")
        try:
            existing_headers, existing_rows = load_csv_data(output_csv)
            # Map existing ratings by row index
            for i, existing_row in enumerate(existing_rows):
                if i < len(rows) and len(existing_row) > len(QUESTIONS):
                    # Store existing ratings for this row
                    row_ratings = {}
                    for q_idx, (_, _, rating_col) in enumerate(QUESTIONS):
                        if rating_col < len(existing_row) and existing_row[rating_col].strip():
                            row_ratings[q_idx] = existing_row[rating_col]
                    if row_ratings:
                        existing_ratings[i] = row_ratings
            print(f"Loaded existing ratings for {len(existing_ratings)} document rows")
        except Exception as e:
            print(f"Warning: Could not load existing ratings: {e}")
            print("Starting fresh...")
    
    # Filter out empty rows and get actual document rows
    document_rows = []
    for i, row in enumerate(rows):
        if len(row) >= 4 and row[0].strip() and row[1].strip() and row[3].strip():
            document_rows.append((i, row))
    
    print(f"Found {len(document_rows)} document rows to process")
    
    # Apply start_row and max_rows filtering
    if start_row > 0:
        document_rows = document_rows[start_row:]
        print(f"Starting from row {start_row}: {len(document_rows)} rows remaining")
    
    if max_rows:
        document_rows = document_rows[:max_rows]
        print(f"Processing maximum {max_rows} rows: {len(document_rows)} rows selected")
    
    if test_mode:
        document_rows = document_rows[:3]  # Only first 3 for testing
        print(f"TEST MODE: Processing only {len(document_rows)} documents")
    
    # Load existing rationales if available
    rationales = {
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "total_documents": len(document_rows),
            "total_evaluations": len(document_rows) * len(QUESTIONS)
        },
        "evaluations": {}
    }
    
    if os.path.exists(output_json):
        print(f"Found existing rationales file: {output_json}")
        print("Loading existing rationales...")
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_rationales = json.load(f)
                if "evaluations" in existing_rationales:
                    rationales["evaluations"] = existing_rationales["evaluations"]
                    print(f"Loaded existing rationales for {len(rationales['evaluations'])} documents")
        except Exception as e:
            print(f"Warning: Could not load existing rationales: {e}")
            print("Starting with empty rationales...")
    
    total_evaluations = len(document_rows) * len(QUESTIONS)
    current_evaluation = 0
    skipped_evaluations = 0
    completed_evaluations = 0
    
    print(f"\nStarting evaluation of {total_evaluations} document-question pairs...")
    print("=" * 70)
    
    # Process each document
    for doc_idx, (original_row_idx, row) in enumerate(document_rows):
        doc_type = row[0]
        doc_name = row[1] 
        filename = row[3]
        
        print(f"\nDocument {doc_idx + 1}/{len(document_rows)}: {doc_name}")
        print(f"Type: {doc_type}, File: {filename}")
        
        doc_key = f"doc_{original_row_idx}_{filename}"
        rationales["evaluations"][doc_key] = {
            "document_info": {
                "type": doc_type,
                "name": doc_name,
                "filename": filename,
                "original_row": original_row_idx + 1  # +1 for 1-based indexing
            },
            "questions": {}
        }
        
        # Process each question for this document
        for q_idx, (question, summary_col, rating_col) in enumerate(QUESTIONS):
            current_evaluation += 1
            
            # Check if this evaluation already exists
            has_existing_rating = (original_row_idx in existing_ratings and 
                                 q_idx in existing_ratings[original_row_idx])
            has_existing_rationale = (doc_key in rationales["evaluations"] and 
                                    f"Q{q_idx + 1}" in rationales["evaluations"][doc_key].get("questions", {}))
            
            if has_existing_rating or has_existing_rationale:
                print(f"  Q{q_idx + 1}: Already completed - skipping")
                skipped_evaluations += 1
                # Restore existing rating to current row if needed
                if has_existing_rating:
                    existing_score = existing_ratings[original_row_idx][q_idx]
                    if rating_col < len(rows[original_row_idx]):
                        rows[original_row_idx][rating_col] = existing_score
                    else:
                        # Extend the row if needed
                        while len(rows[original_row_idx]) <= rating_col:
                            rows[original_row_idx].append("")
                        rows[original_row_idx][rating_col] = existing_score
                continue
            
            summary = row[summary_col] if summary_col < len(row) else ""
            
            # Skip if no summary
            if not summary.strip():
                print(f"  Q{q_idx + 1}: Skipping (no summary)")
                if rating_col < len(rows[original_row_idx]):
                    rows[original_row_idx][rating_col] = ""
                else:
                    # Extend the row if needed
                    while len(rows[original_row_idx]) <= rating_col:
                        rows[original_row_idx].append("")
                    rows[original_row_idx][rating_col] = ""
                continue
            
            print(f"  Q{q_idx + 1}: {question[:50]}..." if len(question) > 50 else f"  Q{q_idx + 1}: {question}")
            
            # Call the judge
            result = call_judge(question, summary)
            score = result.get('score', 1)
            rationale = result.get('rationale', 'No rationale provided')
            
            # Update the CSV row with the score
            if rating_col < len(rows[original_row_idx]):
                rows[original_row_idx][rating_col] = str(score)
            else:
                # Extend the row if needed
                while len(rows[original_row_idx]) <= rating_col:
                    rows[original_row_idx].append("")
                rows[original_row_idx][rating_col] = str(score)
            
            # Store the rationale
            rationales["evaluations"][doc_key]["questions"][f"Q{q_idx + 1}"] = {
                "question": question,
                "score": score,
                "normalized_score": result.get('normalized_score', (score - 1) / 4.0),
                "rationale": rationale,
                "summary_length": len(summary)
            }
            
            print(f"    Score: {score}/5 ({result.get('normalized_score', 0.0):.2f} normalized)")
            completed_evaluations += 1
            
            # Progress indicator
            progress = (current_evaluation / total_evaluations) * 100
            print(f"    Progress: {current_evaluation}/{total_evaluations} ({progress:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Saving results...")
    
    # Save the updated CSV
    save_csv_with_ratings(headers, rows, output_csv)
    print(f"✓ Rated CSV saved to: {output_csv}")
    
    # Save the rationales
    save_rationales(rationales, output_json)
    print(f"✓ Rationales saved to: {output_json}")
    
    print(f"\nProcessing complete!")
    print(f"Documents processed: {len(document_rows)}")
    print(f"Total evaluations: {current_evaluation}")
    print(f"  - New evaluations completed: {completed_evaluations}")
    print(f"  - Existing evaluations skipped: {skipped_evaluations}")
    print(f"  - Empty summaries skipped: {current_evaluation - completed_evaluations - skipped_evaluations}")
    print(f"CSV with ratings: {output_csv}")
    print(f"Rationales (JSON): {output_json}")
    
    if completed_evaluations == 0 and skipped_evaluations > 0:
        print(f"\n🎉 All evaluations were already complete! No new work needed.")
    elif completed_evaluations > 0:
        print(f"\n✅ Successfully completed {completed_evaluations} new evaluations.")
        if skipped_evaluations > 0:
            print(f"📋 Resumed from previous session - {skipped_evaluations} evaluations were preserved.")

def main():
    parser = argparse.ArgumentParser(description='Rate document relevance using LLM-as-a-Judge')
    parser.add_argument('--test', action='store_true', help='Test mode - process only first 3 documents')
    parser.add_argument('--start-row', type=int, default=0, help='Start processing from this document row (0-based)')
    parser.add_argument('--max-rows', type=int, help='Maximum number of documents to process')
    parser.add_argument('--input', default='results/analysis_180825_v3_clean.csv', help='Input CSV file')
    parser.add_argument('--output-csv', default='results/analysis_180825_v3_rated.csv', help='Output CSV file')
    parser.add_argument('--output-json', default='results/rating_rationales.json', help='Output JSON file for rationales')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Check if judge.py exists
    if not os.path.exists('judge.py'):
        print("Error: judge.py script not found in current directory.")
        sys.exit(1)
    
    print("Document Relevance Rating Script")
    print("=" * 40)
    print(f"Input file: {args.input}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output JSON: {args.output_json}")
    if args.test:
        print("Mode: TEST (first 3 documents only)")
    elif args.max_rows:
        print(f"Mode: LIMITED ({args.max_rows} documents max)")
    else:
        print("Mode: FULL PROCESSING")
    
    if args.start_row > 0:
        print(f"Starting from document row: {args.start_row}")
    
    print()
    
    try:
        process_documents(
            args.input, 
            args.output_csv, 
            args.output_json, 
            test_mode=args.test,
            start_row=args.start_row,
            max_rows=args.max_rows
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
