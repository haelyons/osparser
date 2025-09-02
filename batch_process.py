import os
import subprocess
import sys
import re
import csv

QUESTIONS = [
    "What do they say about climate change",
    "What do they say about threats or pressures related to climate change",
    "How does Climate Change impact species, habitats and ecosystems?",
    "What do they say about gaps in relation to climate change",
    "What do they recommend in relation to climate change, including issues that require further investigation and/or research?",
    "What do they say about ocean acidification",
    "How does Ocean Acidification impact species, habitats and ecosystems?"
    #"How does CC affect the pressures on the marine environment?",
    #"How does Ocean Acidification affect the pressures on the marine environment?"
]

SOURCE_DIR = 'sources'
OUTPUT_DIR = 'outputs'
CSV_TEMPLATE = 'results/analysis_010925_v3_clean.csv'  # Default CSV with PDFs to process

def trim_question_for_filename(question: str, max_words: int = 8) -> str:
    """
    Simple word count cap for filename generation.
    """
    words = question.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return question

def sanitize_filename(name: str) -> str:
    """
    Simple filename sanitization - keep only letters, digits, and spaces, then replace spaces with underscores.
    """
    clean = ''.join(c for c in name if c.isalnum() or c.isspace())
    return clean.strip().replace(' ', '_')

def normalize_filename(filename: str) -> str:
    """Normalize filename for comparison by removing extension, lowercasing, and standardizing punctuation"""
    # Remove .pdf extension
    name = filename.replace('.pdf', '').replace('.PDF', '')
    # Convert to lowercase
    name = name.lower()
    # Normalize spaces and punctuation
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation except spaces
    name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
    return name

def get_pdfs_from_csv(csv_path: str, source_dir: str):
    """Get PDF paths from CSV filenames with improved fuzzy matching"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_filenames = [row['Filename'].strip() for row in reader if row['Filename'].strip()]
    
    # Create normalized versions for matching
    csv_normalized = [(filename, normalize_filename(filename)) for filename in csv_filenames]
    
    pdf_files = []
    matched_csv_files = set()
    
    # Find all PDF files and try to match them
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                file_normalized = normalize_filename(file)
                
                # Try to find a match in CSV - prioritize exact matches
                best_match = None
                partial_matches = []
                
                # First pass: look for exact matches
                for csv_original, csv_norm in csv_normalized:
                    if csv_original in matched_csv_files:
                        continue  # Skip already matched CSV files
                    
                    if csv_norm == file_normalized:  # Exact normalized match
                        best_match = csv_original
                        break
                
                # Second pass: look for partial matches only if no exact match found
                if not best_match:
                    for csv_original, csv_norm in csv_normalized:
                        if csv_original in matched_csv_files:
                            continue  # Skip already matched CSV files
                        
                        # More restrictive partial matching - avoid substring matches that are too broad
                        if len(csv_norm) > 3 and len(file_normalized) > 3:  # Avoid matching very short strings
                            # Only allow partial matches if they're reasonably similar in length
                            length_ratio = min(len(csv_norm), len(file_normalized)) / max(len(csv_norm), len(file_normalized))
                            if length_ratio >= 0.5:  # At least 50% length similarity
                                if csv_norm in file_normalized or file_normalized in csv_norm:
                                    partial_matches.append((csv_original, csv_norm))
                        
                        # Handle cases where CSV has extra timestamp/version info
                        elif len(csv_norm) > len(file_normalized) and file_normalized in csv_norm:
                            length_ratio = len(file_normalized) / len(csv_norm)
                            if length_ratio >= 0.3:  # File name is at least 30% of CSV name
                                partial_matches.append((csv_original, csv_norm))
                    
                    # If we have partial matches, prefer the one with highest similarity
                    if partial_matches:
                        # Use word-based similarity for better matching
                        best_score = 0
                        for csv_original, csv_norm in partial_matches:
                            csv_words = set(csv_norm.split())
                            file_words = set(file_normalized.split())
                            
                            if csv_words and file_words:
                                common_words = csv_words.intersection(file_words)
                                word_similarity = len(common_words) / len(csv_words)
                                
                                # Require high word overlap AND reasonable similarity
                                if word_similarity >= 0.7 and len(common_words) >= 2 and word_similarity > best_score:
                                    best_match = csv_original
                                    best_score = word_similarity
                
                if best_match:
                    pdf_files.append(file_path)
                    matched_csv_files.add(best_match)
    
    # Report unmatched files from CSV
    unmatched_files = [csv_file for csv_file, _ in csv_normalized if csv_file not in matched_csv_files]
    
    if unmatched_files:
        print(f"Warning: {len(unmatched_files)} files from CSV not found:")
        for missing in unmatched_files[:5]:
            print(f"  - {missing}")
        if len(unmatched_files) > 5:
            print(f"  ... and {len(unmatched_files) - 5} more")
    
    return pdf_files

def find_pdfs(start_path: str):
    """
    Recursively finds all PDF files in the given directory.
    """
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                yield os.path.join(root, file)

def main():
    """
    Main function to orchestrate the batch processing.
    """
    csv_file = sys.argv[1] if len(sys.argv) > 1 else CSV_TEMPLATE
    
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        sys.exit(1)

    pdf_files = get_pdfs_from_csv(csv_file, SOURCE_DIR)
    total_files = len(pdf_files)
    total_questions = len(QUESTIONS)
    print(f"Processing {total_files} PDFs from {csv_file}")
    
    # Track statistics
    stats = {
        'total_tasks': 0,
        'successful': 0,
        'skipped': 0,
        'failed': 0
    }

    for i, pdf_path in enumerate(pdf_files):
        # Create the corresponding output directory structure
        relative_path = os.path.relpath(pdf_path, SOURCE_DIR)
        
        dir_name = os.path.dirname(relative_path)
        file_name, _ = os.path.splitext(os.path.basename(relative_path))
        
        # Truncate filename if it starts with a project code like 'p12345_'
        match = re.match(r'p\d+_(.*)', file_name)
        if match:
            truncated_name = match.group(1)
        else:
            truncated_name = file_name
            
        doc_output_dir = os.path.join(OUTPUT_DIR, dir_name, truncated_name)
        os.makedirs(doc_output_dir, exist_ok=True)

        print(f"\nProcessing file {i+1}/{total_files}: {pdf_path}")
        print(f"  -> Output directory: {doc_output_dir}")

        for j, question in enumerate(QUESTIONS):
            print(f"  - Running question {j+1}/{total_questions}: '{question[:50]}...'")

            # Trim question to max 8 words for filename
            trimmed_question = trim_question_for_filename(question)
            question_filename = sanitize_filename(trimmed_question)
            output_filename_base = f"q{j+1:02d}_{question_filename}"
            output_pdf = os.path.join(doc_output_dir, f"{output_filename_base}.pdf")
            output_json = os.path.join(doc_output_dir, f"{output_filename_base}.json")
            
            # Halt and resume functionality: skip if output already exists
            if os.path.exists(output_pdf) and os.path.exists(output_json):
                print(f"    >> Skipping, output already exists.")
                stats['skipped'] += 1
                continue
            
            stats['total_tasks'] += 1

            # Construct the command to run highlighter.py
            command = [
                sys.executable, 'highlighter.py',
                '--pdf', pdf_path,
                '--question', question,
                '--output-pdf', output_pdf,
                '--output-json', output_json,
                # '--no-bm25' # Optional: uncomment if you want to disable BM25 for speed
            ]

            try:
                # For the first question of the first file, show the loading messages
                if i == 0 and j == 0:
                    print(f"    Running first question - showing GPU/model loading messages...")
                    result = subprocess.run(command, check=True, text=True, encoding='utf-8')
                else:
                    # For subsequent questions, capture output to keep it clean
                    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                
                # Verify that output files were actually created
                if os.path.exists(output_pdf) and os.path.exists(output_json):
                    print(f"    >> Success")
                    stats['successful'] += 1
                else:
                    print(f"    >> Failed - no output files created")
                    stats['failed'] += 1
                    
            except subprocess.CalledProcessError as e:
                print(f"    >> Error (exit code {e.returncode})")
                stats['failed'] += 1
                # Show key error info only if it's not the common "no text extracted" case
                if e.returncode != 1:  # 1 is our standard exit code for PDF processing issues
                    if hasattr(e, 'stderr') and e.stderr and "ERROR:" in e.stderr:
                        error_lines = [line.strip() for line in e.stderr.split('\n') if line.strip() and 'ERROR:' in line]
                        if error_lines:
                            print(f"    {error_lines[0]}")
            except Exception as e:
                print(f"    >> Unexpected error: {e}")
                stats['failed'] += 1

    print("\nBatch processing complete.")
    print(f"\nSummary:")
    print(f"  >> Successful: {stats['successful']}")
    print(f"  >> Skipped: {stats['skipped']}")
    print(f"  >> Failed: {stats['failed']}")
    print(f"  >> Total processed: {stats['total_tasks']}")
    
    if stats['failed'] > 0:
        print(f"\nNote: Failed tasks are typically due to malformed PDFs, missing text, or processing errors.")
        if stats['failed'] > stats['successful']:
            sys.exit(1)  # Exit with error code if more failures than successes

if __name__ == '__main__':
    main()