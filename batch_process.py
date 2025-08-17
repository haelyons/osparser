import os
import subprocess
import sys
import re

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
OUTPUT_DIR = 'new_outputs'

def sanitize_filename(name: str) -> str:
    """
    Sanitizes a string to be used as a valid filename by replacing
    special characters with underscores.
    """
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_')]).rstrip().replace(' ', '_')

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
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        sys.exit(1)

    pdf_files = list(find_pdfs(SOURCE_DIR))
    total_files = len(pdf_files)
    total_questions = len(QUESTIONS)
    print(f"Found {total_files} PDF files to process against {total_questions} questions.")

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

            # Sanitize the question to create a valid filename
            question_filename = sanitize_filename(question)
            
            # Define output paths with question number prefix
            output_filename_base = f"q{j+1:02d}_{question_filename}"
            output_pdf = os.path.join(doc_output_dir, f"{output_filename_base}.pdf")
            output_json = os.path.join(doc_output_dir, f"{output_filename_base}.json")
            
            # Halt and resume functionality: skip if output already exists
            if os.path.exists(output_pdf) and os.path.exists(output_json):
                print(f"    Skipping, output already exists.")
                continue

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
                # Run the command
                subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                print(f"    Successfully processed. Outputs saved.")
            except subprocess.CalledProcessError as e:
                print(f"    Error processing '{pdf_path}' with question '{question[:50]}...':")
                print(f"    Return code: {e.returncode}")
                print(f"    Output:\n{e.stdout}")
                print(f"    Error Output:\n{e.stderr}")
            except Exception as e:
                print(f"    An unexpected error occurred: {e}")

    print("\nBatch processing complete.")

if __name__ == '__main__':
    main()