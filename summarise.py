import os
import json
import csv
import sys
import re
from pathlib import Path
import anthropic
from typing import Dict, List, Tuple, Optional

# Read API key from .keys file
def get_api_key() -> str:
    """Read the Anthropic API key from .keys file"""
    try:
        with open('.keys', 'r') as f:
            content = f.read().strip()
            for line in content.split('\n'):
                if line.startswith('ANTHROPIC_API_KEY='):
                    return line.split('=', 1)[1]
        raise ValueError("ANTHROPIC_API_KEY not found in .keys file")
    except FileNotFoundError:
        raise FileNotFoundError(".keys file not found. Please create it with your ANTHROPIC_API_KEY.")

# Questions from batch_process.py
QUESTIONS = [
    "What do they say about climate change",
    "What do they say about threats or pressures related to climate change", 
    "How does Climate Change impact species, habitats and ecosystems?",
    "What do they say about gaps in relation to climate change",
    "What do they recommend in relation to climate change, including issues that require further investigation and/or research?",
    "What do they say about ocean acidification",
    "How does Ocean Acidification impact species, habitats and ecosystems?"
]

def build_context_from_json(json_data: dict) -> str:
    """
    Build context string from JSON ranking data.
    Uses u_shaped_ranking for better attention window utilization.
    """
    rankings = json_data.get("u_shaped_ranking", json_data.get("standard_ranking", []))
    
    context_parts = []
    for item in rankings:
        context_window = item.get("context_window", "")
        page_num = item.get("page_number", "")
        if context_window:
            context_parts.append(f"[Page {page_num}] {context_window}")
    
    return "\n\n".join(context_parts)

def call_claude_api(question: str, context: str, api_key: str) -> str:
    """
    Call Claude API with the engineered prompt to summarize the context.
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are an expert marine science analyst tasked with summarizing technical information from OSPAR reports. Based only on the provided context below, answer the user's question comprehensively.

The question: {question}

Follow these steps: 
1. First, carefully read through ALL the provided context passages and the user's question to identify all relevant facts, findings, data points, and explicitly mentioned gaps or recommendations. 
2. Second, synthesize these points into a coherent, detailed, and comprehensive analysis. Do not use generic phrases like "The document discusses..." or "The text mentions...". Instead, state the findings directly and authoritatively. For example, instead of "The document states that temperatures are rising," write "Temperatures in the OSPAR Maritime Area have risen by X°C...". 
3. Be thorough - you have substantial context to work with. Use it all to provide a complete picture. 
4. Ensure your summary is objective and strictly derived from the provided text. Do not add any information not present in the context. If the context is insufficient to answer the question fully, state what is available and note the limitations. 
5. If the provided context doesn't address the specific question asked, clearly explain what the context does contain and why it's not relevant to the question.
6. Ignore statements regarding OSPAR's vision or purpose in the reported context, and do not repeat or quote them. Stick to reporting the exact findings as presented in the context provided. Focus only on substantive scientific findings, data, and analysis. Do not include any organizational statements, visions, purposes, or background information about OSPAR itself.

Answer in plain text, with a maximum of 250 words, without subheadings or sections. Be scientific and concise.

Context:
{context}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20000,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        # Extract text content from the response
        content = message.content[0].text if message.content else ""
        return content.strip()
        
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return f"ERROR: Failed to generate summary - {str(e)}"

def find_matching_csv_entry(pdf_name: str, csv_data: List[Dict]) -> Optional[Dict]:
    """
    Find the matching entry in CSV data based on the PDF filename.
    Handles the truncated names and project codes.
    """
    # Remove extension and any project code prefix
    clean_name = pdf_name.replace('.pdf', '')
    match = re.match(r'p\d+_(.*)', clean_name)
    if match:
        clean_name = match.group(1)
    
    for entry in csv_data:
        filename = entry.get('Filename', '')
        if filename:
            # Try exact match first
            if filename == pdf_name or filename == clean_name:
                return entry
            
            # Try partial matching (for cases where CSV has truncated names)
            csv_clean = filename.replace('.pdf', '')
            csv_match = re.match(r'p\d+_(.*)', csv_clean)
            if csv_match:
                csv_clean = csv_match.group(1)
            
            if csv_clean == clean_name or clean_name in csv_clean or csv_clean in clean_name:
                return entry
    
    return None

def extract_pdf_name_from_path(json_path: str) -> str:
    """
    Extract the PDF name from the JSON file path structure.
    """
    path_parts = Path(json_path).parts
    if len(path_parts) >= 3:
        # Get the directory name which should be the document name
        doc_dir = path_parts[-2]
        return doc_dir
    return ""

def get_keyword_counts(json_data: dict) -> Dict[str, int]:
    """
    Extract keyword counts from JSON data.
    """
    keyword_summary = json_data.get("document_keyword_summary", {})
    return keyword_summary.get("keyword_counts", {})

def process_json_files(output_dir: str, csv_file: str, api_key: str, test_mode: bool = False, max_docs: int = None):
    """
    Process all JSON files and update the CSV with summaries and keyword counts.
    """
    # Read the existing CSV
    csv_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)  # Convert to list to allow modifications
        csv_data = list(reader)
    
    # Add new columns for keyword counts if they don't exist
    new_columns = ['climate_change_hits', 'ocean_acidification_hits', 'total_keyword_hits']
    for col in new_columns:
        if col not in fieldnames:
            fieldnames.append(col)
    
    # Ensure all question columns exist in fieldnames
    for question in QUESTIONS:
        if question not in fieldnames:
            fieldnames.append(question)
    
    # Process each JSON file
    json_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Group by document for better processing and resume capability
    doc_files = {}
    for json_path in json_files:
        doc_name = extract_pdf_name_from_path(json_path)
        if doc_name not in doc_files:
            doc_files[doc_name] = []
        doc_files[doc_name].append(json_path)
    
    # Limit for testing
    if test_mode and max_docs:
        doc_items = list(doc_files.items())[:max_docs]
        doc_files = dict(doc_items)
        print(f"TEST MODE: Processing only first {max_docs} documents")
    
    total_files = sum(len(files) for files in doc_files.values())
    total_docs = len(doc_files)
    print(f"Found {total_files} JSON files across {total_docs} documents to process...")
    
    processed_count = 0
    skipped_count = 0
    
    for doc_name, doc_json_files in doc_files.items():
        # Find matching CSV entry
        matching_entry = find_matching_csv_entry(doc_name, csv_data)
        if not matching_entry:
            print(f"Warning: No CSV entry found for {doc_name}")
            continue
        
        # Check if this document already has all summaries (for resume capability)
        all_complete = True
        for question in QUESTIONS:
            if not matching_entry.get(question, "").strip():
                all_complete = False
                break
        
        if all_complete:
            print(f"Skipping {doc_name} - already completed")
            skipped_count += 1
            continue
        
        print(f"\nProcessing document: {doc_name}")
        
        # Process each question for this document
        for json_path in doc_json_files:
            try:
                filename = os.path.basename(json_path)
                if not filename.startswith('q'):
                    continue
                    
                question_match = re.match(r'q(\d+)_', filename)
                if not question_match:
                    continue
                    
                question_num = int(question_match.group(1)) - 1  # Convert to 0-indexed
                if question_num >= len(QUESTIONS):
                    continue
                    
                question = QUESTIONS[question_num]
                
                # Check if this specific question already has an answer (for resume)
                if matching_entry.get(question, "").strip():
                    print(f"  Skipping question {question_num + 1}: already answered")
                    continue
                
                # Read and process JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Build context and call Claude
                context = build_context_from_json(json_data)
                if not context.strip():
                    summary = "No relevant context found in the document."
                else:
                    print(f"  Processing question {question_num + 1}/{len(QUESTIONS)}: {question[:50]}...")
                    summary = call_claude_api(question, context, api_key)
                
                # Update CSV entry with summary
                matching_entry[question] = summary
                
                # Update keyword counts (only for the first question to avoid duplication)
                if question_num == 0:
                    keyword_counts = get_keyword_counts(json_data)
                    matching_entry['climate_change_hits'] = keyword_counts.get('climate change', 0)
                    matching_entry['ocean_acidification_hits'] = keyword_counts.get('ocean acidification', 0)
                    matching_entry['total_keyword_hits'] = sum(keyword_counts.values())
                
                processed_count += 1
                
            except Exception as e:
                print(f"  Error processing {json_path}: {e}")
                continue
    
    # Write updated CSV
    backup_file = csv_file.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(csv_file, backup_file)
        print(f"Created backup: {backup_file}")
    else:
        print(f"Backup already exists: {backup_file}")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} question-document pairs")
    print(f"Skipped: {skipped_count} already completed documents")
    print(f"Updated CSV: {csv_file}")
    print(f"Added keyword count columns: {', '.join(new_columns)}")

def main():
    """Main function to orchestrate the summarization process."""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print("OSPAR Document Summarization Tool")
        print("Usage: python summarizer.py [csv_file] [output_dir]")
        print("  csv_file: Path to the results CSV file (default: results/results_template.csv)")
        print("  output_dir: Path to the outputs directory (default: outputs)")
        print("\nThis script will:")
        print("1. Process JSON files from the highlighting stage")
        print("2. Generate summaries using Claude API")
        print("3. Update the CSV with summaries and keyword counts")
        print("4. Create a backup of the original CSV")
        sys.exit(0)
    
    # Parse arguments, filtering out flags
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    csv_file = args[0] if len(args) > 0 else "results/results_template.csv"
    output_dir = args[1] if len(args) > 1 else "outputs"
    
    # Validate inputs
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        sys.exit(1)
    
    # Get API key
    try:
        api_key = get_api_key()
        print("Successfully loaded API key.")
    except Exception as e:
        print(f"Error loading API key: {e}")
        sys.exit(1)
    
    # Check for test mode
    test_mode = '--test' in sys.argv
    if test_mode:
        print("Running in TEST MODE (max 3 documents)")
    
    # Process files
    try:
        process_json_files(output_dir, csv_file, api_key, test_mode=test_mode, max_docs=3 if test_mode else None)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
