import os
import json
import csv
import sys
import glob
import anthropic
from typing import Dict, List, Optional

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

def get_csv_column_name(question_num: int, question_text: str) -> str:
    """
    Convert question index and text to the CSV column format: Q{n}: {question_text}
    """
    return f"Q{question_num + 1}: {question_text}"

def find_document_json_directory(output_dir: str, doc_name: str) -> Optional[str]:
    """
    Find the directory containing JSON files for a given document name.
    Searches through the output directory structure with case-insensitive matching.
    """
    for root, dirs, files in os.walk(output_dir):
        # Check if this directory name matches the document name (case-insensitive)
        dir_name = os.path.basename(root)
        if dir_name.lower() == doc_name.lower():
            # Verify it contains JSON files
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                return root
    return None

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



def get_keyword_counts(json_data: dict) -> Dict[str, int]:
    """
    Extract keyword counts from JSON data.
    """
    keyword_summary = json_data.get("document_keyword_summary", {})
    return keyword_summary.get("keyword_counts", {})

def save_csv_data(csv_data: List[Dict], fieldnames: List[str], csv_file: str):
    """
    Save CSV data to file.
    """
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

def process_csv_documents(output_dir: str, csv_file: str, api_key: str, test_mode: bool = False, max_docs: int = None):
    """
    Process documents specified in the CSV file and update with summaries and keyword counts.
    """
    # Create backup before processing
    backup_file = csv_file.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(csv_file, backup_file)
        print(f"Created backup: {backup_file}")
    else:
        print(f"Backup already exists: {backup_file}")
    
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
    for i, question in enumerate(QUESTIONS):
        column_name = get_csv_column_name(i, question)
        if column_name not in fieldnames:
            fieldnames.append(column_name)
    
    # Filter CSV entries that have filenames and are processable
    processable_entries = []
    for entry in csv_data:
        filename = entry.get('Filename', '').strip()
        if filename and filename.lower() != 'filename':  # Skip header-like entries
            processable_entries.append(entry)
    
    # Limit for testing
    if test_mode and max_docs:
        processable_entries = processable_entries[:max_docs]
        print(f"TEST MODE: Processing only first {max_docs} documents")
    
    print(f"Found {len(processable_entries)} documents in CSV to process...")
    
    processed_count = 0
    skipped_count = 0
    not_found_count = 0
    
    for entry in processable_entries:
        filename = entry.get('Filename', '')
        doc_name = entry.get('Name', '')
        
        # Convert filename to document directory name (remove .pdf extension)
        doc_dir_name = filename.replace('.pdf', '') if filename.endswith('.pdf') else filename
        
        # Check if this document already has all summaries (for resume capability)
        all_complete = True
        for i, question in enumerate(QUESTIONS):
            column_name = get_csv_column_name(i, question)
            if not entry.get(column_name, "").strip():
                all_complete = False
                break
        
        if all_complete:
            print(f"Skipping {doc_name} - already completed")
            skipped_count += 1
            continue
        
        # Find the JSON files for this document
        json_dir_path = find_document_json_directory(output_dir, doc_dir_name)
        if not json_dir_path:
            print(f"Warning: No JSON directory found for {doc_name} (looking for {doc_dir_name})")
            not_found_count += 1
            continue
        
        print(f"\nProcessing document: {doc_name}")
        
        # Track if any changes were made to this document
        document_updated = False
        
        # Process each question for this document
        for i, question in enumerate(QUESTIONS):
            column_name = get_csv_column_name(i, question)
            
            # Check if this specific question already has an answer (for resume)
            if entry.get(column_name, "").strip():
                print(f"  Skipping question {i + 1}: already answered")
                continue
            
            # Look for the JSON file for this question
            json_filename = f"q{i+1:02d}_*.json"
            json_files = glob.glob(os.path.join(json_dir_path, json_filename))
            
            if not json_files:
                print(f"  Warning: No JSON file found for question {i+1}")
                continue
            
            json_path = json_files[0]  # Take the first match
            
            try:
                # Read and process JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Build context and call Claude
                context = build_context_from_json(json_data)
                if not context.strip():
                    summary = "No relevant context found in the document."
                else:
                    print(f"  Processing question {i + 1}/{len(QUESTIONS)}: {question[:50]}...")
                    summary = call_claude_api(question, context, api_key)
                
                # Update CSV entry with summary
                entry[column_name] = summary
                document_updated = True
                
                # Update keyword counts (only for the first question to avoid duplication)
                if i == 0:
                    keyword_counts = get_keyword_counts(json_data)
                    entry['climate_change_hits'] = keyword_counts.get('climate change', 0)
                    entry['ocean_acidification_hits'] = keyword_counts.get('ocean acidification', 0)
                    entry['total_keyword_hits'] = sum(keyword_counts.values())
                
                processed_count += 1
                
            except Exception as e:
                print(f"  Error processing {json_path}: {e}")
                continue
        
        # Save CSV after each document to preserve progress
        if document_updated:
            save_csv_data(csv_data, fieldnames, csv_file)
            print(f"  Saved progress for {doc_name}")
    
    # Final save to ensure all data is written
    save_csv_data(csv_data, fieldnames, csv_file)
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} question-document pairs")
    print(f"Skipped: {skipped_count} already completed documents")
    print(f"Not found: {not_found_count} documents without JSON directories")
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
        process_csv_documents(output_dir, csv_file, api_key, test_mode=test_mode, max_docs=3 if test_mode else None)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
