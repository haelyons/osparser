# pip install PyMuPDF "sentence-transformers>=2.2.0" numpy torch nltk argparse json rank-bm25 

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch
import nltk
import os
import argparse
import json
import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
from collections import defaultdict

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK's 'punkt' tokenizer...")
    nltk.download('punkt')
    # nltk.download('punkt/tab')

# For accuracy, we select top-tier models from the MTEB leaderboard
# NV-Embed-v2 and BAAI/bge-large-en-v1.5 are SOTA embedding models
BI_ENCODER_MODEL = 'BAAI/bge-large-en-v1.5'

# For the re-ranking step, we use a powerful cross-encoder
# mixedbread-ai/mxbai-rerank-large-v1 is a top open-source performer
CROSS_ENCODER_MODEL = 'mixedbread-ai/mxbai-rerank-large-v1'

# Keywords for sparse retrieval and highlighting
KEYWORDS = [
    "climate change", "ocean acidification", "sea level rise", "warming",
    "temperature increase", "carbon dioxide", "co2", "greenhouse gas",
    "extreme weather", "marine heatwave", "hypoxia", "anoxia"
]

# Sections to exclude from processing
EXCLUDED_SECTIONS = [
    "Key Message", "Executive Summary", "Conclusion", 
    "Bibliography", "References"
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# The bi-encoder is used to create vector embeddings for sentences
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device=device)
# The cross-encoder is used to re-rank the retrieved sentences for maximum relevance
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
print("Models initialized successfully.")

def extract_text_with_styles(pdf_path):
    """Extract all text with complete style information"""
    doc = fitz.open(pdf_path)
    styled_blocks = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        styled_blocks.append({
                            'text': text,
                            'page': page_num,
                            'bbox': span["bbox"],
                            'font_size': round(span["size"], 1),
                            'font_name': span["font"],
                            'flags': span["flags"],
                            'is_bold': bool(span["flags"] & 2**4),
                            'is_italic': bool(span["flags"] & 2**1),
                            'style_signature': (
                                round(span["size"], 1),
                                span["font"],
                                span["flags"]
                            )
                        })
    
    doc.close()
    return styled_blocks

def find_excluded_sections_by_style(pdf_path: str) -> Tuple[List[Tuple[int, int, str]], List[Dict]]:
    """
    Find excluded sections using style-based approach
    Returns (excluded_sections, styled_blocks) where excluded_sections contains
    (start_block_idx, end_block_idx, section_name) tuples
    """
    styled_blocks = extract_text_with_styles(pdf_path)
    
    # Identify all potential headers by finding unique header styles
    header_styles = set()
    potential_headers = []
    
    # Look for text that could be headers (short, potentially capitalized)
    for i, block in enumerate(styled_blocks):
        text = block['text'].strip()
        if (len(text) < 100 and  # Not too long
            len(text) > 2 and    # Not too short
            (text[0].isupper() or  # Starts with capital
             any(keyword in text.lower() for keyword in ['conclusion', 'reference', 'key message', 'executive', 'summary', 'background', 'method', 'result', 'introduction', 'abstract']))):
            potential_headers.append((i, block))
            header_styles.add(block['style_signature'])
    
    excluded_sections = []
    
    for excluded_name in EXCLUDED_SECTIONS:
        for i, block in enumerate(styled_blocks):
            if excluded_name.lower() in block['text'].lower():
                # Skip table of contents entries - look for bold headers
                if block['is_bold'] or block['font_size'] >= 12:
                    # Found an actual excluded section header (not TOC)
                    section_start = i
                    section_end = len(styled_blocks) - 1  # Default to end of document
                    
                    # Find the next header (any header, regardless of style) after this one
                    for j in range(i + 1, len(styled_blocks)):
                        candidate_block = styled_blocks[j]
                        candidate_text = candidate_block['text'].strip()
                        
                        # Check if this looks like a header by comparing with our potential headers
                        if candidate_block['style_signature'] in header_styles:
                            # Additional check: is this text short enough to be a header?
                            if (len(candidate_text) < 100 and 
                                len(candidate_text) > 2 and
                                candidate_text[0].isupper() and
                                (candidate_block['is_bold'] or candidate_block['font_size'] >= 12)):
                                section_end = j - 1
                                break
                    
                    # Check for overlapping sections and avoid duplicates
                    overlaps = False
                    for existing_start, existing_end, _ in excluded_sections:
                        if (section_start <= existing_end and section_end >= existing_start):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        excluded_sections.append((section_start, section_end, block['text']))
    
    return excluded_sections, styled_blocks

def convert_styled_blocks_to_sentences(styled_blocks: List[Dict]) -> List[Dict[str, Any]]:
    """Convert styled blocks back to sentence format for compatibility with existing code"""
    sentences = []
    
    for i, block in enumerate(styled_blocks):
        # Split text into sentences if it's long
        if len(block['text']) > 100:
            # Try to split into sentences
            block_sentences = nltk.sent_tokenize(block['text'])
            for sent in block_sentences:
                if sent.strip():  # Only add non-empty sentences
                    sentences.append({
                        'page_num': block['page'],
                        'content': sent.strip(),
                        'id': len(sentences),
                        'is_excluded': False,
                        'original_block_idx': i,
                        'bbox': block['bbox'],  # Preserve position info
                        'styled_block': block  # Keep reference to original styled block
                    })
        else:
            sentences.append({
                'page_num': block['page'],
                'content': block['text'],
                'id': len(sentences),
                'is_excluded': False,
                'original_block_idx': i,
                'bbox': block['bbox'],  # Preserve position info
                'styled_block': block  # Keep reference to original styled block
            })
    
    return sentences

def mark_excluded_sentences(sentences: List[Dict], excluded_sections: List[Tuple[int, int, str]], styled_blocks: List[Dict]) -> List[Tuple[int, int, str]]:
    """Mark sentences that fall within excluded sections and return sentence-level boundaries"""
    sentence_excluded_sections = []
    
    for start_block, end_block, section_name in excluded_sections:
        # Find sentence indices that correspond to these blocks
        start_sent_idx = None
        end_sent_idx = None
        
        for i, sent in enumerate(sentences):
            block_idx = sent.get('original_block_idx', -1)
            
            # Only mark sentences that are actually within the excluded block range
            if start_block <= block_idx <= end_block:
                if start_sent_idx is None:
                    start_sent_idx = i
                end_sent_idx = i
                sentences[i]['is_excluded'] = True
                sentences[i]['excluded_section'] = section_name
        
        if start_sent_idx is not None and end_sent_idx is not None:
            sentence_excluded_sections.append((start_sent_idx, end_sent_idx, section_name))
    
    return sentence_excluded_sections

def load_and_split_sentences(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]]]:
    """
    Loads a PDF and splits its text content into individual sentences using style-based section detection.
    
    Returns:
        Tuple of (all_sentences, excluded_sections)
    """
    # Use style-based approach to find excluded sections
    excluded_sections, styled_blocks = find_excluded_sections_by_style(pdf_path)
    
    # Convert styled blocks to sentences for compatibility
    all_sentences = convert_styled_blocks_to_sentences(styled_blocks)
    
    # Mark excluded sentences and get sentence-level boundaries
    sentence_excluded_sections = mark_excluded_sentences(all_sentences, excluded_sections, styled_blocks)
    
    return all_sentences, sentence_excluded_sections

def create_sentence_window_nodes(
    all_sentences: List[Dict[str, Any]],
    window_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Creates nodes for each sentence and attaches a "window" of surrounding
    sentences as metadata. Filters out excluded sections and ensures 
    windows don't contain excluded content.
    """
    # Filter out excluded sentences for processing
    included_sentences = [s for s in all_sentences if not s.get('is_excluded', False)]
    
    # Reassign sequential IDs to included sentences for proper indexing
    for i, sentence in enumerate(included_sentences):
        sentence['filtered_id'] = i
    
    nodes = []
    for i, sentence_data in enumerate(included_sentences):
        # Build window carefully, respecting exclusion boundaries
        window_sentences = []
        original_id = sentence_data['id']
        
        # Look backwards for context, stopping at excluded content
        for offset in range(window_size, 0, -1):
            context_id = original_id - offset
            if context_id >= 0 and context_id < len(all_sentences):
                context_sentence = all_sentences[context_id]
                if not context_sentence.get('is_excluded', False):
                    window_sentences.append(context_sentence['content'])
                else:
                    # Stop if we hit excluded content
                    break
        
        # Add the core sentence
        window_sentences.append(sentence_data['content'])
        
        # Look forwards for context, stopping at excluded content
        for offset in range(1, window_size + 1):
            context_id = original_id + offset
            if context_id < len(all_sentences):
                context_sentence = all_sentences[context_id]
                if not context_sentence.get('is_excluded', False):
                    window_sentences.append(context_sentence['content'])
                else:
                    # Stop if we hit excluded content
                    break
        
        window_text = " ".join(window_sentences)
        
        nodes.append({
            'id': i,  # Use sequential ID for included sentences
            'original_id': sentence_data['id'],  # Keep track of original ID
            'content': sentence_data['content'],
            'page_num': sentence_data['page_num'],
            'window': window_text
        })
    return nodes

def count_keyword_hits(text: str, keywords: List[str]) -> Dict[str, int]:
    """
    Counts occurrences of each keyword in the given text (case-insensitive).
    """
    text_lower = text.lower()
    hits = {}
    for keyword in keywords:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            hits[keyword] = count
    return hits

def perform_bm25_retrieval(
    nodes: List[Dict[str, Any]], 
    query: str, 
    top_k: int = 50
) -> List[Dict[str, Any]]:
    """
    Performs BM25-based sparse retrieval on the sentence nodes.
    """
    # Tokenize documents for BM25 (simple whitespace tokenization)
    tokenized_docs = [node['content'].lower().split() for node in nodes]
    
    # Initialize BM25
    bm25 = BM25Okapi(tokenized_docs)
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Return top nodes with their BM25 scores
    return [
        {**nodes[i], 'bm25_score': scores[i]} 
        for i in top_indices if scores[i] > 0
    ]

def create_u_shaped_ranking(items: List[Any]) -> List[Any]:
    """
    Reorders items to address LLM positional bias ("lost in the middle").
    Arranges items in a U-shape, with the most important at the beginning and
    end, and the least important in the middle.
    Input is assumed to be sorted by importance (most to least).
    The resulting pattern is: [1st, 3rd, 5th, ..., 6th, 4th, 2nd].
    """
    if len(items) <= 2:
        return items

    # Split items into odd and even indexed lists
    first_half = items[::2]
    second_half = items[1::2]

    # Reverse the second half and concatenate to create the U-shape
    return first_half + second_half[::-1]


def highlight_pdf_dual_color(
    pdf_path: str,
    output_path: str,
    highlight_nodes: List[Dict[str, Any]],
    all_sentences: List[Dict[str, Any]],
    window_size: int,
    keywords: List[str] = None,
    excluded_sections: List[Tuple[int, int, str]] = None
):
    """
    Opens a PDF, highlights core sentences (bright yellow) and their
    context windows (light yellow), optionally highlights keywords (green),
    and highlights excluded sections (red), then saves the result.
    """
    doc = fitz.open(pdf_path)
    CORE_HIGHLIGHT_COLOR = (1, 1, 0)  # Bright yellow
    CONTEXT_HIGHLIGHT_COLOR = (0.95, 0.95, 0.6)  # Light yellow
    KEYWORD_HIGHLIGHT_COLOR = (0.6, 1, 0.6)  # Light green
    EXCLUDED_HIGHLIGHT_COLOR = (1, 0.6, 0.6)  # Light red

    processed_sentences = set()
    for node_data in highlight_nodes:
        original_id = node_data.get('original_id', node_data['id'])
        
        # Find the original sentence
        target_sentence = None
        for sent in all_sentences:
            if sent['id'] == original_id:
                target_sentence = sent
                break
        
        if target_sentence and not target_sentence.get('is_excluded', False):
            # Highlight context window
            start_idx = max(0, original_id - window_size)
            end_idx = min(len(all_sentences), original_id + window_size + 1)
            
            for i in range(start_idx, end_idx):
                if i == original_id or i in processed_sentences:
                    continue
                
                if i < len(all_sentences) and not all_sentences[i].get('is_excluded', False):
                    sentence_data = all_sentences[i]
                    page_num = sentence_data['page_num']
                    text = sentence_data['content']
                    
                    if page_num < len(doc):
                        page = doc.load_page(page_num)
                        areas = page.search_for(text, quads=True)
                        if areas:
                            annot = page.add_highlight_annot(areas)
                            annot.set_colors(stroke=CONTEXT_HIGHLIGHT_COLOR)
                            annot.update()
                            processed_sentences.add(i)

    for node_data in highlight_nodes:
        original_id = node_data.get('original_id', node_data['id'])
        
        # Find the original sentence for core highlighting
        target_sentence = None
        for sent in all_sentences:
            if sent['id'] == original_id:
                target_sentence = sent
                break
        
        if target_sentence and not target_sentence.get('is_excluded', False):
            page_num = target_sentence['page_num']
            text = target_sentence['content']

            if page_num < len(doc):
                page = doc.load_page(page_num)
                areas = page.search_for(text, quads=True)
                if areas:
                    annot = page.add_highlight_annot(areas)
                    annot.set_colors(stroke=CORE_HIGHLIGHT_COLOR)
                    annot.update()

    # Highlight keywords if provided
    if keywords:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for keyword in keywords:
                # Search for keyword with case-insensitive matching
                # Use regex pattern for case-insensitive search
                areas = page.search_for(keyword, quads=True)
                # Also search for lowercase version to catch more matches
                if not areas:
                    areas = page.search_for(keyword.lower(), quads=True)
                if not areas:
                    areas = page.search_for(keyword.upper(), quads=True)
                if areas:
                    annot = page.add_highlight_annot(areas)
                    annot.set_colors(stroke=KEYWORD_HIGHLIGHT_COLOR)
                    annot.update()

    # Highlight excluded sections in red using position-based highlighting
    if excluded_sections:
        print(f"Highlighting {len(excluded_sections)} excluded sections in red...")
        highlighted_areas = set()  # Track highlighted areas to avoid duplicates
        
        for start_idx, end_idx, section_name in excluded_sections:
            print(f"  Highlighting section '{section_name}' (sentences {start_idx}-{end_idx})")
            excluded_highlights_count = 0
            
            for i in range(start_idx, min(end_idx + 1, len(all_sentences))):
                sentence_data = all_sentences[i]
                
                # Only highlight if this sentence is actually marked as excluded
                if sentence_data.get('is_excluded', False):
                    page_num = sentence_data['page_num']
                    
                    if page_num < len(doc) and 'bbox' in sentence_data:
                        bbox = sentence_data['bbox']
                        # Create a unique identifier for this highlight area
                        area_id = (page_num, tuple(bbox))
                        
                        if area_id not in highlighted_areas:
                            page = doc.load_page(page_num)
                            # Use the exact bounding box from the styled block
                            rect = fitz.Rect(bbox)
                            annot = page.add_highlight_annot(rect)
                            annot.set_colors(stroke=EXCLUDED_HIGHLIGHT_COLOR)
                            annot.update()
                            highlighted_areas.add(area_id)
                            excluded_highlights_count += 1
            
            print(f"    Applied {excluded_highlights_count} position-based highlights for '{section_name}'")

    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    print(f"Highlighted PDF saved to: {output_path}")


def highlight_relevant_content_advanced(
    pdf_path: str,
    question: str,
    output_pdf_path: str,
    top_k_initial: int = 50,
    window_size: int = 2,
    use_bm25: bool = True,
    alpha: float = 0.7
) -> Dict[str, Any]:
    """
    Performs an advanced RAG pipeline combining dense and sparse retrieval.
    Returns structured JSON with core sentences, context windows, keyword hits,
    and U-shaped ranking for addressing LLM positional bias.
    
    Args:
        alpha: Weight for combining dense (semantic) and sparse (BM25) scores.
               Higher values favor semantic similarity.
    """
    print("Loading PDF and splitting into sentences...")
    sentences, excluded_sections = load_and_split_sentences(pdf_path)
    
    if excluded_sections:
        print(f"Found {len(excluded_sections)} excluded sections:")
        for start_idx, end_idx, section_name in excluded_sections:
            sentence_count = end_idx - start_idx + 1
            print(f"  - '{section_name}': {sentence_count} sentences excluded")
        total_excluded = sum(end_idx - start_idx + 1 for start_idx, end_idx, _ in excluded_sections)
        print(f"Total excluded sentences: {total_excluded} out of {len(sentences)}")
    else:
        print("No excluded sections found.")
    
    if not sentences:
        print("No text could be extracted from the PDF.")
        return {
            "standard_ranking": [],
            "u_shaped_ranking": [],
            "document_keyword_summary": {"total_keywords_found": 0, "keyword_counts": {}},
            "metadata": {"total_sentences": 0, "chunks_selected": 0, "window_size": window_size, "bm25_enabled": use_bm25, "semantic_weight": alpha if use_bm25 else 1.0}
        }

    total_sentences = len(sentences)
    # Scale selected sentences logarithmically with sensible bounds
    top_n_final = int(np.clip(12.5 * np.log1p(total_sentences), 12, 200))
    print(f"Document has {total_sentences} sentences. Targeting top {top_n_final} relevant chunks.")

    # Ensure rerank headroom scales with final selection size
    top_k_effective = max(top_k_initial, 3 * top_n_final)
    print(f"Reranking headroom: considering top {top_k_effective} candidates before re-ranking")

    print(f"Creating sentence nodes with a window size of {window_size}...")
    nodes = create_sentence_window_nodes(sentences, window_size)
    
    print("Embedding individual sentences for precise retrieval")
    sentence_embeddings = bi_encoder.encode(
        [node['content'] for node in nodes],
        convert_to_tensor=True,
        show_progress_bar=True
    )

    print("Performing initial semantic search")
    query_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    
    similarities = torch.nn.functional.cosine_similarity(query_embedding, sentence_embeddings)
    semantic_scores = similarities.cpu().numpy()
    
    # Perform BM25 retrieval if enabled
    if use_bm25:
        print("Performing BM25 sparse retrieval")
        bm25_candidates = perform_bm25_retrieval(nodes, question, top_k_effective)
        bm25_scores = np.zeros(len(nodes))
        for candidate in bm25_candidates:
            bm25_scores[candidate['id']] = candidate['bm25_score']
        
        # Normalize scores to [0, 1] range for combination
        if semantic_scores.max() > 0:
            semantic_scores = semantic_scores / semantic_scores.max()
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine scores using weighted average
        combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        print(f"Combined semantic and BM25 scores with alpha={alpha}")
    else:
        combined_scores = semantic_scores
    
    top_k_indices = np.argsort(combined_scores)[::-1][:top_k_effective]
    initial_candidates = [nodes[i] for i in top_k_indices if combined_scores[i] > 0]

    print(f"Re-ranking top {len(initial_candidates)} candidates")
    cross_encoder_inputs = [[question, candidate['window']] for candidate in initial_candidates]
    cross_encoder_scores = cross_encoder.predict(cross_encoder_inputs, show_progress_bar=True)
    
    reranked_candidates = sorted(
        zip(initial_candidates, cross_encoder_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Selecting final {top_n_final} contexts and preparing for highlighting...")
    final_nodes = [candidate for candidate, score in reranked_candidates[:top_n_final]]
    
    print("Counting keyword hits across the document...")
    full_text = " ".join([sentence['content'] for sentence in sentences])
    document_keyword_hits = count_keyword_hits(full_text, KEYWORDS)
    
    json_output = []
    for node in final_nodes:
        chunk_hits = count_keyword_hits(node['window'], KEYWORDS)
        json_output.append({
            "core_sentence": node['content'],
            "context_window": node['window'],
            "page_number": node['page_num'],
            "keyword_hits": chunk_hits
        })
    
    # Create U-shaped ranking for positional bias mitigation
    print("Creating U-shaped ranking to address LLM positional bias...")
    u_shaped_output = create_u_shaped_ranking(json_output.copy())
    
    print(f"Highlighting the {len(final_nodes)} most relevant sentences and their context in the PDF...")
    highlight_pdf_dual_color(pdf_path, output_pdf_path, final_nodes, sentences, window_size, KEYWORDS, excluded_sections)

    # Create excluded content debugging information
    excluded_content_debug = []
    for start_idx, end_idx, section_name in excluded_sections:
        section_sentences = []
        for i in range(start_idx, min(end_idx + 1, len(sentences))):
            if sentences[i].get('is_excluded', False):
                section_sentences.append({
                    "sentence_id": i,
                    "page_number": sentences[i]['page_num'],
                    "content": sentences[i]['content'][:100] + "..." if len(sentences[i]['content']) > 100 else sentences[i]['content'],
                    "full_length": len(sentences[i]['content'])
                })
        
        excluded_content_debug.append({
            "section_name": section_name,
            "start_sentence": start_idx,
            "end_sentence": end_idx,
            "sentence_count": len(section_sentences),
            "page_range": f"{min(s['page_number'] for s in section_sentences) if section_sentences else 'N/A'}-{max(s['page_number'] for s in section_sentences) if section_sentences else 'N/A'}",
            "sentences": section_sentences[:5],  # Only show first 5 for brevity
            "total_sentences_in_section": len(section_sentences)
        })

    return {
        "standard_ranking": json_output,
        "u_shaped_ranking": u_shaped_output,
        "document_keyword_summary": {
            "total_keywords_found": len(document_keyword_hits),
            "keyword_counts": document_keyword_hits
        },
        "excluded_content_debug": excluded_content_debug,
        "metadata": {
            "total_sentences": len(sentences),
            "chunks_selected": len(final_nodes),
            "window_size": window_size,
            "bm25_enabled": use_bm25,
            "semantic_weight": alpha if use_bm25 else 1.0,
            "excluded_sections": [
                {
                    "section_name": section_name,
                    "start_sentence": start_idx,
                    "end_sentence": end_idx,
                    "sentence_count": end_idx - start_idx + 1
                } for start_idx, end_idx, section_name in excluded_sections
            ],
            "total_excluded_sentences": sum(end_idx - start_idx + 1 for start_idx, end_idx, _ in excluded_sections)
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Highlight relevant content in a PDF based on a question.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--question", type=str, required=True, help="The question to ask the document.")
    parser.add_argument("--no-bm25", action="store_true", help="Disable BM25 sparse retrieval (use only semantic search).")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for combining semantic and BM25 scores (0.0-1.0, higher favors semantic).")
    parser.add_argument("--output-pdf", type=str, help="Path to save the output highlighted PDF file.")
    parser.add_argument("--output-json", type=str, help="Path to save the output JSON file.")
    
    args = parser.parse_args()

    pdf_file = args.pdf
    user_question = args.question
    
    if not os.path.exists(pdf_file):
        print(f"Error: The specified PDF file was not found: {pdf_file}")
    else:
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        
        # Use provided output paths or create default ones
        output_pdf_file = args.output_pdf if args.output_pdf else f"highlighted_{base_name}.pdf"
        output_json_file = args.output_json if args.output_json else f"highlighted_{base_name}.json"
        
        # Create output directories if they don't exist
        if output_pdf_file and os.path.dirname(output_pdf_file):
            os.makedirs(os.path.dirname(output_pdf_file), exist_ok=True)
        if output_json_file and os.path.dirname(output_json_file):
            os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

        result = highlight_relevant_content_advanced(
            pdf_path=pdf_file,
            question=user_question,
            output_pdf_path=output_pdf_file,
            use_bm25=not args.no_bm25,
            alpha=args.alpha
        )

        if result and result.get('standard_ranking'):
            print(f"\nSaving structured output to {output_json_file}...")
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("Output saved successfully.")
            
            # Print summary statistics
            metadata = result.get('metadata', {})
            keyword_summary = result.get('document_keyword_summary', {})
            print(f"\nSummary:")
            print(f"- Total sentences processed: {metadata.get('total_sentences', 'N/A')}")
            print(f"- Chunks selected: {metadata.get('chunks_selected', 'N/A')}")
            print(f"- Unique keywords found in document: {keyword_summary.get('total_keywords_found', 0)}")
            if keyword_summary.get('keyword_counts'):
                print("- Keyword counts:", keyword_summary['keyword_counts'])
            print(f"- BM25 hybrid retrieval: {'Enabled' if metadata.get('bm25_enabled') else 'Disabled'}")
        else:
            print("\nNo relevant chunks were found or the PDF could not be processed.")
