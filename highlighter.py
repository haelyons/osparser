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
from typing import List, Dict, Any

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK's 'punkt' tokenizer...")
    nltk.download('punkt')
    # nltk.download('punkt/tab')

# For accuracy, we select top-tier models from the MTEB leaderboard.
# NV-Embed-v2 and BAAI/bge-large-en-v1.5 are SOTA embedding models.
# We will use bge-large-en-v1.5 for this implementation.
BI_ENCODER_MODEL = 'BAAI/bge-large-en-v1.5'

# For the re-ranking step, we use a powerful cross-encoder.
# mixedbread-ai/mxbai-rerank-large-v1 is a top open-source performer.
CROSS_ENCODER_MODEL = 'mixedbread-ai/mxbai-rerank-large-v1'

# Keywords for sparse retrieval and highlighting
KEYWORDS = [
    "climate change", "ocean acidification", "sea level rise", "warming",
    "temperature increase", "carbon dioxide", "co2", "greenhouse gas",
    "extreme weather", "marine heatwave", "hypoxia", "anoxia"
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# The bi-encoder is used to create vector embeddings for sentences.
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device=device)
# The cross-encoder is used to re-rank the retrieved sentences for maximum relevance.
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
print("Models initialized successfully.")

def load_and_split_sentences(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Loads a PDF and splits its text content into individual sentences,
    keeping track of their page number and a unique ID.
    """
    doc = fitz.open(pdf_path)
    all_sentences = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        sentences = nltk.sent_tokenize(text)
        for sentence_text in sentences:
            cleaned_sentence = sentence_text.replace('\n', ' ').strip()
            if cleaned_sentence:
                all_sentences.append({
                    'page_num': page_num,
                    'content': cleaned_sentence,
                    'id': len(all_sentences)
                })
    doc.close()
    return all_sentences

def create_sentence_window_nodes(
    all_sentences: List[Dict[str, Any]],
    window_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Creates nodes for each sentence and attaches a "window" of surrounding
    sentences as metadata.
    """
    nodes = []
    for i, sentence_data in enumerate(all_sentences):
        start_index = max(0, i - window_size)
        end_index = min(len(all_sentences), i + window_size + 1)
        window_text = " ".join(
            [all_sentences[j]['content'] for j in range(start_index, end_index)]
        )
        nodes.append({
            'id': sentence_data['id'],
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
    keywords: List[str] = None
):
    """
    Opens a PDF, highlights core sentences (bright yellow) and their
    context windows (light yellow), and optionally highlights keywords (green),
    then saves the result.
    """
    doc = fitz.open(pdf_path)
    CORE_HIGHLIGHT_COLOR = (1, 1, 0)  # Bright yellow
    CONTEXT_HIGHLIGHT_COLOR = (0.95, 0.95, 0.6)  # Light yellow
    KEYWORD_HIGHLIGHT_COLOR = (0.6, 1, 0.6)  # Light green

    processed_sentences = set()
    for node_data in highlight_nodes:
        core_id = node_data['id']
        start_idx = max(0, core_id - window_size)
        end_idx = min(len(all_sentences), core_id + window_size + 1)
        
        for i in range(start_idx, end_idx):
            if i == core_id or i in processed_sentences:
                continue
            
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
        core_id = node_data['id']
        sentence_data = all_sentences[core_id]
        page_num = sentence_data['page_num']
        text = sentence_data['content']

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
    sentences = load_and_split_sentences(pdf_path)
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
    
    # Count keyword hits for the entire document
    print("Counting keyword hits across the document...")
    full_text = " ".join([sentence['content'] for sentence in sentences])
    document_keyword_hits = count_keyword_hits(full_text, KEYWORDS)
    
    # Create standard JSON output
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
    highlight_pdf_dual_color(pdf_path, output_pdf_path, final_nodes, sentences, window_size, KEYWORDS)

    return {
        "standard_ranking": json_output,
        "u_shaped_ranking": u_shaped_output,
        "document_keyword_summary": {
            "total_keywords_found": len(document_keyword_hits),
            "keyword_counts": document_keyword_hits
        },
        "metadata": {
            "total_sentences": len(sentences),
            "chunks_selected": len(final_nodes),
            "window_size": window_size,
            "bm25_enabled": use_bm25,
            "semantic_weight": alpha if use_bm25 else 1.0
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
        if output_pdf_file:
            os.makedirs(os.path.dirname(output_pdf_file), exist_ok=True)
        if output_json_file:
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
