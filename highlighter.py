# pip install PyMuPDF "sentence-transformers>=2.2.0" numpy torch nltk argparse json 

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch
import nltk
import os
import argparse
import json
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

def highlight_pdf_dual_color(
    pdf_path: str,
    output_path: str,
    highlight_nodes: List[Dict[str, Any]],
    all_sentences: List[Dict[str, Any]],
    window_size: int
):
    """
    Opens a PDF, highlights core sentences (bright yellow) and their
    context windows (light yellow), and saves the result.
    """
    doc = fitz.open(pdf_path)
    CORE_HIGHLIGHT_COLOR = (1, 1, 0)
    CONTEXT_HIGHLIGHT_COLOR = (0.95, 0.95, 0.6)

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

    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    print(f"Highlighted PDF saved to: {output_path}")


def highlight_relevant_content_advanced(
    pdf_path: str,
    question: str,
    output_pdf_path: str,
    top_k_initial: int = 50,
    window_size: int = 2
) -> List[Dict[str, str]]:
    """
    Performs an advanced RAG pipeline and returns a structured JSON
    with core sentences and their context windows.
    """
    print("Loading PDF and splitting into sentences...")
    sentences = load_and_split_sentences(pdf_path)
    if not sentences:
        print("No text could be extracted from the PDF.")
        return []

    total_sentences = len(sentences)
    # Scale selected sentences logarithmically with sensible bounds
    top_n_final = int(np.clip(14 * np.log1p(total_sentences), 12, 200))
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
    
    top_k_indices = torch.topk(similarities, k=min(top_k_effective, len(nodes))).indices
    initial_candidates = [nodes[i] for i in top_k_indices.cpu()]

    print(f"Re-ranking top {len(initial_candidates)} candidates")
    cross_encoder_inputs = [[question, candidate['content']] for candidate in initial_candidates]
    cross_encoder_scores = cross_encoder.predict(cross_encoder_inputs, show_progress_bar=True)
    
    reranked_candidates = sorted(
        zip(initial_candidates, cross_encoder_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Selecting final {top_n_final} contexts and preparing for highlighting...")
    final_nodes = [candidate for candidate, score in reranked_candidates[:top_n_final]]
    
    json_output = [
        {
            "core_sentence": node['content'],
            "context_window": node['window'],
            "page_number": node['page_num']
        } 
        for node in final_nodes
    ]

    print(f"Highlighting the {len(final_nodes)} most relevant sentences and their context in the PDF...")
    highlight_pdf_dual_color(pdf_path, output_pdf_path, final_nodes, sentences, window_size)

    return json_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Highlight relevant content in a PDF based on a question.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--question", type=str, required=True, help="The question to ask the document.")
    
    args = parser.parse_args()

    pdf_file = args.pdf
    user_question = args.question
    
    if not os.path.exists(pdf_file):
        print(f"Error: The specified PDF file was not found: {pdf_file}")
    else:
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        output_pdf_file = f"highlighted_{base_name}.pdf"
        output_json_file = f"highlighted_{base_name}.json"

        relevant_chunks_json = highlight_relevant_content_advanced(
            pdf_path=pdf_file,
            question=user_question,
            output_pdf_path=output_pdf_file
        )

        if relevant_chunks_json:
            print(f"\nSaving structured output to {output_json_file}...")
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(relevant_chunks_json, f, indent=2, ensure_ascii=False)
            print("Output saved successfully.")
        else:
            print("\nNo relevant chunks were found or the PDF could not be processed.")
