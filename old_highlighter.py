# standalone.py
#
# Description:
# A standalone script for an advanced Retrieval-Augmented Generation (RAG) pipeline
# to analyze a single PDF document. This script serves as the v3 prototype for the
# OSPAR climate parser.
#
# Features:
# 1. Intelligent PDF preprocessing to identify and exclude sections like
#    "Executive Summary" and "Conclusion" based on text style.
# 2. Semantic chunking with size limits to create thematically coherent text segments.
# 3. Accurate page tracking using character offsets and metadata.
# 4. Retrieval of relevant passages ("highlighting") using a vector store.
# 5. Detailed, grounded summary generation using an LLM.
# 6. Structured output with summary, highlights, and keyword hit counts.
# 7. Optional visual highlighting of retrieved passages in a new PDF.
#
# Dependencies:
# pip install pymupdf langchain langchain-community langchain-experimental anthropic \
#             sentence-transformers faiss-cpu numpy

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import pymupdf  # Fitz
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import anthropic

# --- Configuration ---
# Load API key from .keys file and set it as an environment variable if found.
KEYS_FILE = ".keys"
if os.path.exists(KEYS_FILE):
    with open(KEYS_FILE, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# It is recommended to set the API key as an environment variable
# for security, or place it in a .keys file.
# Example .keys file content: ANTHROPIC_API_KEY=your_api_key
try:
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
except KeyError:
    print("Error: ANTHROPIC_API_KEY environment variable not set, and .keys file not found or configured.")
    exit(1)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Updated to use Claude Sonnet 4 - the latest available model
LLM_MODEL_NAME = "claude-sonnet-4-20250514"
# Keywords for hit count analysis
CLIMATE_KEYWORDS = [
    "climate change", "ocean acidification", "sea level rise", "warming",
    "temperature increase", "carbon dioxide", "co2", "greenhouse gas",
    "extreme weather", "marine heatwave", "hypoxia", "anoxia"
]

@dataclass
class ChunkMetadata:
    """Metadata for tracking chunk origins"""
    start_page: int
    end_page: int
    char_start: int
    char_end: int

class Parser:
    """
    A class encapsulating the RAG pipeline for PDF document analysis.
    """
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initializes the parser, loading the embedding model.
        """
        print(f"Initializing parser and loading embedding model: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.vector_store = None
        self.original_pages = []
        self.page_char_offsets = []  # Track character offsets for each page
        self.full_text_with_metadata = ""  # Full text with page markers
        self._last_processing_data = None  # Store processing data for highlighting
        print("Initialization complete.")

    def _get_font_statistics(self, doc: pymupdf.Document) -> Tuple[float, float]:
        """
        Analyzes the document to find the most common font size (body text)
        and a likely heading font size.

        Returns:
            A tuple of (most_common_size, heading_size_threshold).
        """
        sizes = {}
        for page in doc:
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES)["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            size = round(span["size"])
                            sizes[size] = sizes.get(size, 0) + len(span["text"])

        if not sizes:
            return 10.0, 12.0  # Default values if no text is found

        # Most common font size is likely the body text
        most_common_size = max(sizes, key=sizes.get)

        # A simple heuristic for heading size: slightly larger than body text.
        # This can be tuned for more specific document formats.
        heading_size_threshold = most_common_size + 1.5

        return most_common_size, heading_size_threshold

    def preprocess_pdf(self, pdf_path: str) -> Tuple[str, List[str], List[str], Dict]:
        """
        Loads a PDF, removes specified sections based on heading styles,
        filters out headers/footers, tracks page boundaries and character offsets.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A tuple containing:
            - cleaned text of the document
            - list of excluded section text
            - list of header/footer text that was filtered
            - metadata dictionary with page information
        """
        print(f"Preprocessing PDF: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        
        # Store original page text for later reference
        self.original_pages = []
        
        most_common_size, heading_size_threshold = self._get_font_statistics(doc)
        print(f"Font analysis: Body text size ~{most_common_size}pt, Heading threshold ~{heading_size_threshold}pt")

        full_text_parts = []
        excluded_sections = []
        headers_footers = []
        sections_to_exclude = ["executive summary", "summary", "conclusion", "recommendations", "references", "acknowledgements"]
        exclude_mode = False
        
        # Track character offsets and page boundaries
        char_offset = 0
        self.page_char_offsets = []
        
        for page_num, page in enumerate(doc):
            page_start_offset = char_offset
            page_text_parts = []
            
            # Store original page text
            original_page_text = page.get_text()
            self.original_pages.append(original_page_text)
            
            # Process page with style-aware extraction
            page_height = page.rect.height
            y_header_limit = page_height * 0.10
            y_footer_limit = page_height * 0.90

            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES)["blocks"]
            sorted_blocks = sorted(blocks, key=lambda b: b.get('bbox', (0, 0, 0, 0))[1])

            for block in sorted_blocks:
                if "lines" not in block:
                    continue

                block_bbox = block.get("bbox", (0, 0, 0, 0))
                is_header_footer = (block_bbox[1] < y_header_limit or block_bbox[3] > y_footer_limit)

                block_text = ""
                is_heading = False
                
                if len(block["lines"]) <= 2:
                    if block["lines"] and block["lines"][0]["spans"]:
                        first_span = block["lines"][0]["spans"][0]
                        if first_span["size"] >= heading_size_threshold:
                            is_heading = True

                # Extract text with consistent formatting
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    block_text += line_text.strip() + " "
                
                block_text = block_text.strip()

                if is_header_footer:
                    headers_footers.append(block_text)
                    continue

                if is_heading:
                    if any(sec in block_text.lower() for sec in sections_to_exclude):
                        print(f"  - Found excluded section heading on page {page_num + 1}: '{block_text[:50]}...'")
                        exclude_mode = True
                        excluded_sections.append(block_text)
                        continue
                    else:
                        exclude_mode = False

                if exclude_mode:
                    excluded_sections.append(block_text)
                else:
                    if block_text:
                        page_text_parts.append(block_text)
                        char_offset += len(block_text) + 2  # +2 for paragraph separator
            
            # Combine page text parts
            page_text = "\n\n".join(page_text_parts)
            if page_text:
                full_text_parts.append(page_text)
            
            # Record page boundaries
            self.page_char_offsets.append({
                'page_num': page_num + 1,
                'start_offset': page_start_offset,
                'end_offset': char_offset,
                'text_length': len(page_text)
            })
        
        doc.close()
        
        full_text = "\n\n".join(full_text_parts)
        self.full_text_with_metadata = full_text
        
        metadata = {
            'total_pages': len(self.page_char_offsets),
            'total_chars': char_offset,
            'page_offsets': self.page_char_offsets
        }
        
        print(f"Preprocessing finished. Excluded {len(excluded_sections)} sections, {len(headers_footers)} headers/footers")
        print(f"Document has {metadata['total_pages']} pages, {metadata['total_chars']} characters")
        
        return full_text, excluded_sections, headers_footers, metadata

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Splits the preprocessed text into semantic chunks with size limits and page metadata.

        Args:
            text: The full text to be chunked.
            chunk_size: Maximum chunk size in characters (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)

        Returns:
            A list of LangChain Document objects with metadata
        """
        print(f"Chunking text with size limit {chunk_size} chars and {chunk_overlap} char overlap...")
        
        # First, try semantic chunking with size constraints
        docs = self._semantic_chunk_with_limits(text, chunk_size)
        
        # If semantic chunking produces too-large chunks, fall back to recursive splitting
        max_chunk_size = max(len(doc.page_content) for doc in docs) if docs else 0
        if max_chunk_size > chunk_size * 1.5:  # Allow 50% overflow
            print(f"  Semantic chunks too large (max: {max_chunk_size}). Using recursive splitter...")
            docs = self._recursive_chunk(text, chunk_size, chunk_overlap)
        
        # Add page metadata to each chunk
        docs_with_metadata = []
        for doc in docs:
            chunk_text = doc.page_content
            metadata = self._find_chunk_pages(chunk_text, text)
            doc.metadata.update(metadata)
            docs_with_metadata.append(doc)
            
            # Debug output for first few chunks
            if len(docs_with_metadata) <= 3:
                print(f"  Chunk {len(docs_with_metadata)}: {len(chunk_text)} chars, pages {metadata['start_page']}-{metadata['end_page']}")
        
        print(f"Created {len(docs_with_metadata)} chunks")
        print(f"Chunk sizes: min={min(len(d.page_content) for d in docs_with_metadata)}, "
              f"max={max(len(d.page_content) for d in docs_with_metadata)}, "
              f"avg={sum(len(d.page_content) for d in docs_with_metadata) // len(docs_with_metadata)}")
        
        return docs_with_metadata

    def _semantic_chunk_with_limits(self, text: str, max_chunk_size: int) -> List[Document]:
        """Semantic chunking with size constraints"""
        try:
            # Use semantic chunker with lower percentile for more aggressive splitting
            text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75  # Lower percentile = more splits
            )
            docs = text_splitter.create_documents([text])
            
            # Post-process to enforce size limits
            limited_docs = []
            for doc in docs:
                if len(doc.page_content) <= max_chunk_size:
                    limited_docs.append(doc)
                else:
                    # Split oversized chunks
                    sub_chunks = self._split_large_chunk(doc.page_content, max_chunk_size)
                    for sub_chunk in sub_chunks:
                        limited_docs.append(Document(page_content=sub_chunk))
            
            return limited_docs
        except Exception as e:
            print(f"  Semantic chunking failed: {e}. Falling back to recursive splitter.")
            return self._recursive_chunk(text, max_chunk_size, 200)

    def _recursive_chunk(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Fallback recursive character text splitting"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        return text_splitter.create_documents([text])

    def _split_large_chunk(self, text: str, max_size: int) -> List[str]:
        """Split a large chunk into smaller pieces at sentence boundaries"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_size]]  # Last resort: hard cut

    def _find_chunk_pages(self, chunk_text: str, full_text: str) -> Dict[str, Any]:
        """Find which pages a chunk spans using character offsets"""
        # Find chunk position in full text
        chunk_start = full_text.find(chunk_text[:100])  # Use first 100 chars for matching
        if chunk_start == -1:
            # Try with normalized whitespace
            normalized_chunk = ' '.join(chunk_text[:100].split())
            normalized_full = ' '.join(full_text.split())
            chunk_start = normalized_full.find(normalized_chunk)
            if chunk_start == -1:
                return {'start_page': -1, 'end_page': -1, 'char_start': -1, 'char_end': -1}
        
        chunk_end = chunk_start + len(chunk_text)
        
        # Find pages that contain this chunk
        start_page = -1
        end_page = -1
        
        for page_info in self.page_char_offsets:
            if chunk_start >= page_info['start_offset'] and chunk_start < page_info['end_offset']:
                start_page = page_info['page_num']
            if chunk_end > page_info['start_offset'] and chunk_end <= page_info['end_offset']:
                end_page = page_info['page_num']
                
        # Handle edge cases
        if start_page == -1 and self.page_char_offsets:
            if chunk_start < self.page_char_offsets[0]['start_offset']:
                start_page = 1
        if end_page == -1 and self.page_char_offsets:
            if chunk_end > self.page_char_offsets[-1]['end_offset']:
                end_page = self.page_char_offsets[-1]['page_num']
        
        # If still not found, use the page where most of the chunk is
        if start_page == -1 or end_page == -1:
            for i, page_info in enumerate(self.page_char_offsets):
                overlap_start = max(chunk_start, page_info['start_offset'])
                overlap_end = min(chunk_end, page_info['end_offset'])
                if overlap_end > overlap_start:
                    if start_page == -1:
                        start_page = page_info['page_num']
                    end_page = page_info['page_num']
        
        return {
            'start_page': start_page,
            'end_page': end_page,
            'char_start': chunk_start,
            'char_end': chunk_end
        }

    def build_vector_store(self, docs: List[Document]):
        """
        Builds an in-memory FAISS vector store from the document chunks.
        """
        print("Building FAISS vector store...")
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        print("Vector store built successfully.")

    def retrieve_passages(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant passages for a given query from the vector store.
        Uses metadata for accurate page tracking.

        Args:
            query: The user's question.
            top_k: The number of passages to retrieve (default: 15)

        Returns:
            A list of dictionaries containing retrieved passages with metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not built. Please run build_vector_store first.")

        print(f"Retrieving top {top_k} passages for query: '{query}'")
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

        highlights = []
        for doc, score in results_with_scores:
            # Use metadata if available
            start_page = doc.metadata.get('start_page', -1)
            end_page = doc.metadata.get('end_page', -1)
            
            # Determine display page number
            if start_page == end_page and start_page != -1:
                page_display = f"Page {start_page}"
            elif start_page != -1 and end_page != -1:
                page_display = f"Pages {start_page}-{end_page}"
            else:
                page_display = "Page unknown"
            
            highlights.append({
                "text": doc.page_content,
                "score": float(score),
                "page_number": start_page,
                "page_range": page_display,
                "metadata": doc.metadata
            })
        
        print(f"Retrieved {len(highlights)} passages.")
        # Debug: print page numbers for first few
        for i, h in enumerate(highlights[:3]):
            print(f"  Highlight {i+1}: {h['page_range']}, score: {h['score']:.3f}")
        
        return highlights

    def generate_summary(self, query: str, highlights: List[Dict[str, Any]]) -> str:
        """
        Generates a detailed summary using an LLM, grounded in the retrieved passages.

        Args:
            query: The user's question.
            highlights: The list of retrieved passages from retrieve_passages.

        Returns:
            A string containing the generated summary.
        """
        if not highlights:
            return "No relevant information was found in the document to answer this question."

        print("Generating summary from retrieved passages...")
        context = "\n\n---\n\n".join([h['text'] for h in highlights])

        # Advanced Chain-of-Thought style prompt for more comprehensive analysis
        prompt = f"""
You are an expert marine science analyst tasked with summarizing technical information from OSPAR reports. Based *only* on the provided context below, answer the user's question comprehensively.

Follow these steps:
1. First, carefully read through ALL the provided context passages and the user's question to identify all relevant facts, findings, data points, and explicitly mentioned gaps or recommendations.
2. Second, synthesize these points into a coherent, detailed, and comprehensive analysis. Do not use generic phrases like "The document discusses..." or "The text mentions...". Instead, state the findings directly and authoritatively. For example, instead of "The document states that temperatures are rising," write "Temperatures in the OSPAR Maritime Area have risen by X°C...".
3. Be thorough - with {len(highlights)} passages provided, you have substantial context to work with. Use it all to provide a complete picture.
4. Ensure your summary is objective and strictly derived from the provided text. Do not add any information not present in the context. If the context is insufficient to answer the question fully, state what is available and note the limitations.
5. If the provided context doesn't address the specific question asked, clearly explain what the context does contain and why it's not relevant to the question.

Answer in plain text, with a maximum of 250-500 words, without subheadings or sections. Be scientific and concise.

CONTEXT ({len(highlights)} passages):
---
{context}
---

QUESTION: {query}

COMPREHENSIVE ANALYSIS:
"""
        try:
            message = self.anthropic_client.messages.create(
                model=LLM_MODEL_NAME,
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            summary = message.content[0].text
            print("Summary generated successfully.")
            return summary
        except Exception as e:
            print(f"An error occurred during LLM call: {e}")
            return "Error: Could not generate summary."

    def calculate_keyword_hits(self, text: str) -> Dict[str, int]:
        """
        Calculates the number of occurrences of predefined keywords in the text.

        Args:
            text: The text to search within.

        Returns:
            A dictionary with keywords as keys and their counts as values.
        """
        hits = {}
        for keyword in CLIMATE_KEYWORDS:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
            if count > 0:
                hits[keyword] = count
        return hits

    def process_document(self, pdf_path: str, query: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline on a single document and query.

        Args:
            pdf_path: Path to the PDF file.
            query: The question to ask about the document.
            chunk_size: Maximum size for text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)

        Returns:
            A dictionary containing the full analysis result.
        """
        # 1. Preprocess PDF with metadata tracking
        clean_text, excluded_sections, headers_footers, metadata = self.preprocess_pdf(pdf_path)
        if not clean_text.strip():
            return {"error": "No content could be extracted from the PDF after preprocessing."}

        # 2. Chunk Text with size limits and metadata
        docs = self.chunk_text(clean_text, chunk_size, chunk_overlap)

        # 3. Build Vector Store
        self.build_vector_store(docs)

        # 4. Retrieve Passages with improved page tracking
        highlights = self.retrieve_passages(query)

        # 5. Generate Summary
        summary = self.generate_summary(query, highlights)

        # 6. Calculate Keyword Hits
        keyword_hits = self.calculate_keyword_hits(clean_text)

        # 7. Assemble final result with comprehensive metadata
        result = {
            "query": query,
            "summary": summary,
            "highlights": highlights,
            "keyword_hits": keyword_hits,
            "preprocessing_info": {
                "excluded_sections_count": len(excluded_sections),
                "headers_footers_count": len(headers_footers),
                "total_excluded_chars": sum(len(s) for s in excluded_sections),
                "total_header_footer_chars": sum(len(s) for s in headers_footers)
            },
            "chunking_info": {
                "chunk_size_target": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(docs),
                "chunk_size_stats": {
                    "min": min(len(d.page_content) for d in docs),
                    "max": max(len(d.page_content) for d in docs),
                    "avg": sum(len(d.page_content) for d in docs) // len(docs)
                }
            },
            "metadata": {
                "pdf_path": pdf_path,
                "total_pages": metadata['total_pages'],
                "total_processed_chars": len(clean_text),
                "retrieved_chunks": len(highlights),
                "retrieval_coverage": f"{len(highlights)}/{len(docs)} chunks ({len(highlights)/len(docs)*100:.1f}%)"
            }
        }
        
        # Store processing data for highlighting
        self._last_processing_data = {
            "all_chunks": docs,
            "excluded_sections": excluded_sections,
            "headers_footers": headers_footers,
            "metadata": metadata
        }
        
        return result

    def create_comprehensive_highlighted_pdf(self, original_pdf_path: str, highlights: List[Dict[str, Any]], 
                                           all_chunks: List[Document], excluded_text: List[str], 
                                           output_pdf_path: str):
        """
        Creates a comprehensive PDF with different types of highlighting to show the full processing pipeline.

        Color Legend:
        - GREEN: Retrieved passages (most relevant to query)
        - YELLOW: Other processed chunks (chunked but not retrieved)
        - RED: Filtered content (excluded sections, headers, footers)
        - PURPLE: Climate keyword hits

        Args:
            original_pdf_path: Path to the original PDF.
            highlights: The list of retrieved passages.
            all_chunks: All chunks that were processed and added to vector store.
            excluded_text: Text that was excluded during preprocessing.
            output_pdf_path: Path to save the new highlighted PDF.
        """
        print(f"Creating comprehensive highlighted PDF: {output_pdf_path}")
        doc = pymupdf.open(original_pdf_path)
        
        # Color definitions
        COLORS = {
            'retrieved': (0, 1, 0),      # Green - retrieved passages
            'chunked': (1, 1, 0),       # Yellow - chunked but not retrieved
            'filtered': (1, 0, 0),      # Red - filtered out content
            'keywords': (0.5, 0, 1)     # Purple - keyword hits
        }
        
        # Track what we've already highlighted to avoid overlaps
        highlighted_areas = set()
        
        # 1. Highlight retrieved passages (GREEN - highest priority)
        print("  - Highlighting retrieved passages...")
        for highlight in highlights:
            self._highlight_text_in_pdf(doc, highlight["text"], COLORS['retrieved'], 
                                      highlight.get("page_number"), highlighted_areas)
        
        # 2. Highlight other processed chunks (YELLOW)
        print("  - Highlighting other chunked content...")
        retrieved_texts = {h["text"] for h in highlights}
        for chunk in all_chunks:
            if chunk.page_content not in retrieved_texts:
                self._highlight_text_in_pdf(doc, chunk.page_content, COLORS['chunked'], 
                                          chunk.metadata.get('start_page'), highlighted_areas)
        
        # 3. Highlight filtered content (RED)
        print("  - Highlighting filtered content...")
        for excluded in excluded_text:
            if excluded.strip():
                self._highlight_text_in_pdf(doc, excluded, COLORS['filtered'], 
                                          None, highlighted_areas)
        
        # 4. Highlight keyword occurrences (PURPLE)
        print("  - Highlighting climate keywords...")
        self._highlight_keywords_in_pdf(doc, CLIMATE_KEYWORDS, COLORS['keywords'], highlighted_areas)
        
        # 5. Add legend to first page
        self._add_legend_to_pdf(doc, COLORS)
        
        doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
        doc.close()
        print("Comprehensive highlighted PDF saved.")
    
    def _highlight_text_in_pdf(self, doc: pymupdf.Document, text: str, color: tuple, 
                              page_number: Optional[int], highlighted_areas: set):
        """
        Helper method to highlight text in PDF with overlap detection.
        """
        if not text.strip():
            return
            
        # Search strategy: try multiple text lengths for better matching
        search_texts = [
            text[:200],  # First 200 chars
            text[:100],  # First 100 chars  
            text[:50],   # First 50 chars
        ]
        
        pages_to_search = [doc[page_number - 1]] if page_number and page_number > 0 else doc
        
        for search_text in search_texts:
            found = False
            for page in pages_to_search:
                areas = page.search_for(search_text.strip())
                if areas:
                    for area in areas:
                        area_key = (page.number, round(area.x0), round(area.y0), round(area.x1), round(area.y1))

                        if area_key not in highlighted_areas:
                            highlight_annot = page.add_highlight_annot(area)
                            highlight_annot.set_colors(stroke=color)
                            highlight_annot.update()
                            highlighted_areas.add(area_key)
                            found = True 

                    if found:
                        break
            if found:
                break 
    
    def _add_legend_to_pdf(self, doc: pymupdf.Document, colors: Dict[str, tuple]):
        """
        Add a color legend to the first page of the PDF.
        """
        first_page = doc[0]
        
        # Legend text
        legend_text = [
            "RAG Processing Legend:",
            "🟢 Green: Retrieved passages (most relevant)",
            "🟡 Yellow: Chunked but not retrieved", 
            "🔴 Red: Filtered content (excluded)",
            "🟣 Purple: Climate keywords"
        ]
        
        # Add legend box in top-right corner
        page_rect = first_page.rect
        legend_rect = pymupdf.Rect(page_rect.width - 250, 10, page_rect.width - 10, 120)
        
        # Add white background
        legend_bg = first_page.add_rect_annot(legend_rect)
        legend_bg.set_colors(fill=(1, 1, 1), stroke=(0, 0, 0))
        legend_bg.update()
        
        # Add legend text
        y_pos = 25
        for line in legend_text:
            text_rect = pymupdf.Rect(page_rect.width - 240, y_pos, page_rect.width - 20, y_pos + 12)
            first_page.insert_text((text_rect.x0, text_rect.y0 + 8), line, 
                                 fontsize=8, color=(0, 0, 0))
            y_pos += 15

    def _highlight_keywords_in_pdf(self, doc: pymupdf.Document, keywords: List[str], 
                                     color: tuple, highlighted_areas: set):
        """
        Highlight keyword occurrences throughout the document.
        """
        for page in doc:
            page_text = page.get_text()
            for keyword in keywords:
                matches = re.finditer(r'\b' + re.escape(keyword) + r'\b', 
                                      page_text, re.IGNORECASE)
                for match in matches:
                    areas = page.search_for(match.group())
                    for area in areas:
                        area_key = (page.number, round(area.x0), round(area.y0), 
                                    round(area.x1), round(area.y1))
                        
                        if area_key not in highlighted_areas:
                            highlight_annot = page.add_highlight_annot(area)
                            highlight_annot.set_colors(stroke=color)
                            highlight_annot.update()
                            highlighted_areas.add(area_key)


if __name__ == '__main__':
    import argparse
    
    # Set up command line arguments
    parser_args = argparse.ArgumentParser(description='RAG-based PDF analysis for OSPAR documents')
    parser_args.add_argument('pdf_path', help='Path to the PDF file to analyze')
    parser_args.add_argument('question', help='Question to ask about the document')
    parser_args.add_argument('--chunk-size', type=int, default=1000,
                           help='Maximum chunk size in characters (default: 1000)')
    parser_args.add_argument('--chunk-overlap', type=int, default=200,
                           help='Overlap between chunks in characters (default: 200)')
    parser_args.add_argument('--no-highlight', action='store_true', 
                           help='Skip creating highlighted PDF output')
    parser_args.add_argument('--output-dir', default='./outputs', 
                           help='Directory to save outputs (default: ./outputs)')
    
    args = parser_args.parse_args()
    
    # Validate PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing PDF: {args.pdf_path}")
    print(f"Question: {args.question}")
    print(f"Chunk size: {args.chunk_size} chars")
    print(f"Chunk overlap: {args.chunk_overlap} chars")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Initialize and run the parser
    parser = Parser()
    analysis_result = parser.process_document(
        pdf_path=args.pdf_path, 
        query=args.question,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Save JSON results
    pdf_basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
    json_output_path = os.path.join(args.output_dir, f"{pdf_basename}_analysis.json")
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    # Print summary and key stats
    print("\n--- ANALYSIS COMPLETE ---\n")
    print(f"Summary:\n{analysis_result.get('summary', 'No summary generated')}\n")
    print(f"Chunking Statistics:")
    if 'chunking_info' in analysis_result:
        stats = analysis_result['chunking_info']['chunk_size_stats']
        print(f"  - Total chunks: {analysis_result['chunking_info']['total_chunks']}")
        print(f"  - Chunk sizes: min={stats['min']}, max={stats['max']}, avg={stats['avg']}")
    print(f"\nKeyword Hits: {analysis_result.get('keyword_hits', {})}")
    print(f"\nFull results saved to: {json_output_path}")

    # Create a visually highlighted PDF (unless disabled)
    if not args.no_highlight and "error" not in analysis_result:
        highlighted_pdf_path = os.path.join(args.output_dir, f"highlighted_{os.path.basename(args.pdf_path)}")
        
        # Use comprehensive highlighting with stored processing data
        if parser._last_processing_data:
            parser.create_comprehensive_highlighted_pdf(
                original_pdf_path=args.pdf_path,
                highlights=analysis_result["highlights"],
                all_chunks=parser._last_processing_data["all_chunks"],
                excluded_text=parser._last_processing_data["excluded_sections"] + parser._last_processing_data["headers_footers"],
                output_pdf_path=highlighted_pdf_path
            )
            print(f"Highlighted PDF saved to: {highlighted_pdf_path}")
        else:
            print("Warning: No processing data available, skipping PDF highlighting")