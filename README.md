## OSPAR Climate Parser v3

**RAG pipeline for extracting and analyzing climate-related information from OSPAR marine assessment documents.**

## Overview

This tool uses semantic search (BAAI/bge-large-en-v1.5), sparse retrieval / keyword search (BM25), and cross-encoder reranking (mxbai-rerank-large-v1) to identify relevant content in PDF documents, then generates summaries and relevance judgments using Claude.

## Setup

```bash
# Create and activate virtual environment
python -m venv rag-env
source rag-env/bin/activate  # Windows: rag-env\Scripts\activate

# Install dependencies
pip install PyMuPDF "sentence-transformers>=2.2.0" numpy torch nltk rank-bm25 anthropic PyYAML

# Configure API key (for summarization and judging stages)
echo "ANTHROPIC_API_KEY=your_key_here" > .keys
```

## Quick Start

```bash
# Configure your analysis run
cp config/config_template.yaml config/config.yaml
# Edit config/config.yaml with your questions, keywords, and run metadata

# Stage 1: Highlight relevant content in PDFs
python batch_process.py

# Stage 2: Generate summaries from highlighted content
python summarise.py results/results_template.csv outputs/{run_name}_{date}

# Stage 3: Judge relevance of summaries
python batch_judge.py
```

## Configuration

All analysis runs are configured via `config/config.yaml`:

```yaml
run:
  name: "ecosystem_services_analysis"  # Identifier for this analysis run
  date: "2025-10-19"                   # Date of analysis (YYYY-MM-DD)
  description: "Analysis focused on ecosystem services approaches and valuation"

questions:
  - "What are the approaches of this assessment on ecosystem services related to the topic examined in the assessment"
  - "How is this assessment valuing ecosystem services is this assessment, if at all?"

keywords:  # Terms for sparse retrieval (BM25)
  - "ecosystem services"
  - "valuation"
  - "economic value"
  - "natural capital"

excluded_sections:  # Document sections to skip during processing
  - "Key Message"
  - "Executive Summary"
  - "Conclusion"
  - "Bibliography"
  - "References"

paths:
  source_dir: "sources"                                    # Input PDFs
  output_base_dir: "outputs"                               # Base output directory
  csv_template: "results/analysis_010925_v3_clean.csv"    # CSV listing PDFs to process
```

Each run creates a separate output directory (`outputs/{run_name}_{date}/`) to preserve previous analyses.

## Pipeline Stages

We use stages to seperate concerns, and allow human review at key moments. The semantic / sparse retrieval stage is essential to making the system more deterministic - a fundamental issue in LLM based systems - and is what characterises our implementation of retrival augmented generation (RAG). Previously we were (broadly) simply placing the large documents in context windows, but this made us more succeptible to hallucination (especially positional bias) and wasn't observable (barring out of scope interpretability methods or brute force that used tokens beyond our budget).

Stage 1 is called "Highlighting". We use hybrid retrieval approach (semantic + BM25) to highlight passages relevant to the input question, including core sentences and their sorrounding context (as part of sparse retrieval). Before proceeding to the next step, the highlighted PDFs are exported for review -- you can manually validate what will be used as part of the summarisation (stage 2)

Stage 2 is called "Summarisation". We generate concise summaries (250 words max.) from the "highlighted" content using the Claude API and plce them in the CSV template which also contains a reference for the files used as the input in Stage 1. The prompt was interated quickly with an ACOPS LLM export, and could likely be significantly improved.

Stage 3 is called "Judging". We rate the summaries on a scale 1-5 based on how clearly they answer the question with an LLM-as-judge approach, inspired by HuggingFace docs. We also store the rationales for the judgements in JSON for observability.


## Individual Files

```bash
# Process a single PDF
python highlighter.py \
  --pdf "path/to/document.pdf" \
  --question "What do they say about climate change?" \
  --output-pdf "output.pdf" \
  --output-json "output.json"
```

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional, for faster processing)
- Anthropic API key (for stages 2 and 3)
