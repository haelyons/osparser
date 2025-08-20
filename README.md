## OSPAR Climate Parser v3

### Basic Setup

Note this application runs MUCH faster when you have CUDA-enabled GPU, but it WILL work on a machine without one -- just quite slowly.

```bash
# Create virtual environment
python -m venv rag-env

# Activate (Linux/Mac)
source rag-env/bin/activate
# rag-env\Scripts\activate or Activate.ps1 on Windows

# Install packages
pip install PyMuPDF "sentence-transformers>=2.2.0" numpy torch nltk rank-bm25 anthropic
```

### Anthropic API key (not needed for highlighting, just summaries)

Create `.keys` file in project root:
```
ANTHROPIC_API_KEY=your_key_here
```

### Usage

#### Individual Highlighting
Uses our retrieval pipeline to input a single PDF and output a JSON of the top-ranked sentences and a PDF with those sentences highlighted, and the excluded parts also indicated.
```bash
python highlighter.py --pdf "path/to/document.pdf" --question "What do they say about climate change?" --output-pdf "output.pdf" --output-json "output.json"
```

#### Summarisation
Generates summaries based on the JSON outputs of each PDF and enters them into the results template
```bash
python summarise.py results/results_template.csv outputs
```

#### Batch Highlighting
Automatically highlights all PDFs in `sources/` directory against all predefined climate questions.
```bash
python batch_process.py
```

### Directory Structure
- `sources/` - Input PDFs
- `outputs/` - Generated highlights and JSON
- `results/` - CSV templates and results
