import yaml
import os

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file and return structured config dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Build output directory with run metadata
    run_name = config['run']['name']
    run_date = config['run']['date']
    output_dir = f"{config['paths']['output_base_dir']}/{run_name}_{run_date}"
    
    return {
        'QUESTIONS': config['questions'],
        'KEYWORDS': config['keywords'],
        'EXCLUDED_SECTIONS': config['excluded_sections'],
        'SOURCE_DIR': config['paths']['source_dir'],
        'OUTPUT_DIR': output_dir,
        'CSV_TEMPLATE': config['paths']['csv_template'],
        'RUN_NAME': run_name,
        'RUN_DATE': run_date,
        'RUN_DESCRIPTION': config['run'].get('description', '')
    }

# Load config once at module level
CONFIG = load_config()
QUESTIONS = CONFIG['QUESTIONS']
KEYWORDS = CONFIG['KEYWORDS']
EXCLUDED_SECTIONS = CONFIG['EXCLUDED_SECTIONS']
SOURCE_DIR = CONFIG['SOURCE_DIR']
OUTPUT_DIR = CONFIG['OUTPUT_DIR']
CSV_TEMPLATE = CONFIG['CSV_TEMPLATE']
RUN_NAME = CONFIG['RUN_NAME']
RUN_DATE = CONFIG['RUN_DATE']
RUN_DESCRIPTION = CONFIG['RUN_DESCRIPTION']


