# Download and Embed Data

This module handles the complete data pipeline for scraping Dell support documentation and embedding it into a Qdrant vector database for semantic search.

## Pipeline Overview

The pipeline consists of 5 sequential steps:

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `scrape_sitemaps.py` | Dell sitemap URLs | `dell_sitemaps_success_*.jsonl` |
| 2 | `clean_sitemaps.py` | JSONL files | `urls_to_scrape.csv` |
| 3 | `scrape_relevant_urls.py` | CSV | `scraped_pages-final.csv` |
| 4 | `clean_scraped_text.py` | CSV | `text_to_embed.csv` |
| 5 | `embed_text.py` | CSV | Qdrant database |

## Running the Pipeline

### Full Pipeline

Run all steps in sequence:

```bash
python run_downloading_embedding_data_pipeline.py
```

This passes data in memory between steps whilst still writing intermediate files for persistence and debugging.

### Individual Steps

Each script can also be run standalone:

```bash
python scrape_sitemaps.py      # Step 1
python clean_sitemaps.py       # Step 2
python scrape_relevant_urls.py # Step 3
python clean_scraped_text.py   # Step 4
python embed_text.py           # Step 5
```

## Script Details

### `scrape_sitemaps.py`

Recursively scrapes Dell's XML sitemaps to discover all available URLs.

- Starts from the root sitemap (`index-support-href-sitemap.xml.gz`)
- Handles both sitemap indexes and URL sets
- Uses parallel workers for efficient downloading
- Outputs JSONL files with sitemap URL to page URLs mapping

### `clean_sitemaps.py`

Filters the scraped sitemap URLs to extract relevant documentation pages.

- Reads JSONL files from step 1
- Filters for `en-us` and `en-uk` URLs only
- Removes duplicates
- Outputs a CSV of URLs to scrape

### `scrape_relevant_urls.py`

Downloads and extracts text content from each URL.

- Uses Playwright for HTML pages (handles JavaScript-rendered content)
- Uses PyMuPDF for PDF documents
- Falls back to OCR (Tesseract) for scanned PDFs
- Includes retry logic with exponential backoff
- Runs in parallel with multiple workers
- Saves intermediate results to handle crashes gracefully

### `clean_scraped_text.py`

Cleans and normalises the scraped text for embedding.

- Separates HTML and PDF content for different cleaning strategies
- Removes common boilerplate text (headers, footers, navigation)
- Normalises whitespace and encoding
- Extracts product identifiers from URLs
- Combines cleaned HTML and PDF data

### `embed_text.py`

Generates embeddings and stores them in Qdrant.

- Uses the `BAAI/bge-large-en-v1.5` sentence transformer model
- Splits long documents into overlapping chunks (512 tokens max, 50 token overlap)
- Skips URLs already present in Qdrant (incremental updates)
- Creates a payload index on the `product` field for filtered search
- Batches embeddings for efficient processing

## Dependencies

- `requests` - HTTP client for sitemap fetching
- `pandas` - Data manipulation
- `playwright` - Browser automation for HTML scraping
- `PyMuPDF` (fitz) - PDF text extraction
- `pytesseract` - OCR for scanned PDFs
- `sentence-transformers` - Embedding model
- `qdrant-client` - Vector database client

## Configuration

Key configuration values are defined at the top of each script:

- **Qdrant**: `http://localhost:6333` (default)
- **Collection name**: `chunks`
- **Embedding model**: `BAAI/bge-large-en-v1.5`
- **Batch size**: 64
- **Chunk overlap**: 50 tokens

## Data Directory

All intermediate and output files are stored in `../data/` relative to this directory.
