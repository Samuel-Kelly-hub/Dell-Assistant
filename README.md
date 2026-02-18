# Dell Technical Support Assistant

An automated technical support system for Dell products. Scrapes Dell's support documentation, embeds it into a vector database, and uses a multi-agent chatbot to answer user questions via retrieval-augmented generation (RAG).

I use several ways of selecting relevant parts of PDF documents for an LLM to analyse:
RAG
For long documents, reading the contents, identifying the relevant section(s) and then the selected section only.
For short documents, reading the entire thing

The chatbot utilises human-in-the-loop, logs its runs, and generates tickets for unsolved problems.
## Project Structure

```
Dell_Assistant/
  chatbot_backend/        # Multi-agent chatbot (LangGraph + LangChain)
  download_and_embed_data/# Data pipeline: scraping, cleaning, embedding
  data/                   # Intermediate and output data files
  logs/                   # Session logs and support tickets
  qdrant_storage/         # Qdrant vector database persistent storage
  requirements.txt
```

Each module has its own README with detailed documentation:
- [`chatbot_backend/README.md`](chatbot_backend/README.md)
- [`download_and_embed_data/README.md`](download_and_embed_data/README.md)

## How It Works

### 1. Data Pipeline (`download_and_embed_data/`)

Scrapes Dell's sitemaps to discover support pages, downloads and extracts text from HTML pages and PDFs, cleans the content, and embeds it into a Qdrant vector database using the `BAAI/bge-large-en-v1.5` sentence transformer model.

### 2. Chatbot (`chatbot_backend/`)

A LangGraph state machine with 5 LLM-powered agents:

1. **Information Gatherer** — collects details about the user's issue
2. **RAG Retriever** — searches the vector database for relevant documentation
3. **Quality Checker** — assesses whether the retrieved context is relevant
4. **Answer Formulator** — produces a user-facing answer from the context
5. **Feedback Collector** — records whether the user was satisfied

Additional features include a PDF fallback agent (extracts pages from PDFs via TOC analysis when RAG search fails), post-answer clarification flow, automatic escalation to human support, and session logging with ticket creation.

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- An OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Tesseract OCR (only needed for the data pipeline, for scanned PDFs)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Playwright browsers (only needed for the data pipeline):

```bash
playwright install chromium
```

3. Ensure Docker is running (Qdrant is started automatically when the chatbot launches).

## Usage

### Running the data pipeline

```bash
cd download_and_embed_data
python run_downloading_embedding_data_pipeline.py
```

### Running the chatbot

```bash
python chatbot_backend/stream.py
```

The chatbot will:
1. Start Qdrant if it is not already running
2. Prompt you to select a Dell product
3. Ask you to describe your technical issue
4. Search the knowledge base and formulate an answer
5. Collect feedback and log the session

## Data

| File | Description |
|------|-------------|
| `data/product_list.csv` | Allowlisted Dell product slugs for product selection |
| `data/urls_to_scrape.csv` | Filtered URLs from Dell sitemaps |
| `data/scraped_pages-final.csv` | Extracted text from all scraped pages |
| `data/text_to_embed.csv` | Cleaned text ready for embedding |
| `logs/support_log.csv` | Session log with status, attempts, and feedback |
| `logs/tickets.csv` | Tickets created for unsatisfied or escalated sessions |
