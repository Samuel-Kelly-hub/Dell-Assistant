# Chatbot Backend

Multi-agent technical support chatbot built with LangGraph and LangChain. Uses a retrieval-augmented generation (RAG) pipeline backed by a Qdrant vector database to answer Dell product support questions.

## Architecture Overview

The system is a LangGraph `StateGraph` with 5 LLM-powered agents connected by conditional routing. The graph manages conversation state, retries, and fallback paths automatically.

```
User Question
     |
     v
[Product Selector] -----> [Agent 1: Information Gatherer] <---> [Ask User for Details]
                                      |
                                      v
                           [Agent 2: RAG Retriever] <------+
                                      |                    |
                                      v                    |
                           [Agent 3: Quality Checker] -----+  (retry loop, max 3)
                                      |
                              sufficient? ----no----> [PDF Fallback] ----> [Escalate]
                                      |
                                     yes
                                      |
                                      v
                           [Agent 4: Answer Formulator]
                                      |
                                      v
                           [Present Answer] ----> [Ask for Clarification] ----> [Clarification Assessor]
                                      |
                                      v
                           [Agent 5: Feedback Collector]
                                      |
                                      v
                              [Log Result / Create Ticket]
```

## Running the Chatbot

```bash
python chatbot_backend/stream.py
```

Requires Docker to be running (Qdrant is started automatically if not already running).

## Agents

| Agent | File | Model | Purpose |
|-------|------|-------|---------|
| Information Gatherer | `agents/information_gatherer.py` | gpt-5-mini | Collects enough detail from the user to form an effective search query. Asks up to 3 follow-up questions. |
| RAG Retriever | `agents/rag_retriever.py` | gpt-5-nano | Formulates a search query and retrieves matching chunks from Qdrant. |
| Quality Checker | `agents/quality_checker.py` | gpt-5-mini | Assesses whether retrieved context is relevant enough to answer the question. |
| Answer Formulator | `agents/answer_formulator.py` | gpt-5-mini | Produces the final user-facing answer from the accumulated context. |
| Feedback Collector | `agents/feedback_collector.py` | gpt-5-nano | Classifies user feedback as satisfied, unsatisfied, or uncertain. |
| Clarification Assessor | `agents/clarification_assessor.py` | gpt-5-mini | Determines whether user clarification after an answer is actionable for a new search. |
| PDF Fallback | `agents/pdf_fallback.py` | gpt-5-mini | Analyses the table of contents of a relevant PDF to extract targeted pages when RAG search fails. |

All agents use structured output (`with_structured_output`) and include retry logic with exponential backoff for transient API errors.

## Graph Flow

1. **Product Selector** — user picks a Dell product from a fuzzy-matched list.
2. **Information Gatherer** — gathers symptoms/details from the user (up to 3 rounds).
3. **RAG Retriever + Quality Checker loop** — searches Qdrant, checks if the context is relevant, retries with a refined query if not (up to 3 attempts).
4. **Answer Formulator** — produces an answer if context is sufficient.
5. **PDF Fallback** — if all RAG attempts fail, tries to extract relevant pages from the most-referenced PDF using TOC analysis.
6. **Escalation** — if PDF fallback also finds nothing, escalates to a human agent.
7. **Clarification** — after an answer, the user can provide additional information to trigger a new search.
8. **Feedback** — collects user satisfaction (skipped automatically if escalated).
9. **Logging** — writes to `support_log.csv`; creates a ticket in `tickets.csv` if the user was unsatisfied.

## File Structure

```
chatbot_backend/
  stream.py               # Entry point — builds and invokes the graph
  graph.py                # LangGraph StateGraph definition, nodes, and routing
  state.py                # AgentState TypedDict with reducers
  config.py               # Models, retry limits, Qdrant settings, file paths
  prompts.py              # All LLM system prompts
  schemas.py              # Pydantic models for structured LLM output
  infrastructure.py       # Docker/Qdrant container management
  agents/
    information_gatherer.py
    rag_retriever.py
    quality_checker.py
    answer_formulator.py
    feedback_collector.py
    clarification_assessor.py
    pdf_fallback.py
  tools/
    product_selector.py   # Fuzzy product matching against product_list.csv
    rag_search.py         # Qdrant vector search with sentence-transformers
  logging_/
    csv_logger.py         # Appends rows to support_log.csv
    ticket_writer.py      # Appends rows to tickets.csv
```

## Configuration

Key settings in `config.py`:

| Setting | Value |
|---------|-------|
| Qdrant URL | `http://localhost:6333` |
| Qdrant collection | `chunks` |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| RAG top-K results | 3 |
| Max gatherer attempts | 3 |
| Max retrieval attempts | 3 |
| PDF TOC page threshold | 10 (PDFs with fewer pages are returned in full) |

## Logging

Two CSV files are written to `../logs/`:

- **`support_log.csv`** — one row per session recording product, question, status (`success`/`failure`/`escalated`), attempt counts, and feedback.
- **`tickets.csv`** — one row per unsatisfied or escalated session, including the answer given and a description for follow-up.

## Dependencies

- `langgraph` — graph orchestration
- `langchain-core`, `langchain-openai` — LLM integration and structured output
- `qdrant-client` — vector database client
- `sentence-transformers` — embedding model for RAG search
- `PyMuPDF` (fitz) — PDF text extraction for the fallback agent
- `docker` — Qdrant container management
- `requests` — Qdrant health checks
