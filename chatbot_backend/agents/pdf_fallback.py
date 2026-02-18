import re
import time
from collections import Counter
from pathlib import Path

import fitz
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS, PDF_DIR, PDF_TOC_PAGE_THRESHOLD
from chatbot_backend.prompts import PDF_TOC_ANALYSIS_PROMPT
from chatbot_backend.schemas import TOCAnalysis
from chatbot_backend.state import AgentState

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def _sanitise_url_for_filename(url: str) -> str:
    """Turn a URL into a Windows-safe filename base.

    Mirrors the logic in download_and_embed_data/scrape_relevant_urls.py.
    """
    base = re.sub(r"^https?://", "", url)
    base = re.sub(r'[\\/:*?"<>|]', "_", base)
    base = base[:240]
    return base


def _find_pdf_on_disk(url: str) -> Path | None:
    """Locate the local PDF file corresponding to a URL.

    The scraper saves files as '<sanitised_url>-1.pdf', '-2.pdf', etc.
    We return the first one that exists (attempt 1).
    """
    base = _sanitise_url_for_filename(url)
    candidate = PDF_DIR / f"{base}-1.pdf"
    if candidate.exists():
        return candidate
    return None


def _get_most_frequent_pdf_url(rag_history: list[dict]) -> str | None:
    """Count PDF URL occurrences across all rag_history entries; return the most frequent."""
    url_counts: Counter[str] = Counter()
    for entry in rag_history:
        for url in entry.get("urls", []):
            if url.lower().endswith(".pdf"):
                url_counts[url] += 1

    if not url_counts:
        return None

    return url_counts.most_common(1)[0][0]


def _extract_pages_text(pdf_path: Path, page_numbers: list[int]) -> str:
    """Extract text from specific pages of a PDF (1-indexed page numbers).

    Args:
        pdf_path: Path to the PDF file on disk.
        page_numbers: List of 1-indexed page numbers to extract.

    Returns:
        Concatenated text from the requested pages.
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    parts = []
    for page_num in page_numbers:
        idx = page_num - 1  # Convert to 0-indexed
        if 0 <= idx < total_pages:
            page_text = doc[idx].get_text()
            if page_text.strip():
                parts.append(f"--- Page {page_num} ---\n{page_text}")
    doc.close()
    return "\n\n".join(parts)


def _get_page_count(pdf_path: Path) -> int:
    """Return the total number of pages in a PDF."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def pdf_fallback_node(state: AgentState) -> dict:
    """PDF fallback agent — attempt to find useful information from the most relevant PDF."""
    rag_history = state.get("rag_history", [])
    classified_question = state.get("classified_question", "")
    product_name = state["product_name"]

    # Step 1: Find the most frequently referenced PDF URL
    pdf_url = _get_most_frequent_pdf_url(rag_history)
    if pdf_url is None:
        return {
            "pdf_fallback_used": False,
        }

    # Step 2: Map URL to local file
    pdf_path = _find_pdf_on_disk(pdf_url)
    if pdf_path is None:
        return {
            "pdf_fallback_used": False,
        }

    # Step 3: Check page count
    page_count = _get_page_count(pdf_path)

    if page_count <= PDF_TOC_PAGE_THRESHOLD:
        # Small PDF: extract ALL pages and present to user
        all_pages = list(range(1, page_count + 1))
        extracted_text = _extract_pages_text(pdf_path, all_pages)

        if not extracted_text.strip():
            return {
                "pdf_fallback_used": False,
            }

        fallback_answer = (
            f"We could not find an exact answer to your query after multiple "
            f"search attempts, but the following document may contain useful "
            f"information.\n\n"
            f"Source: {pdf_url}\n"
            f"Pages: all ({page_count} pages)\n\n"
            f"{extracted_text}"
        )
        return {
            "pdf_fallback_used": True,
            "final_answer": fallback_answer,
            "messages": [AIMessage(content=fallback_answer)],
        }

    # Step 4: Large PDF — extract first 10 pages and check for TOC
    first_pages_text = _extract_pages_text(
        pdf_path, list(range(1, PDF_TOC_PAGE_THRESHOLD + 1))
    )

    if not first_pages_text.strip():
        return {
            "pdf_fallback_used": False,
        }

    # Call LLM to analyse TOC
    llm = ChatOpenAI(model=AGENT_MODELS["pdf_fallback"])
    structured_llm = llm.with_structured_output(TOCAnalysis, strict=True)

    messages = [
        SystemMessage(content=PDF_TOC_ANALYSIS_PROMPT),
        HumanMessage(
            content=(
                f"Product: {product_name}\n"
                f"Question: {classified_question}\n\n"
                f"Extracted text from first {PDF_TOC_PAGE_THRESHOLD} pages:\n"
                f"{first_pages_text}"
            )
        ),
    ]

    print(f"\n{'-' * 60}")
    print("[PDF TOC Analyser] INPUT:")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")

    toc_analysis = None
    for attempt in range(MAX_RETRIES):
        try:
            toc_analysis = structured_llm.invoke(messages)
            break
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"LLM call failed (attempt {attempt + 1}), retrying: {e}")
                time.sleep(RETRY_DELAYS[attempt])
            else:
                print(f"LLM call failed after {MAX_RETRIES} attempts: {e}")
        except Exception as e:
            print(f"Unexpected LLM error: {e}")
            break

    if toc_analysis is not None:
        print(f"[PDF TOC Analyser] OUTPUT: {toc_analysis}")

    if toc_analysis is None:
        return {
            "pdf_fallback_used": False,
        }

    if not toc_analysis.has_toc:
        return {
            "pdf_fallback_used": False,
        }

    if not toc_analysis.relevant_pages:
        return {
            "pdf_fallback_used": False,
        }

    # Extract relevant pages
    extracted_text = _extract_pages_text(pdf_path, toc_analysis.relevant_pages)

    if not extracted_text.strip():
        return {
            "pdf_fallback_used": False,
        }

    section_note = ""
    if toc_analysis.most_relevant_section_title:
        section_note = (
            f"Relevant section: {toc_analysis.most_relevant_section_title}\n"
        )

    pages_str = ", ".join(str(p) for p in toc_analysis.relevant_pages)
    fallback_answer = (
        f"We could not find an exact answer to your query after multiple "
        f"search attempts, but the following section from a related document "
        f"may contain useful information.\n\n"
        f"Source: {pdf_url}\n"
        f"{section_note}"
        f"Pages: {pages_str}\n\n"
        f"{extracted_text}"
    )

    return {
        "pdf_fallback_used": True,
        "final_answer": fallback_answer,
        "messages": [AIMessage(content=fallback_answer)],
    }
