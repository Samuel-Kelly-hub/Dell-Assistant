import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS
from chatbot_backend.prompts import RAG_RETRIEVER_PROMPT
from chatbot_backend.schemas import RAGRetrieverQuery
from chatbot_backend.state import AgentState
from chatbot_backend.tools.rag_search import rag_search

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def _format_previous_attempts(rag_history: list[dict]) -> str:
    """Format previous search attempts for the LLM."""
    if not rag_history:
        return "(none)"

    parts = []
    for i, entry in enumerate(rag_history, 1):
        query = entry.get("query", "")
        context = entry.get("context", "")
        result_summary = context[:200] + "..." if len(context) > 200 else context
        parts.append(f"Attempt {i}: Query: \"{query}\" → {result_summary if result_summary else '(no results)'}")
    return "\n".join(parts)


def rag_retriever_node(state: AgentState) -> dict:
    """Agent 2 — formulate a search query and retrieve context from RAG."""
    llm = ChatOpenAI(model=AGENT_MODELS["rag_retriever"])
    structured_llm = llm.with_structured_output(RAGRetrieverQuery, strict=True)

    product_name = state["product_name"]
    classified_question = state.get("classified_question", "")
    user_clarification = state.get("user_clarification", "")
    information_gap = state.get("information_gap", "")
    rag_history = state.get("rag_history", [])

    previous_attempts_str = _format_previous_attempts(rag_history)

    # Build context for the LLM
    content_parts = [
        f"Product: {product_name}",
        f"Question: {classified_question}",
    ]
    if user_clarification:
        content_parts.append(f"User's additional information: {user_clarification}")
    if information_gap:
        content_parts.append(f"Information gap to address: {information_gap}")
    content_parts.append(f"Previous search attempts:\n{previous_attempts_str}")

    messages = [
        SystemMessage(content=RAG_RETRIEVER_PROMPT),
        HumanMessage(content="\n".join(content_parts)),
    ]

    print(f"\n{'-' * 60}")
    print("[RAG Retriever] INPUT:")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")

    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = structured_llm.invoke(messages)
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

    if response is not None:
        print(f"[RAG Retriever] OUTPUT: {response}")

    if response is None:
        # Fallback: use classified_question as the search query
        fallback_query = f"{product_name} {classified_question}"
        context, urls = rag_search(product_name, fallback_query)
        return {
            "rag_history": [{"query": fallback_query, "context": context, "urls": urls}],
            "messages": [
                AIMessage(content=f"RAG search completed (fallback). Query: {fallback_query}")
            ],
        }

    # Use the LLM-generated search query to retrieve context
    context, urls = rag_search(product_name, response.search_query)

    # Append to history (reducer will handle merging)
    new_history_entry = [{"query": response.search_query, "context": context, "urls": urls}]

    return {
        "rag_history": new_history_entry,
        "messages": [
            AIMessage(
                content=f"RAG search completed. Query: {response.search_query}"
            )
        ],
    }
