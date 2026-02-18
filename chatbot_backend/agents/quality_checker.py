import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS
from chatbot_backend.prompts import CONTEXT_SUFFICIENCY_PROMPT
from chatbot_backend.schemas import ContextSufficiencyAssessment
from chatbot_backend.state import AgentState

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def _format_rag_history(rag_history: list[dict]) -> str:
    """Format RAG history for display to the LLM."""
    if not rag_history:
        return "(no previous searches)"

    parts = []
    for i, entry in enumerate(rag_history, 1):
        query = entry.get("query", "")
        context = entry.get("context", "")
        parts.append(
            f"--- Attempt {i} ---\n"
            f"Query: {query}\n"
            f"Result: {context if context else '(empty — no results found)'}"
        )
    return "\n\n".join(parts)


def quality_checker_node(state: AgentState) -> dict:
    """Agent 3 — check whether the retrieved context is sufficient."""
    llm = ChatOpenAI(model=AGENT_MODELS["quality_checker"])
    structured_llm = llm.with_structured_output(
        ContextSufficiencyAssessment, strict=True
    )

    product_name = state["product_name"]
    classified_question = state.get("classified_question", "")
    rag_history = state.get("rag_history", [])

    history_str = _format_rag_history(rag_history)

    messages = [
        SystemMessage(content=CONTEXT_SUFFICIENCY_PROMPT),
        HumanMessage(
            content=(
                f"Product: {product_name}\n"
                f"Question: {classified_question}\n\n"
                f"Search history:\n{history_str}"
            )
        ),
    ]

    print(f"\n{'-' * 60}")
    print("[Quality Checker] INPUT:")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")

    assessment = None
    for attempt in range(MAX_RETRIES):
        try:
            assessment = structured_llm.invoke(messages)
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

    if assessment is not None:
        print(f"[Quality Checker] OUTPUT: {assessment}")

    new_retrieval_attempts = state.get("retrieval_attempts", 0) + 1
    if assessment is None:
        # On LLM failure, assume context is insufficient and request more info
        return {
            "is_context_sufficient": False,
            "retrieval_attempts": new_retrieval_attempts,
            "information_gap": "Unable to assess context quality due to LLM error",
            "messages": [
                AIMessage(content="Context assessment failed. Attempting another search.")
            ],
        }

    if not assessment.is_sufficient:
        return {
            "is_context_sufficient": False,
            "retrieval_attempts": new_retrieval_attempts,
            "information_gap": assessment.information_gap,
            "messages": [
                AIMessage(
                    content=(
                        f"Context insufficient. Information gap: {assessment.information_gap}"
                    )
                )
            ],
        }

    return {
        "is_context_sufficient": True,
        "retrieval_attempts": new_retrieval_attempts,
        "information_gap": "",
    }
