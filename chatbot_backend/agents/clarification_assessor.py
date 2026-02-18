import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS
from chatbot_backend.prompts import CLARIFICATION_ASSESSOR_PROMPT
from chatbot_backend.schemas import ClarificationAssessment
from chatbot_backend.state import AgentState

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def _extract_queries(rag_history: list[dict]) -> list[str]:
    """Extract just the query strings from RAG history."""
    return [entry.get("query", "") for entry in rag_history if entry.get("query")]


def clarification_assessor_node(state: AgentState) -> dict:
    """Assess whether user's clarification is actionable for another RAG attempt."""
    llm = ChatOpenAI(model=AGENT_MODELS["clarification_assessor"])
    structured_llm = llm.with_structured_output(ClarificationAssessment, strict=True)

    classified_question = state.get("classified_question", "")
    user_clarification = state.get("user_clarification", "")
    rag_history = state.get("rag_history", [])

    previous_queries = _extract_queries(rag_history)
    queries_str = "\n".join(f"- {q}" for q in previous_queries) if previous_queries else "(none)"

    content = (
        f"Original classified question: {classified_question}\n\n"
        f"User's new clarification: {user_clarification}\n\n"
        f"Search queries already tried:\n{queries_str}"
    )

    messages = [
        SystemMessage(content=CLARIFICATION_ASSESSOR_PROMPT),
        HumanMessage(content=content),
    ]

    print(f"\n{'-' * 60}")
    print("[Clarification Assessor] INPUT:")
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
        print(f"[Clarification Assessor] OUTPUT: {response}")

    if response is None:
        # On LLM failure, treat clarification as not actionable
        return {
            "clarification_is_actionable": False,
            "clarification_used": True,
        }

    result = {
        "clarification_is_actionable": response.is_actionable,
        "clarification_used": True,
    }

    if response.is_actionable:
        result["information_gap"] = response.information_gap
        result["retrieval_attempts"] = 0

    return result
