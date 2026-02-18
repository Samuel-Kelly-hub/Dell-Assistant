import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS
from chatbot_backend.prompts import FORMULATE_ANSWER_PROMPT
from chatbot_backend.schemas import FormulatedAnswer
from chatbot_backend.state import AgentState

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def _format_all_context(rag_history: list[dict]) -> str:
    """Concatenate all contexts from RAG history for answer formulation."""
    if not rag_history:
        return "(no context available)"

    parts = []
    for i, entry in enumerate(rag_history, 1):
        query = entry.get("query", "")
        context = entry.get("context", "")
        if context:
            parts.append(
                f"--- Search {i}: \"{query}\" ---\n{context}"
            )
    return "\n\n".join(parts) if parts else "(no relevant context found)"


def answer_formulator_node(state: AgentState) -> dict:
    """Agent 4 — formulate the final answer from sufficient context."""
    llm = ChatOpenAI(model=AGENT_MODELS["answer_formulator"])
    structured_llm = llm.with_structured_output(FormulatedAnswer, strict=True)

    product_name = state["product_name"]
    classified_question = state.get("classified_question", "")
    rag_history = state.get("rag_history", [])

    all_context = _format_all_context(rag_history)

    messages = [
        SystemMessage(content=FORMULATE_ANSWER_PROMPT),
        HumanMessage(
            content=(
                f"Product: {product_name}\n"
                f"Question: {classified_question}\n\n"
                f"Retrieved context:\n{all_context}"
            )
        ),
    ]

    print(f"\n{'-' * 60}")
    print("[Answer Formulator] INPUT:")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")

    formulated = None
    for attempt in range(MAX_RETRIES):
        try:
            formulated = structured_llm.invoke(messages)
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

    if formulated is not None:
        print(f"[Answer Formulator] OUTPUT: {formulated}")

    if formulated is None:
        fallback_answer = (
            "I apologise, but I am currently unable to formulate a response. "
            "Please try again later or contact Dell support directly."
        )
        return {
            "final_answer": fallback_answer,
            "messages": [AIMessage(content=fallback_answer)],
        }

    return {
        "final_answer": formulated.answer,
        "messages": [AIMessage(content=formulated.answer)],
    }
