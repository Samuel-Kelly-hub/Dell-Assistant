from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def add_rag_history(
    existing: list[dict] | None, new: list[dict] | None
) -> list[dict]:
    """Reducer that appends new RAG history entries to existing list."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


class AgentState(TypedDict):
    chat_id: str
    messages: Annotated[list[AnyMessage], add_messages]

    # Product selector
    product_name: str

    # Agent 1
    classified_question: str
    gatherer_attempts: int
    has_enough_info: bool

    # Agent 2 — RAG retrieval history
    rag_history: Annotated[list[dict], add_rag_history]  # {"query": str, "context": str, "urls": list[str]}

    # Agent 3 — Quality checking
    retrieval_attempts: int
    is_context_sufficient: bool
    information_gap: str  # Description of what information is missing
    final_answer: str
    escalated_to_human: bool

    # PDF fallback
    pdf_fallback_used: bool

    # Feedback
    user_satisfied: bool
    feedback_collected: bool

    # Post-answer clarification
    clarification_used: bool
    user_clarification: str
    clarification_is_actionable: bool

    # Feedback uncertainty
    feedback_uncertain: bool
