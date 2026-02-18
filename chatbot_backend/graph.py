from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from chatbot_backend.agents.answer_formulator import answer_formulator_node
from chatbot_backend.agents.clarification_assessor import clarification_assessor_node
from chatbot_backend.agents.feedback_collector import feedback_collector_node
from chatbot_backend.agents.information_gatherer import information_gatherer_node
from chatbot_backend.agents.pdf_fallback import pdf_fallback_node
from chatbot_backend.agents.quality_checker import quality_checker_node
from chatbot_backend.agents.rag_retriever import rag_retriever_node
from chatbot_backend.config import DATA_DIR, MAX_GATHERER_ATTEMPTS, MAX_RETRIEVAL_ATTEMPTS
from chatbot_backend.logging_.csv_logger import write_support_log
from chatbot_backend.logging_.ticket_writer import write_ticket
from chatbot_backend.state import AgentState
from chatbot_backend.tools.product_selector import get_product_candidates


# ── Nodes & routing (in graph flow order) ────────────────────────────────────

# Product selector — first node in the graph

def product_selector_node(state: AgentState) -> dict:
    """Prompt the user to select a product and enter their question."""
    raw = input("\nEnter the Dell product name (or 'general' for a general question): ").strip()

    if raw.lower() == "general":
        slug = "general"
    else:
        canonical, candidates, exact = get_product_candidates(raw, DATA_DIR, k=10)

        print("\nTop matches:")
        for i, c in enumerate(candidates, 1):
            marker = "  <-- exact match" if c == canonical else ""
            print(f"  {i}. {c}{marker}")

        choice = input("\nSelect a number (Enter = 1): ").strip()
        if choice == "" or choice == "1":
            idx = 0
        else:
            try:
                idx = int(choice) - 1
                if not 0 <= idx < len(candidates):
                    idx = 0
            except ValueError:
                idx = 0
        slug = candidates[idx]
        print(f"\nSelected product: {slug}")

    question = input("\nDescribe your technical issue below.\nYou: ").strip()

    return {
        "product_name": slug,
        "messages": [HumanMessage(content=question)],
    }


# Agent 1 — information_gatherer_node (imported from agents/)

def ask_user_for_details_node(state: AgentState) -> dict:
    """Print the follow-up question and read the user's response."""
    last_ai = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if last_ai:
        print(f"\nAgent: {last_ai.content}")

    user_input = input("You: ").strip()
    return {
        "messages": [HumanMessage(content=user_input)],
    }


def route_after_gatherer(state: AgentState) -> str:
    """Route after Agent 1: to RAG if enough info, else ask user or move on."""
    if state.get("has_enough_info", False):
        return "rag_retriever"

    if state.get("gatherer_attempts", 0) >= MAX_GATHERER_ATTEMPTS:
        # Exhausted clarification attempts — move on with what we have
        return "rag_retriever"

    return "ask_user_for_details"


# Agent 2 — rag_retriever_node (imported from agents/)

# Agent 3 — quality_checker_node (imported from agents/)

def route_after_quality_checker(state: AgentState) -> str:
    """Route after Agent 3: formulate answer, loop to RAG, or try PDF fallback."""
    if state.get("is_context_sufficient", False):
        return "answer_formulator"

    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "pdf_fallback"

    return "rag_retriever"


# Agent 4 — answer_formulator_node (imported from agents/)

def present_answer_node(state: AgentState) -> dict:
    """Display the final answer to the user."""
    answer = state.get("final_answer", "No answer available.")
    print(f"\n{'=' * 60}")
    print("Dell Technical Support — Answer")
    print(f"{'=' * 60}")
    print(answer)
    print(f"{'=' * 60}\n")
    return {
        "escalated_to_human": False,
    }


def route_after_present_answer(state: AgentState) -> str:
    """Route after presenting answer: to clarification or pre-feedback logging."""
    if state.get("clarification_used", False):
        return "log_pre_feedback"
    return "ask_for_clarification"


def ask_for_clarification_node(state: AgentState) -> dict:
    """Ask user if they have additional information, read their response."""
    print("Do you have any additional information that might help? (or press Enter to skip)")
    user_input = input("You: ").strip()

    if not user_input:
        return {
            "user_clarification": "",
            "clarification_is_actionable": False,
            "clarification_used": True,
        }

    return {
        "user_clarification": user_input,
    }


def route_after_clarification(state: AgentState) -> str:
    """Route after clarification: skip assessor if empty, otherwise assess."""
    if not state.get("user_clarification"):
        return "log_pre_feedback"
    return "clarification_assessor"


def route_after_clarification_assessor(state: AgentState) -> str:
    """Route after clarification assessment: to RAG or pre-feedback logging."""
    if state.get("clarification_is_actionable", False):
        return "rag_retriever"
    return "log_pre_feedback"


def route_after_log_pre_feedback(state: AgentState) -> str:
    """Route after pre-feedback logging: skip feedback if escalated."""
    if state.get("escalated_to_human", False):
        return "log_result"
    return "collect_feedback"


def route_after_pdf_fallback(state: AgentState) -> str:
    """Route after PDF fallback: present answer if found, otherwise escalate."""
    if state.get("pdf_fallback_used", False):
        return "present_fallback_answer"
    return "escalate"


def present_fallback_answer_node(state: AgentState) -> dict:
    """Display the PDF fallback answer to the user."""
    answer = state.get("final_answer", "No answer available.")
    print(f"\n{'=' * 60}")
    print("Dell Technical Support — Additional Information")
    print(f"{'=' * 60}")
    print(answer)
    print(f"{'=' * 60}\n")
    return {
        "escalated_to_human": False,
    }


def escalate_node(state: AgentState) -> dict:
    """Inform the user that the issue is being escalated."""
    notice = (
        "We were unable to find a sufficient answer to your query after "
        "multiple attempts. Your case has been escalated to a human support "
        "agent who will be in touch shortly."
    )
    print(f"\n{'=' * 60}")
    print("Dell Technical Support — Escalation Notice")
    print(f"{'=' * 60}")
    print(notice)
    print(f"{'=' * 60}\n")
    return {
        "escalated_to_human": True,
        "final_answer": notice,
    }


# Agent 5 — feedback_collector_node (imported from agents/)

def log_pre_feedback_node(state: AgentState) -> dict:
    """Write an initial log row before feedback collection (feedback_collected=False)."""
    chat_id = state.get("chat_id", "")
    product = state["product_name"]
    question = state.get("classified_question", "")
    escalated = state.get("escalated_to_human", False)
    gatherer_attempts = state.get("gatherer_attempts", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)

    if escalated:
        status = "escalated"
    else:
        status = "pending_feedback"

    write_support_log(
        chat_id=chat_id,
        product=product,
        question=question,
        status=status,
        escalated=escalated,
        gatherer_attempts=gatherer_attempts,
        retrieval_attempts=retrieval_attempts,
        user_satisfied=False,
        feedback_uncertain=False,
        feedback_collected=False,
    )

    return {}


def log_result_node(state: AgentState) -> dict:
    """Write to support log CSV, and optionally create a ticket."""
    chat_id = state.get("chat_id", "")
    product = state["product_name"]
    question = state.get("classified_question", "")
    escalated = state.get("escalated_to_human", False)
    user_satisfied = state.get("user_satisfied", False)
    feedback_uncertain = state.get("feedback_uncertain", False)
    gatherer_attempts = state.get("gatherer_attempts", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)
    final_answer = state.get("final_answer", "")

    if escalated:
        status = "escalated"
    elif user_satisfied:
        status = "success"
    else:
        status = "failure"

    write_support_log(
        chat_id=chat_id,
        product=product,
        question=question,
        status=status,
        escalated=escalated,
        gatherer_attempts=gatherer_attempts,
        retrieval_attempts=retrieval_attempts,
        user_satisfied=user_satisfied,
        feedback_uncertain=feedback_uncertain,
        feedback_collected=not escalated,
    )

    if not user_satisfied:
        description = (
            f"Customer was not satisfied with the support provided. "
            f"Product: {product}. Question: {question}. "
            f"Escalated: {escalated}. "
            f"Gatherer attempts: {gatherer_attempts}. "
            f"Retrieval attempts: {retrieval_attempts}."
        )
        write_ticket(
            chat_id=chat_id,
            product=product,
            question=question,
            answer_given=final_answer,
            escalated=escalated,
            description=description,
        )

    print("Session logged successfully.")
    return {}


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the Dell support StateGraph."""
    graph = StateGraph(AgentState)

    # Product Selector
    graph.add_node("product_selector", product_selector_node)
    # Agent 1 — Information Gatherer
    graph.add_node("information_gatherer", information_gatherer_node)
    graph.add_node("ask_user_for_details", ask_user_for_details_node)
    # Agent 2 — RAG Retriever
    graph.add_node("rag_retriever", rag_retriever_node)
    # Agent 3 — Quality Checker
    graph.add_node("quality_checker", quality_checker_node)
    # Agent 4 — Answer Formulator
    graph.add_node("answer_formulator", answer_formulator_node)
    graph.add_node("present_answer", present_answer_node)
    graph.add_node("escalate", escalate_node)
    # PDF fallback
    graph.add_node("pdf_fallback", pdf_fallback_node)
    graph.add_node("present_fallback_answer", present_fallback_answer_node)
    # Post-answer clarification
    graph.add_node("ask_for_clarification", ask_for_clarification_node)
    graph.add_node("clarification_assessor", clarification_assessor_node)
    # Pre-feedback logging
    graph.add_node("log_pre_feedback", log_pre_feedback_node)
    # Agent 5 — Feedback Collector
    graph.add_node("collect_feedback", feedback_collector_node)
    graph.add_node("log_result", log_result_node)

    # Edges from START
    graph.add_edge(START, "product_selector")
    graph.add_edge("product_selector", "information_gatherer")

    # Conditional edge after Agent 1
    graph.add_conditional_edges(
        "information_gatherer",
        route_after_gatherer,
        {
            "rag_retriever": "rag_retriever",
            "ask_user_for_details": "ask_user_for_details",
        },
    )

    # After asking user, loop back to Agent 1
    graph.add_edge("ask_user_for_details", "information_gatherer")

    # Agent 2 always goes to Agent 3
    graph.add_edge("rag_retriever", "quality_checker")

    # Conditional edge after Agent 3
    graph.add_conditional_edges(
        "quality_checker",
        route_after_quality_checker,
        {
            "answer_formulator": "answer_formulator",
            "pdf_fallback": "pdf_fallback",
            "rag_retriever": "rag_retriever",
        },
    )

    # Conditional edge after PDF fallback
    graph.add_conditional_edges(
        "pdf_fallback",
        route_after_pdf_fallback,
        {
            "present_fallback_answer": "present_fallback_answer",
            "escalate": "escalate",
        },
    )

    # PDF fallback answer goes to pre-feedback logging
    graph.add_edge("present_fallback_answer", "log_pre_feedback")

    # Agent 4 goes to present_answer
    graph.add_edge("answer_formulator", "present_answer")

    # Conditional edge after present_answer: to clarification or pre-feedback logging
    graph.add_conditional_edges(
        "present_answer",
        route_after_present_answer,
        {
            "ask_for_clarification": "ask_for_clarification",
            "log_pre_feedback": "log_pre_feedback",
        },
    )

    # Clarification flow
    graph.add_conditional_edges(
        "ask_for_clarification",
        route_after_clarification,
        {
            "clarification_assessor": "clarification_assessor",
            "log_pre_feedback": "log_pre_feedback",
        },
    )
    graph.add_conditional_edges(
        "clarification_assessor",
        route_after_clarification_assessor,
        {
            "rag_retriever": "rag_retriever",
            "log_pre_feedback": "log_pre_feedback",
        },
    )

    # Escalate goes directly to pre-feedback logging (no clarification opportunity)
    graph.add_edge("escalate", "log_pre_feedback")

    # Pre-feedback logging: skip feedback if escalated, otherwise collect it
    graph.add_conditional_edges(
        "log_pre_feedback",
        route_after_log_pre_feedback,
        {
            "collect_feedback": "collect_feedback",
            "log_result": "log_result",
        },
    )
    graph.add_edge("collect_feedback", "log_result")
    graph.add_edge("log_result", END)

    return graph.compile()
