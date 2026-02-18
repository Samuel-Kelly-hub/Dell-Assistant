import uuid

from chatbot_backend.graph import build_graph
from chatbot_backend.infrastructure import ensure_qdrant_running


def main() -> None:
    ensure_qdrant_running()

    print("=" * 60)
    print("  Dell Technical Support — Multi-Agent System")
    print("=" * 60)

    chat_id = str(uuid.uuid4())

    initial_state = {
        "chat_id": chat_id,
        "messages": [],
        "product_name": "",
        "classified_question": "",
        "gatherer_attempts": 0,
        "has_enough_info": False,
        "rag_history": [],
        "retrieval_attempts": 0,
        "is_context_sufficient": False,
        "information_gap": "",
        "final_answer": "",
        "escalated_to_human": False,
        "pdf_fallback_used": False,
        "user_satisfied": False,
        "feedback_collected": False,
        "clarification_used": False,
        "user_clarification": "",
        "clarification_is_actionable": False,
        "feedback_uncertain": False,
    }

    app = build_graph()
    app.invoke(initial_state)

    print("\nThank you for contacting Dell Technical Support. Goodbye!")


if __name__ == "__main__":
    main()
