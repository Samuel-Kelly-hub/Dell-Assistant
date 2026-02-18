import csv
from datetime import datetime, timezone

from chatbot_backend.config import LOGS_DIR, SUPPORT_LOG_CSV

SUPPORT_LOG_HEADERS = [
    "timestamp",
    "chat_id",
    "product",
    "question",
    "status",
    "escalated",
    "gatherer_attempts",
    "retrieval_attempts",
    "user_satisfied",
    "feedback_uncertain",
    "feedback_collected",
]


def write_support_log(
    chat_id: str,
    product: str,
    question: str,
    status: str,
    escalated: bool,
    gatherer_attempts: int,
    retrieval_attempts: int,
    user_satisfied: bool,
    feedback_uncertain: bool = False,
    feedback_collected: bool = False,
) -> None:
    """Append a row to the support log CSV."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = SUPPORT_LOG_CSV.exists()

    with open(SUPPORT_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SUPPORT_LOG_HEADERS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            chat_id,
            product,
            question,
            status,
            escalated,
            gatherer_attempts,
            retrieval_attempts,
            user_satisfied,
            feedback_uncertain,
            feedback_collected,
        ])
