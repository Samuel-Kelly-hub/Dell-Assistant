import csv
from datetime import datetime, timezone

from chatbot_backend.config import LOGS_DIR, TICKETS_CSV

TICKET_HEADERS = [
    "timestamp",
    "chat_id",
    "product",
    "question",
    "answer_given",
    "escalated",
    "description",
]


def write_ticket(
    chat_id: str,
    product: str,
    question: str,
    answer_given: str,
    escalated: bool,
    description: str,
) -> None:
    """Append a row to the tickets CSV."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = TICKETS_CSV.exists()

    with open(TICKETS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(TICKET_HEADERS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            chat_id,
            product,
            question,
            answer_given,
            escalated,
            description,
        ])
