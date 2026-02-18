import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS
from chatbot_backend.prompts import FEEDBACK_CLASSIFICATION_PROMPT
from chatbot_backend.schemas import UserFeedback
from chatbot_backend.state import AgentState

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def feedback_collector_node(state: AgentState) -> dict:
    """Agent 5 — collect and classify user feedback."""
    print("Was this information sufficient? (yes/no)")
    user_input = input("You: ").strip()

    llm = ChatOpenAI(model=AGENT_MODELS["feedback_collector"])
    structured_llm = llm.with_structured_output(UserFeedback, strict=True)

    messages = [
        SystemMessage(content=FEEDBACK_CLASSIFICATION_PROMPT),
        HumanMessage(content=user_input),
    ]

    print(f"\n{'-' * 60}")
    print("[Feedback Collector] INPUT:")
    for msg in messages:
        print(f"  [{type(msg).__name__}] {msg.content}")

    feedback = None
    for attempt in range(MAX_RETRIES):
        try:
            feedback = structured_llm.invoke(messages)
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

    if feedback is not None:
        print(f"[Feedback Collector] OUTPUT: {feedback}")

    if feedback is None:
        # On LLM failure, assume satisfied but uncertain
        return {
            "user_satisfied": True,
            "feedback_collected": True,
            "feedback_uncertain": True,
            "messages": [HumanMessage(content=user_input)],
        }

    # If uncertain, we treat as satisfied (no ticket) but flag for review
    if feedback.is_uncertain:
        return {
            "user_satisfied": True,
            "feedback_collected": True,
            "feedback_uncertain": True,
            "messages": [HumanMessage(content=user_input)],
        }

    return {
        "user_satisfied": feedback.is_satisfied,
        "feedback_collected": True,
        "feedback_uncertain": False,
        "messages": [HumanMessage(content=user_input)],
    }
