import time

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot_backend.config import AGENT_MODELS
from chatbot_backend.prompts import INFORMATION_GATHERER_PROMPT
from chatbot_backend.schemas import InformationGathererResponse
from chatbot_backend.state import AgentState

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff


def information_gatherer_node(state: AgentState) -> dict:
    """Agent 1 — assess info sufficiency and classify the question in one call."""
    llm = ChatOpenAI(model=AGENT_MODELS["information_gatherer"])
    structured_llm = llm.with_structured_output(
        InformationGathererResponse, strict=True
    )

    product_name = state["product_name"]

    messages = [
        SystemMessage(content=INFORMATION_GATHERER_PROMPT.format(product_name=product_name)),
        *state["messages"],
    ]

    print(f"\n{'-' * 60}")
    print("[Information Gatherer] INPUT:")
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
        print(f"[Information Gatherer] OUTPUT: {response}")

    if response is None:
        return {
            "has_enough_info": False,
            "gatherer_attempts": state.get("gatherer_attempts", 0) + 1,
            "messages": [
                AIMessage(
                    content="Could you please provide more details about your issue?"
                )
            ],
        }

    if not response.has_enough_info:
        return {
            "has_enough_info": False,
            "gatherer_attempts": state.get("gatherer_attempts", 0) + 1,
            "messages": [AIMessage(content=response.follow_up_question)],
        }

    return {
        "has_enough_info": True,
        "classified_question": response.classified_question,
        "messages": [
            AIMessage(
                content=f"Understood. Issue: {response.classified_question}"
            )
        ],
    }
