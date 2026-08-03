# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import datetime
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig, ResumabilityConfig
from google.adk.models import Gemini, LlmResponse
from google.genai import types

from app.app_utils.observability import (
    log_intent_and_outcome,
    redact_pii,
    redact_pii_from_dict,
)
from app.tools import (
    escalate_to_human,
    query_knowledge_base,
    save_customer_details,
)


MODEL = "gemini-3.5-flash"

SUPPORT_COORDINATOR_INSTRUCTION = """
You are the Lead Support Orchestrator for Global Retail Hub.
Your role is to orchestrate customer support ticket routing and FAQ resolution with sentiment awareness, accuracy, and professionalism.

Customer details currently stored in session state:
{customer_details}

### Operating Rules and Constraints:

1. **Customer State Management**:
   - Always keep customer details in session state.
   - When a customer provides their name, email, or customer ID (`user_id`), immediately call `save_customer_details` to record them in session state so they are preserved across turn compactions.

2. **Sentiment-Aware Escalation**:
   - Carefully analyze the customer's emotional tone and intent.
   - **Sentiment Scoring System:** Assign a score from 0.0 (Extremely Angry/Furious) to 1.0 (Happy/Delighted). 
   - **Trigger Threshold:** If the customer exhibits frustration, demands a refund, or their sentiment score falls **below 0.3** (High Frustration/Fury), they must be escalated.
   
3. **Pre-requisite Validation**:
   - Before invoking the `escalate_to_human` tool, you **MUST** have the customer's `user_id`.
   - If the customer has not mentioned their `user_id`, do NOT call the tool yet. Instead, politely ask the customer: "I want to get you to a specialist immediately, could you please provide your User ID?"

4. **Knowledge Retrieval**:
   - For standard, polite questions regarding store hours, shipping times, return windows, or general Global Retail Hub FAQ topics (Sentiment >= 0.4), invoke the `query_knowledge_base` tool.
   - Answer the customer's question clearly based on the retrieved information.

5. **Out-of-Scope Guardrails**:
   - If the customer asks unrelated questions (e.g., general knowledge, coding, weather), politely state: "That request is outside my retail support action area." Do not call any tools.

6. **Truthfulness**:
   - Never fabricate or hallucinate any policies, shipping rates, or order statuses.
   - Only state facts retrieved from tool outputs.
"""


async def initialize_customer_state(callback_context: CallbackContext) -> None:
    """Initialize customer_details and user:consolidated_memory in session state to preserve details across compaction."""
    if "customer_details" not in callback_context.state:
        callback_context.state["customer_details"] = {}
    if "user:consolidated_memory" not in callback_context.state:
        callback_context.state["user:consolidated_memory"] = {}
    log_intent_and_outcome(
        intent="initialize_session_state",
        outcome="customer_state_initialized",
    )


async def consolidate_user_memory(callback_context: CallbackContext) -> None:
    """Perform expensive memory consolidation asynchronously as a background task.

    Summarizes and compresses user issue logs into persistent user-scoped
    memory without blocking the immediate conversational response.
    """
    await asyncio.sleep(0)  # Yield control to the async event loop

    issue_logs = callback_context.state.get("user:issue_logs", [])
    if not isinstance(issue_logs, list) or not issue_logs:
        return

    customer_details = callback_context.state.get("customer_details", {})
    user_id = customer_details.get("user_id", "UNKNOWN")

    queries = [
        redact_pii(str(log.get("query", "")))
        for log in issue_logs
        if isinstance(log, dict) and log.get("query")
    ]

    consolidated_memory = {
        "user_id": user_id,
        "issue_count": len(issue_logs),
        "recent_queries": queries[-5:],
        "summary": redact_pii(
            f"Consolidated profile for {user_id}: {len(issue_logs)} issue(s) logged. "
            f"Recent queries: {'; '.join(queries[-3:])}"
        ),
        "last_consolidated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }

    callback_context.state["user:consolidated_memory"] = consolidated_memory
    log_intent_and_outcome(
        intent="consolidate_user_memory",
        outcome="memory_consolidated_with_pii_redaction",
        metadata={"user_id": user_id, "issue_count": len(issue_logs)},
    )


_background_tasks: set[asyncio.Task[Any]] = set()


def schedule_memory_consolidation(
    callback_context: CallbackContext,
) -> asyncio.Task[Any] | None:
    """Schedule expensive memory consolidation as a non-blocking background async task."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = loop.create_task(consolidate_user_memory(callback_context))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def record_user_issue_log(callback_context: CallbackContext) -> None:
    """Capture the turn summary and append it to a permanent user-scoped log list in the database."""
    issue_logs = callback_context.state.get("user:issue_logs", [])
    if not isinstance(issue_logs, list):
        issue_logs = []

    customer_details = callback_context.state.get("customer_details", {})
    user_id = customer_details.get("user_id", "UNKNOWN")

    query_text = ""
    if callback_context.user_content:
        query_text = str(callback_context.user_content)
    elif callback_context.session and hasattr(callback_context.session, "events"):
        for event in reversed(callback_context.session.events):
            if getattr(event, "author", "") == "user" and getattr(
                event, "content", None
            ):
                query_text = str(event.content)
                break

    redacted_query = redact_pii(query_text)
    redacted_details = redact_pii_from_dict(customer_details)
    log_entry = {
        "user_id": user_id,
        "query": redacted_query,
        "customer_details": redacted_details,
    }

    callback_context.state["user:issue_logs"] = issue_logs + [log_entry]
    log_intent_and_outcome(
        intent="record_user_issue",
        outcome="issue_logged_with_pii_redaction",
        metadata={"user_id": user_id},
    )
    schedule_memory_consolidation(callback_context)


KB_SPECIALIST_INSTRUCTION = """
You are the Knowledge Base Specialist for Global Retail Hub.
Your sole responsibility is to search the internal Knowledge Base (FAQ) for Global Retail Hub policies, store hours, shipping times, and return procedures using the `query_knowledge_base` tool.
Answer clearly and accurately based only on retrieved facts. Never make up or hallucinate policies.
If a request is outside Global Retail Hub FAQ scope, state that it is outside retail support.
"""

ESCALATION_SPECIALIST_INSTRUCTION = """
You are the Escalation & De-escalation Specialist for Global Retail Hub.
Your role is to handle angry, frustrated customers, refund disputes, or urgent issues that require human intervention.
1. Evaluate customer sentiment from 0.0 (Extremely Angry) to 1.0 (Happy).
2. Before calling `escalate_to_human`, verify you have the customer's `user_id`. If not provided, ask for their User ID starting with 'CUST-'.
3. Invoke `escalate_to_human` immediately for sentiment below 0.3 or refund/cancellation demands.
"""

kb_specialist = Agent(
    name="kb_specialist",
    model="gemini-3.5-flash-lite",
    description="Specialist agent that searches the internal Knowledge Base (FAQ) for Global Retail Hub store hours, shipping times, return policies, and standard procedures.",
    instruction=KB_SPECIALIST_INSTRUCTION,
    tools=[query_knowledge_base],
)

escalation_specialist = Agent(
    name="escalation_specialist",
    model="gemini-3.5-pro",
    description="Specialist agent that handles customer escalations, urgent refund requests, angry/frustrated users, or issues outside standard retail policies.",
    instruction=ESCALATION_SPECIALIST_INSTRUCTION,
    tools=[escalate_to_human],
)


async def input_guardrail_and_routing_callback(
    callback_context: CallbackContext, llm_request: Any
) -> Any | None:
    """Multi-layered safety guardrail (Layer 1: pre-model) & dynamic model router.

    1. Screens for prompt injections and prohibited/toxic topics before LLM invocation.
    2. Dynamically upgrades model tier for high complexity/frustration queries.
    """
    user_content = str(callback_context.user_content or "").lower()

    # 1. Input Safety Guardrail: check for prompt injections or prohibited requests
    prohibited_keywords = [
        "ignore previous instructions",
        "bypass guardrail",
        "system prompt",
        "jailbreak",
        "hack",
    ]
    if any(keyword in user_content for keyword in prohibited_keywords):
        log_intent_and_outcome(
            intent="process_user_prompt",
            outcome="blocked_by_input_guardrail",
            severity="WARNING",
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Request blocked by Global Retail Hub safety guardrails: prohibited input detected."
                    )
                ],
            )
        )

    # 2. Dynamic Model Routing: upgrade model tier for urgent/high-complexity requests
    high_complexity_keywords = [
        "manager",
        "furious",
        "lawyer",
        "unacceptable",
        "refund",
        "emergency",
        "angry",
        "urgent",
    ]
    if any(keyword in user_content for keyword in high_complexity_keywords):
        llm_request.model = "gemini-3.5-pro"
        log_intent_and_outcome(
            intent="model_routing",
            outcome="routed_to_pro_model",
            metadata={"model": "gemini-3.5-pro"},
        )
    else:
        log_intent_and_outcome(
            intent="model_routing",
            outcome="default_model_selected",
            metadata={"model": "gemini-3.5-flash"},
        )

    return None


# Backward compatibility alias for tests
dynamic_model_routing_callback = input_guardrail_and_routing_callback


async def tool_execution_guardrail_callback(
    tool: Any, args: dict[str, Any], tool_context: Any
) -> dict[str, Any] | None:
    """Multi-layered safety guardrail (Layer 2a: before tool execution).

    Enforces strict argument validation and scope boundaries before any tool runs.
    """
    if getattr(tool, "name", "") == "escalate_to_human":
        input_obj = args.get("input")
        sentiment = (
            getattr(input_obj, "sentiment_score", None)
            if input_obj
            else args.get("sentiment_score")
        )
        if sentiment is not None and (
            not isinstance(sentiment, (int, float))
            or not (0.0 <= float(sentiment) <= 1.0)
        ):
            return {
                "status": "error",
                "error_recovery": "Guardrail violation: sentiment_score must be between 0.0 and 1.0.",
            }
    return None


async def tool_output_sanitization_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Multi-layered safety guardrail (Layer 2b: after tool execution).

    Sanitizes tool output payloads to prevent internal system leakage or unhandled stack traces.
    """
    if isinstance(tool_response, dict):
        if "traceback" in tool_response or "exception" in tool_response:
            return {
                "status": "error",
                "error_recovery": "A system error occurred. Please try again or escalate to support.",
            }
    return None


async def output_safety_guardrail_callback(
    callback_context: CallbackContext, llm_response: Any
) -> Any | None:
    """Multi-layered safety guardrail (Layer 3: post-model output scan).

    Scans generated response text for ungrounded/hallucinated shipping windows or policy violations.
    """
    if (
        llm_response
        and hasattr(llm_response, "content")
        and llm_response.content
        and getattr(llm_response.content, "parts", None)
    ):
        for part in llm_response.content.parts:
            text = getattr(part, "text", "") or ""
            if (
                "lifetime warranty" in text.lower()
                or "unlimited returns" in text.lower()
            ):
                part.text = (
                    "Standard retail policy applies: Returns are accepted within 30 days. "
                    "Please refer to official Global Retail Hub terms."
                )
    return None


root_agent = Agent(
    name="support_coordinator",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=SUPPORT_COORDINATOR_INSTRUCTION,
    tools=[escalate_to_human, query_knowledge_base, save_customer_details],
    sub_agents=[kb_specialist, escalation_specialist],
    before_agent_callback=initialize_customer_state,
    before_model_callback=input_guardrail_and_routing_callback,
    after_model_callback=output_safety_guardrail_callback,
    before_tool_callback=tool_execution_guardrail_callback,
    after_tool_callback=tool_output_sanitization_callback,
    after_agent_callback=record_user_issue_log,
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
    ),
)
