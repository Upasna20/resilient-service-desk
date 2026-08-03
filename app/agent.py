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
from google.adk.apps.app import EventsCompactionConfig
from google.adk.models import Gemini
from google.genai import types

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
        str(log.get("query", ""))
        for log in issue_logs
        if isinstance(log, dict) and log.get("query")
    ]

    consolidated_memory = {
        "user_id": user_id,
        "issue_count": len(issue_logs),
        "recent_queries": queries[-5:],
        "summary": (
            f"Consolidated profile for {user_id}: {len(issue_logs)} issue(s) logged. "
            f"Recent queries: {'; '.join(queries[-3:])}"
        ),
        "last_consolidated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }

    callback_context.state["user:consolidated_memory"] = consolidated_memory


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

    log_entry = {
        "user_id": user_id,
        "query": query_text,
        "customer_details": customer_details,
    }

    callback_context.state["user:issue_logs"] = issue_logs + [log_entry]
    schedule_memory_consolidation(callback_context)


root_agent = Agent(
    name="support_coordinator",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=SUPPORT_COORDINATOR_INSTRUCTION,
    tools=[escalate_to_human, query_knowledge_base, save_customer_details],
    before_agent_callback=initialize_customer_state,
    after_agent_callback=record_user_issue_log,
)

app = App(
    root_agent=root_agent,
    name="app",
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
    ),
)
