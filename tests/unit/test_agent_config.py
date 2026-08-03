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
from typing import Any

import pytest
from google.adk.sessions.database_session_service import DatabaseSessionService

from app.agent import (
    app,
    consolidate_user_memory,
    dynamic_model_routing_callback,
    escalation_specialist,
    kb_specialist,
    record_user_issue_log,
    root_agent,
    schedule_memory_consolidation,
)
from app.app_utils.services import get_session_service
from app.tools import (
    CustomerDetailsInput,
    EscalationInput,
    escalate_to_human,
    query_knowledge_base,
    save_customer_details,
)


def test_root_agent_config() -> None:
    """Test that root_agent is configured according to requirements."""
    assert root_agent.name == "support_coordinator"
    assert root_agent.model.model == "gemini-3.5-flash"

    # Verify tool registration
    registered_tools = list(root_agent.tools)
    assert escalate_to_human in registered_tools
    assert query_knowledge_base in registered_tools
    assert save_customer_details in registered_tools
    assert len(registered_tools) == 3

    # Verify multi-agent hierarchy and sub-agent configuration
    sub_agent_map = {agent.name: agent for agent in root_agent.sub_agents}
    assert "kb_specialist" in sub_agent_map
    assert "escalation_specialist" in sub_agent_map
    assert sub_agent_map["kb_specialist"] is kb_specialist
    assert sub_agent_map["escalation_specialist"] is escalation_specialist
    assert sub_agent_map["kb_specialist"].model == "gemini-3.5-flash-lite"
    assert sub_agent_map["escalation_specialist"].model == "gemini-3.5-pro"
    assert query_knowledge_base in list(sub_agent_map["kb_specialist"].tools)
    assert escalate_to_human in list(sub_agent_map["escalation_specialist"].tools)

    # Verify dynamic model routing callback
    assert root_agent.before_model_callback is not None

    # Verify callbacks and compaction policy
    assert root_agent.before_agent_callback is not None
    assert root_agent.after_agent_callback is not None
    assert app.events_compaction_config is not None
    assert app.events_compaction_config.compaction_interval == 3
    assert app.events_compaction_config.overlap_size == 1

    # Verify system instructions content and guardrails
    instruction = str(root_agent.instruction)
    assert "Lead Support Orchestrator" in instruction
    assert "Global Retail Hub" in instruction
    assert "Customer details currently stored in session state" in instruction
    assert "escalate_to_human" in instruction
    assert "query_knowledge_base" in instruction
    assert "save_customer_details" in instruction
    assert "frustration" in instruction.lower() or "anger" in instruction.lower()
    assert "outside" in instruction.lower()
    assert "hallucinate" in instruction.lower() or "fabricate" in instruction.lower()


@pytest.mark.asyncio
async def test_dynamic_model_routing_callback() -> None:
    """Verify before_model_callback dynamically routes high complexity/frustration queries to gemini-3.5-pro."""

    class DummyCallbackContext:
        def __init__(self, text: str) -> None:
            self.user_content = text
            self.state: dict[str, Any] = {}

    class DummyRequest:
        def __init__(self, model: str) -> None:
            self.model = model

    # Normal query remains on default model
    ctx_normal = DummyCallbackContext("What are your store hours?")
    req_normal = DummyRequest("gemini-3.5-flash")
    await dynamic_model_routing_callback(ctx_normal, req_normal)
    assert req_normal.model == "gemini-3.5-flash"

    # High frustration/complexity query routes to pro model
    ctx_urgent = DummyCallbackContext(
        "I am furious, I need a manager and refund immediately!"
    )
    req_urgent = DummyRequest("gemini-3.5-flash")
    await dynamic_model_routing_callback(ctx_urgent, req_urgent)
    assert req_urgent.model == "gemini-3.5-pro"


def test_save_customer_details_state_persistence() -> None:
    """Verify that save_customer_details writes into tool_context.state."""

    class DummyContext:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    ctx = DummyContext()
    res = save_customer_details(
        CustomerDetailsInput(
            user_id="CUST-12345", name="Alice", email="alice@example.com"
        ),
        tool_context=ctx,
    )
    assert res["status"] == "success"
    assert ctx.state["customer_details"] == {
        "user_id": "CUST-12345",
        "name": "Alice",
        "email": "alice@example.com",
    }


def test_escalate_to_human_state_persistence() -> None:
    """Verify that escalate_to_human updates user_id in tool_context.state when provided."""

    class DummyContext:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {"customer_details": {}}

    ctx = DummyContext()
    res = escalate_to_human(
        EscalationInput(
            user_id="CUST-99999", issue_summary="Broken item", sentiment_score=0.1
        ),
        tool_context=ctx,
    )
    assert res["status"] == "success"
    assert ctx.state["customer_details"]["user_id"] == "CUST-99999"


@pytest.mark.asyncio
async def test_record_user_issue_log_callback() -> None:
    """Verify after_agent_callback captures turn summary into user-scoped persistent state."""

    class DummyCallbackContext:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {
                "customer_details": {"user_id": "CUST-4444", "name": "Eve"}
            }
            self.user_content = "Please help with refund"
            self.session = None

    ctx = DummyCallbackContext()
    await record_user_issue_log(ctx)
    assert "user:issue_logs" in ctx.state
    assert len(ctx.state["user:issue_logs"]) == 1
    log_item = ctx.state["user:issue_logs"][0]
    assert log_item["user_id"] == "CUST-4444"
    assert log_item["query"] == "Please help with refund"
    # Verify customer_details remains unprefixed (ephemeral session scope)
    assert "customer_details" in ctx.state

    # Allow scheduled background memory consolidation task to complete
    await asyncio.sleep(0.01)
    assert "user:consolidated_memory" in ctx.state
    assert ctx.state["user:consolidated_memory"]["issue_count"] == 1


def test_get_session_service_database() -> None:
    """Verify get_session_service returns DatabaseSessionService backed by SQLite."""
    service = get_session_service()
    assert isinstance(service, DatabaseSessionService)


@pytest.mark.asyncio
async def test_consolidate_user_memory() -> None:
    """Verify that consolidate_user_memory summarizes issue logs into user:consolidated_memory."""

    class DummyCallbackContext:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {
                "customer_details": {"user_id": "CUST-7777", "name": "Bob"},
                "user:issue_logs": [
                    {
                        "user_id": "CUST-7777",
                        "query": "First issue",
                        "customer_details": {},
                    },
                    {
                        "user_id": "CUST-7777",
                        "query": "Second issue",
                        "customer_details": {},
                    },
                ],
            }

    ctx = DummyCallbackContext()
    await consolidate_user_memory(ctx)
    assert "user:consolidated_memory" in ctx.state
    consolidated = ctx.state["user:consolidated_memory"]
    assert consolidated["user_id"] == "CUST-7777"
    assert consolidated["issue_count"] == 2
    assert "First issue" in consolidated["summary"]
    assert "Second issue" in consolidated["summary"]


@pytest.mark.asyncio
async def test_schedule_memory_consolidation() -> None:
    """Verify that schedule_memory_consolidation schedules an async background task."""

    class DummyCallbackContext:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {
                "customer_details": {"user_id": "CUST-8888", "name": "Sara"},
                "user:issue_logs": [
                    {
                        "user_id": "CUST-8888",
                        "query": "Refund request",
                        "customer_details": {},
                    },
                ],
            }

    ctx = DummyCallbackContext()
    task = schedule_memory_consolidation(ctx)
    assert task is not None
    await task
    assert "user:consolidated_memory" in ctx.state
    assert ctx.state["user:consolidated_memory"]["issue_count"] == 1
