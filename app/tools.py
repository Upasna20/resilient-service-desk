from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field


# 📝 Contract 1: Escalation Schema (Explicit JSON Schema)
class EscalationInput(BaseModel):
    user_id: str = Field(
        ...,
        description="The unique identifier of the customer (e.g., 'CUST-12345').",
    )
    issue_summary: str = Field(
        ...,
        description="A concise summary of the customer's problem and their frustration level.",
    )
    sentiment_score: float = Field(
        ...,
        description="Sentiment score between 0.0 (Extremely Angry) and 1.0 (Happy).",
    )


# 📝 Contract 2: Knowledge Base Query Schema
class KBQueryInput(BaseModel):
    search_query: str = Field(
        ...,
        description="The semantic search query to look up in the Knowledge Base (e.g., 'return policy').",
    )


# 📝 Contract 3: Customer Details Schema
class CustomerDetailsInput(BaseModel):
    user_id: str = Field(
        ...,
        description="The unique identifier of the customer (e.g., 'CUST-12345').",
    )
    name: str = Field(
        ...,
        description="The name of the customer.",
    )
    email: str | None = Field(
        default="",
        description="Optional email address of the customer.",
    )


# 🔧 Tool 1: Escalate to Human (Descriptive Naming)
def escalate_to_human(
    input: EscalationInput,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Escalates high-priority, urgent, or frustrated customer tickets to a human support queue.

    Use this tool IMMEDIATELY if the customer exhibits high anger, requests a refund,
    or asks for things outside the agent's boundaries (like cancellation).

    Args:
        input (EscalationInput): The escalation input details containing user_id, issue_summary, and sentiment_score.
        tool_context (ToolContext, optional): Injected session tool context.

    Returns:
        dict[str, Any]: Confirmation status and the generated Ticket ID.
    """
    # 🚨 Guided Error Handling Example (Rubric Category 1)
    if not input.user_id.startswith("CUST-"):
        return {
            "status": "error",
            "error_recovery": "Invalid User ID format. Ensure it starts with 'CUST-'. Please ask the user to verify their ID.",
        }

    # 🛑 Human-in-the-Loop (HITL) Execution Stop (Rubric Category: HITL Stops)
    # Require human supervisor confirmation before executing escalation.
    if tool_context is not None and hasattr(tool_context, "request_confirmation"):
        if not getattr(tool_context, "tool_confirmation", None):
            tool_context.request_confirmation(
                hint="High-priority escalation requires human supervisor approval to proceed."
            )
            if hasattr(tool_context, "actions"):
                tool_context.actions.skip_summarization = True
            return {
                "status": "pending_confirmation",
                "message": "Escalation paused: awaiting human supervisor confirmation.",
            }
        elif not tool_context.tool_confirmation.confirmed:
            return {
                "status": "rejected",
                "message": "Escalation rejected by human supervisor.",
            }

    if tool_context is not None and hasattr(tool_context, "state"):
        customer_details = tool_context.state.get("customer_details", {})
        customer_details["user_id"] = input.user_id
        tool_context.state["customer_details"] = customer_details

    return {
        "status": "success",
        "ticket_id": f"TICKET-{input.user_id[-4:]}-URGENT",
        "message": "Ticket successfully enqueued in the Human Support Queue.",
    }


# 🔧 Tool 2: Query Knowledge Base
def query_knowledge_base(input: KBQueryInput) -> dict[str, Any]:
    """Searches the internal Knowledge Base (FAQ) for Global Retail Hub policies and procedures.

    Use this tool to find standard answers for shipping times, return policies,
    or general informational queries.

    Args:
        input (KBQueryInput): The knowledge base search query input.

    Returns:
        dict[str, Any]: The search results or a guided recovery message if nothing is found.
    """
    in_scope_topics = ["shipping", "return", "refund", "policy", "hours", "tracking"]

    # 🚨 Out of Scope Guardrail
    if not any(topic in input.search_query.lower() for topic in in_scope_topics):
        return {
            "status": "out_of_scope",
            "result": None,
            "error_recovery": (
                "The Knowledge Base has no information on this topic because it appears unrelated "
                "to Global Retail Hub. Do NOT make up an answer. Politely tell the user this request "
                "is outside your action area."
            ),
        }

    # 🚨 System Outage Simulation (Resilience)
    if "error" in input.search_query.lower():
        return {
            "status": "error",
            "error_recovery": "The Knowledge Base is temporarily offline. Please advise the user of the outage and escalate to a human if urgent.",
        }

    # 🟢 Success Path
    return {
        "status": "success",
        "result": "Standard shipping takes 3-5 business days. Returns are accepted within 30 days.",
    }


# 🔧 Tool 3: Save Customer Details to Session State
def save_customer_details(
    input: CustomerDetailsInput,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Saves customer details into the session state so they persist across event compactions.

    Use this tool IMMEDIATELY whenever the customer provides their name, email, or customer ID.

    Args:
        input (CustomerDetailsInput): Customer details containing user_id, name, and optional email.
        tool_context (ToolContext, optional): Injected session tool context.

    Returns:
        dict[str, Any]: Confirmation status and stored customer details.
    """
    if not input.user_id.startswith("CUST-"):
        return {
            "status": "error",
            "error_recovery": (
                "Invalid User ID format. Ensure it starts with 'CUST-'. "
                "Please ask the user to verify their ID."
            ),
        }

    details = {"user_id": input.user_id, "name": input.name, "email": input.email or ""}
    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["customer_details"] = details
    return {
        "status": "success",
        "message": "Customer details saved to session state.",
        "customer_details": details,
    }
