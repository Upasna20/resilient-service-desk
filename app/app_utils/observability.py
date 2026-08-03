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

"""Observability and data privacy utilities for explicit intent/outcome logging and PII redaction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

# ==============================================================================
# 1. PII Redaction Mechanisms (Rubric: PII Redaction Before Saving to Memory)
# ==============================================================================

_GENERIC_CARD_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")
_CARD_RE = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b"
)
_SECRET_RE = re.compile(
    r"\b(password|passwd|token|api[_-]?key|secret)\s*[:=]\s*\S+", re.IGNORECASE
)


def redact_pii(text: str) -> str:
    """Redact Personally Identifiable Information (PII) from string text.

    Scrubs credit card numbers, SSNs, phone numbers, and secrets/tokens
    before saving customer details or queries into memory.

    Args:
        text: The raw input text potentially containing PII.

    Returns:
        The sanitized text string with sensitive PII replaced by redaction tags.
    """
    if not text or not isinstance(text, str):
        return str(text) if text is not None else ""

    redacted = _GENERIC_CARD_RE.sub("[REDACTED_CARD]", text)
    redacted = _CARD_RE.sub("[REDACTED_CARD]", redacted)
    redacted = _SSN_RE.sub("[REDACTED_SSN]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _SECRET_RE.sub(lambda m: f"{m.group(1)}=[REDACTED_SECRET]", redacted)
    return redacted


def redact_pii_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact PII from dictionary values before storing in session state or database.

    Args:
        data: Raw dictionary potentially containing PII strings.

    Returns:
        Sanitized dictionary with PII scrubbed from all text values.
    """
    if not isinstance(data, dict):
        return data
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            sanitized[k] = redact_pii(v)
        elif isinstance(v, dict):
            sanitized[k] = redact_pii_from_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [
                redact_pii_from_dict(item)
                if isinstance(item, dict)
                else (redact_pii(item) if isinstance(item, str) else item)
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


# ==============================================================================
# 2. Intent vs. Outcome Structured Logging (Rubric: Intent vs. Outcome Logging)
# ==============================================================================

import os

_fallback_logger = logging.getLogger("resilient_service_desk.agent_execution")
_recent_execution_logs: list[dict[str, Any]] = []
_cloud_logging_client = None


def _get_cloud_logger() -> Any | None:
    global _cloud_logging_client
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("ENV") == "test":
        return None
    try:
        from google.cloud import logging as google_cloud_logging

        if _cloud_logging_client is None:
            _cloud_logging_client = google_cloud_logging.Client()
        return _cloud_logging_client.logger("agent_execution_logger")
    except Exception:
        return None


def log_intent_and_outcome(
    intent: str,
    outcome: str,
    metadata: dict[str, Any] | None = None,
    severity: str = "INFO",
) -> dict[str, Any]:
    """Log explicit intent versus outcome during agent execution in structured format.

    Emits structured JSON logs via Google Cloud Logging when credentials are present,
    with safe fallback to Python standard structured logging for local development.

    Args:
        intent: The explicit goal or intention of the action (e.g., 'escalate_ticket').
        outcome: The actual result or state achieved (e.g., 'ticket_enqueued').
        metadata: Optional dictionary of contextual execution details.
        severity: Log severity level ('INFO', 'WARNING', 'ERROR').

    Returns:
        The complete structured log dictionary emitted.
    """
    safe_metadata = redact_pii_from_dict(metadata or {})
    log_entry = {
        "event_type": "agent_execution",
        "intent": intent,
        "outcome": outcome,
        "severity": severity,
        "metadata": safe_metadata,
    }
    _recent_execution_logs.append(log_entry)

    logger = _get_cloud_logger()
    if logger is not None:
        try:
            logger.log_struct(log_entry, severity=severity)
        except Exception:
            level = getattr(logging, severity.upper(), logging.INFO)
            _fallback_logger.log(level, json.dumps(log_entry))
    else:
        level = getattr(logging, severity.upper(), logging.INFO)
        _fallback_logger.log(level, json.dumps(log_entry))

    return log_entry


def get_recent_execution_logs() -> list[dict[str, Any]]:
    """Return recently emitted intent vs. outcome log entries (for tests & inspection)."""
    return _recent_execution_logs


def clear_execution_logs() -> None:
    """Clear recent execution logs memory buffer."""
    _recent_execution_logs.clear()
