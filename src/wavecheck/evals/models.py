"""Thin, logged wrapper around the Anthropic Messages API.

The call pattern follows the official docs:
https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
The SDK reads ANTHROPIC_API_KEY from the environment (loaded from .env here) and retries
transient errors on its own.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from wavecheck.io import append_jsonl

load_dotenv()

try:
    # Use the operating system's certificate store (Windows/macOS), so HTTPS works behind a
    # corporate proxy that re-signs TLS traffic with its own (system-trusted) certificate.
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # optional; plain certifi is used otherwise
    pass


def get_client() -> anthropic.Anthropic:
    """Create an API client (reads ANTHROPIC_API_KEY)."""
    return anthropic.Anthropic()


def complete(
    client: Any,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    log_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one user message and return a JSON-serialisable record (also logged if log_path).

    record["text"] joins the text blocks only; record["raw"] keeps the full response, including
    any thinking blocks, so nothing is lost.
    """
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        request["system"] = system
    started = time.perf_counter()
    response = client.messages.create(**request)
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request": request,
        "model": response.model,
        "response_id": response.id,
        "stop_reason": response.stop_reason,
        "text": "".join(block.text for block in response.content if block.type == "text"),
        "usage": response.usage.model_dump(),
        "latency_s": round(time.perf_counter() - started, 3),
        "raw": response.model_dump(),
        "meta": meta or {},
    }
    if log_path is not None:
        append_jsonl(log_path, record)
    return record
