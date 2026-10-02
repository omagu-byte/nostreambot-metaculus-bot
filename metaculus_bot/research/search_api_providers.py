"""Bounded web-search API providers used by the research orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    FIRECRAWL_API_KEY_ENV,
    NIMBLE_AGENT_POLL_INTERVAL_S,
    NIMBLE_AGENT_TIMEOUT_S,
    NIMBLE_API_KEY_ENV,
    SEARCH_API_MAX_RESPONSE_BYTES,
    SEARCH_API_TIMEOUT_S,
    YDC_API_KEY_ENV,
)
from metaculus_bot.research.http_fetch import build_session, read_body_capped
from metaculus_bot.research.raw_log import record_raw_research

logger = logging.getLogger(__name__)

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
NIMBLE_AGENT_RUNS_URL = "https://sdk.nimbleway.com/v2/agents/runs"
YDC_SEARCH_URL = "https://api.you.com/v1/search"
_MAX_RESULTS = 8
_MAX_SNIPPET_CHARS = 1600


async def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    label: str,
    json_body: dict[str, Any] | None = None,
) -> Any:
    """Make one bounded provider request through the shared research transport."""
    async with build_session(timeout_s=SEARCH_API_TIMEOUT_S, connector_limit=4) as session:
        if method == "GET":
            request = session.get(url, headers=headers)
        else:
            request = session.post(url, headers=headers, json=json_body)
        async with request as response:
            body = await read_body_capped(response, max_bytes=SEARCH_API_MAX_RESPONSE_BYTES, label=label)
            if response.status >= 400:
                raise RuntimeError(f"{label} returned HTTP {response.status}")
            if body is None:
                raise RuntimeError(f"{label} returned an empty or oversized response")
            try:
                return json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"{label} returned invalid JSON") from exc


def _question_text(question: MetaculusQuestion) -> str:
    return str(question.question_text).strip()


def _result_items(payload: Any) -> list[Mapping[str, Any]]:
    """Find common result-list shapes used by search APIs."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("results", "web", "news", "data", "hits", "organic"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            nested = _result_items(value)
            if nested:
                return nested
    return []


def _format_search_results(payload: Any) -> str:
    """Render a compact, source-attributed list from structured search results."""
    lines: list[str] = []
    for item in _result_items(payload)[:_MAX_RESULTS]:
        title = str(item.get("title") or item.get("name") or "Untitled result").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        snippet = str(
            item.get("description")
            or item.get("snippet")
            or item.get("content")
            or item.get("markdown")
            or ""
        ).strip()
        if not url and not snippet:
            continue
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS].rstrip() + "..."
        lines.append(f"- [{title}]({url})\n  {snippet}" if url else f"- {title}\n  {snippet}")

    answer = payload.get("answer") if isinstance(payload, Mapping) else None
    if isinstance(answer, str) and answer.strip():
        lines.insert(0, answer.strip())
    return "\n".join(lines)


async def you_search_provider(question: MetaculusQuestion) -> str:
    """Search You.com and render its ranked results as cited research."""
    api_key = os.getenv(YDC_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{YDC_API_KEY_ENV} is required for You.com search")
    payload = await _request_json(
        "POST",
        YDC_SEARCH_URL,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json_body={"query": _question_text(question)},
        label="You.com Search",
    )
    result = _format_search_results(payload)
    record_raw_research(qid=getattr(question, "id_of_question", None), provider="ydc", payload=payload)
    return result


async def firecrawl_search_provider(question: MetaculusQuestion) -> str:
    """Search the web through Firecrawl's v2 Search endpoint."""
    api_key = os.getenv(FIRECRAWL_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{FIRECRAWL_API_KEY_ENV} is required for Firecrawl search")
    payload = await _request_json(
        "POST",
        FIRECRAWL_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={"query": _question_text(question), "limit": _MAX_RESULTS},
        label="Firecrawl Search",
    )
    result = _format_search_results(payload)
    record_raw_research(qid=getattr(question, "id_of_question", None), provider="firecrawl", payload=payload)
    return result


async def _nimble_research(question_text: str, api_key: str, qid: int | None) -> str:
    """Run and poll Nimble's asynchronous Web Search Agent API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started = await _request_json(
        "POST",
        NIMBLE_AGENT_RUNS_URL,
        headers=headers,
        json_body={"input": f"Research this forecasting question using current, credible sources. Cite sources.\n\n{question_text}"},
        label="Nimble Agent Search",
    )
    if not isinstance(started, Mapping):
        raise RuntimeError("Nimble Agent Search returned an invalid run response")
    run_id = started.get("id")
    agent_id = started.get("web_search_agent_id") or started.get("agent_id")
    if not isinstance(run_id, str) or not isinstance(agent_id, str):
        raise RuntimeError("Nimble Agent Search response omitted run or agent id")

    run_url = f"https://sdk.nimbleway.com/v2/agents/{agent_id}/runs/{run_id}"
    result_url = f"{run_url}/result"
    deadline = asyncio.get_running_loop().time() + NIMBLE_AGENT_TIMEOUT_S
    while True:
        status_payload = await _request_json("GET", run_url, headers=headers, label="Nimble Agent status")
        if not isinstance(status_payload, Mapping):
            raise RuntimeError("Nimble Agent Search returned an invalid status response")
        status = str(status_payload.get("status") or "").lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"Nimble Agent Search run ended with status {status}")
        if status_payload.get("is_active") is False or status in {"complete", "completed", "succeeded", "success"}:
            break
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Nimble Agent Search exceeded its wall-clock limit")
        await asyncio.sleep(min(NIMBLE_AGENT_POLL_INTERVAL_S, remaining))

    payload = await _request_json("GET", result_url, headers=headers, label="Nimble Agent result")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Nimble Agent Search returned an invalid result")
    output = payload.get("output")
    content = output.get("content") if isinstance(output, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Nimble Agent Search completed without research content")

    record_raw_research(qid=qid, provider="nimble", payload=payload)
    return content.strip()


async def nimble_agent_provider(question: MetaculusQuestion) -> str:
    """Run Nimble's cited Web Search Agent as a fallback research provider."""
    api_key = os.getenv(NIMBLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{NIMBLE_API_KEY_ENV} is required for Nimble Agent Search")
    return await asyncio.wait_for(
        _nimble_research(_question_text(question), api_key, getattr(question, "id_of_question", None)),
        timeout=NIMBLE_AGENT_TIMEOUT_S,
    )
