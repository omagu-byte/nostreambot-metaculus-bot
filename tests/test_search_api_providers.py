from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.research import search_api_providers as providers
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.provider_diagnostics import ProviderResult


@pytest.mark.asyncio
async def test_you_search_uses_ydc_key_and_formats_results(monkeypatch: pytest.MonkeyPatch, make_mock_question) -> None:
    question = make_mock_question(question_id=42, question_text="Will the bill pass?")
    monkeypatch.setenv("YDC_API_KEY", "test-ydc-key")
    request = AsyncMock(
        return_value={"results": [{"title": "Bill tracker", "url": "https://example.com/bill", "description": "Status"}]}
    )
    monkeypatch.setattr(providers, "_request_json", request)
    raw_log = MagicMock()
    monkeypatch.setattr(providers, "record_raw_research", raw_log)

    result = await providers.you_search_provider(question)

    assert "[Bill tracker](https://example.com/bill)" in result
    assert "Status" in result
    request.assert_awaited_once_with(
        "POST",
        providers.YDC_SEARCH_URL,
        headers={"X-API-Key": "test-ydc-key", "Content-Type": "application/json"},
        json_body={"query": "Will the bill pass?"},
        label="You.com Search",
    )
    raw_log.assert_called_once()


@pytest.mark.asyncio
async def test_firecrawl_uses_bearer_key_and_search_v2(monkeypatch: pytest.MonkeyPatch, make_mock_question) -> None:
    question = make_mock_question(question_id=42, question_text="Will the bill pass?")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")
    request = AsyncMock(
        return_value={"data": [{"title": "Official update", "url": "https://example.com/update", "description": "New facts"}]}
    )
    monkeypatch.setattr(providers, "_request_json", request)
    monkeypatch.setattr(providers, "record_raw_research", MagicMock())

    result = await providers.firecrawl_search_provider(question)

    assert "[Official update](https://example.com/update)" in result
    request.assert_awaited_once_with(
        "POST",
        providers.FIRECRAWL_SEARCH_URL,
        headers={"Authorization": "Bearer test-firecrawl-key", "Content-Type": "application/json"},
        json_body={"query": "Will the bill pass?", "limit": providers._MAX_RESULTS},
        label="Firecrawl Search",
    )


@pytest.mark.asyncio
async def test_nimble_agent_polls_then_reads_cited_result(monkeypatch: pytest.MonkeyPatch, make_mock_question) -> None:
    question = make_mock_question(question_id=42, question_text="Will the bill pass?")
    monkeypatch.setenv("NIMBLE_API_KEY", "test-nimble-key")
    request = AsyncMock(
        side_effect=[
            {"id": "run-1", "web_search_agent_id": "agent-1", "status": "completed"},
            {"status": "completed", "is_active": False},
            {"output": {"type": "text", "content": "Research findings [1]."}},
        ]
    )
    monkeypatch.setattr(providers, "_request_json", request)
    monkeypatch.setattr(providers, "record_raw_research", MagicMock())

    def passthrough_wait_for(awaitable, timeout: float):
        return awaitable

    monkeypatch.setattr(providers.asyncio, "wait_for", passthrough_wait_for)

    result = await providers.nimble_agent_provider(question)

    assert result == "Research findings [1]."
    assert request.await_args_list[0].args[:2] == ("POST", providers.NIMBLE_AGENT_RUNS_URL)
    assert request.await_args_list[0].kwargs["headers"]["Authorization"] == "Bearer test-nimble-key"
    assert request.await_args_list[0].kwargs["json_body"]["input"].endswith("Will the bill pass?")
    assert request.await_args_list[1].args[:2] == (
        "GET",
        "https://sdk.nimbleway.com/v2/agents/agent-1/runs/run-1",
    )
    assert request.await_args_list[2].args[:2] == (
        "GET",
        "https://sdk.nimbleway.com/v2/agents/agent-1/runs/run-1/result",
    )


def test_search_result_format_is_empty_for_no_results() -> None:
    assert providers._format_search_results({"results": []}) == ""


def _clear_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET",
        "EXA_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENROUTER_API_KEY",
        "RESEARCH_PROVIDER",
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "FINANCIAL_DATA_ENABLED",
        "TS_ANCHOR_ENABLED",
        "PREDICTION_MARKETS_ENABLED",
        "RESOLUTION_SOURCE_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def test_search_api_pair_is_primary_and_nimble_is_a_fallback(
    mock_general_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("YDC_API_KEY", "test-ydc")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl")
    monkeypatch.setenv("NIMBLE_API_KEY", "test-nimble")

    orchestrator = ResearchOrchestrator(default_llm=mock_general_llm, summarizer_llm=mock_general_llm)

    assert [name for _, name in orchestrator._select_research_providers()] == ["ydc", "firecrawl"]
    assert [name for _, name in orchestrator._select_research_fallback_providers(fast_path=False)] == ["nimble"]


def test_explicit_legacy_provider_keeps_native_search_as_a_supplement(
    mock_general_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("RESEARCH_PROVIDER", "exa")
    monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
    monkeypatch.setenv("YDC_API_KEY", "test-ydc")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl")
    orchestrator = ResearchOrchestrator(default_llm=mock_general_llm, summarizer_llm=mock_general_llm)
    monkeypatch.setattr(orchestrator, "_select_research_provider", lambda: (AsyncMock(), "exa"))

    assert [name for _, name in orchestrator._select_research_providers()] == ["exa", "native_search"]


def test_fallback_ladder_includes_each_configured_legacy_search_provider(
    mock_general_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_provider_environment(monkeypatch)
    for key in (
        "NIMBLE_API_KEY",
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET",
        "EXA_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(key, "configured")
    orchestrator = ResearchOrchestrator(default_llm=mock_general_llm, summarizer_llm=mock_general_llm)

    assert [name for _, name in orchestrator._select_research_fallback_providers(fast_path=False)] == [
        "nimble",
        "asknews",
        "exa",
        "perplexity",
        "openrouter",
    ]


@pytest.mark.asyncio
async def test_fallback_runs_only_when_both_search_primaries_are_empty(
    mock_general_llm, make_mock_question, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_provider_environment(monkeypatch)
    question = make_mock_question(question_id=42, question_text="Will the bill pass?")
    orchestrator = ResearchOrchestrator(default_llm=mock_general_llm, summarizer_llm=mock_general_llm)
    primary = [(AsyncMock(), "ydc"), (AsyncMock(), "firecrawl")]
    fallback = (AsyncMock(), "nimble")
    primary_results = [
        ProviderResult(name="ydc", status="empty", chars=0, latency_ms=1),
        ProviderResult(name="firecrawl", status="empty", chars=0, latency_ms=1),
    ]
    fallback_results = [ProviderResult(name="nimble", status="ok", chars=15, latency_ms=1)]
    run_providers = AsyncMock(side_effect=[("", primary_results, ""), ("## Web Research (Nimble Agent Search)\nNimble findings", fallback_results, "")])
    monkeypatch.setattr(orchestrator, "_select_research_providers", lambda fast_path=False: primary)
    monkeypatch.setattr(orchestrator, "_select_research_fallback_providers", lambda *, fast_path: [fallback])
    monkeypatch.setattr(orchestrator, "_run_providers_parallel", run_providers)
    monkeypatch.setattr(orchestrator, "_run_gap_fill_passes", AsyncMock(side_effect=lambda _q, research, **_kw: (research, None)))

    result = await orchestrator.run_research(question)

    assert "Nimble findings" in result
    assert run_providers.await_count == 2


@pytest.mark.asyncio
async def test_successful_search_primary_skips_fallback(
    mock_general_llm, make_mock_question, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_provider_environment(monkeypatch)
    question = make_mock_question(question_id=42, question_text="Will the bill pass?")
    orchestrator = ResearchOrchestrator(default_llm=mock_general_llm, summarizer_llm=mock_general_llm)
    primary = [(AsyncMock(), "ydc"), (AsyncMock(), "firecrawl")]
    run_providers = AsyncMock(
        return_value=(
            "## Web Research (You.com)\nUseful result",
            [ProviderResult(name="ydc", status="ok", chars=12, latency_ms=1)],
            "",
        )
    )
    fallback_selector = MagicMock(side_effect=AssertionError("fallback must not run"))
    monkeypatch.setattr(orchestrator, "_select_research_providers", lambda fast_path=False: primary)
    monkeypatch.setattr(orchestrator, "_select_research_fallback_providers", fallback_selector)
    monkeypatch.setattr(orchestrator, "_run_providers_parallel", run_providers)
    monkeypatch.setattr(orchestrator, "_run_gap_fill_passes", AsyncMock(side_effect=lambda _q, research, **_kw: (research, None)))

    result = await orchestrator.run_research(question)

    assert "Useful result" in result
    fallback_selector.assert_not_called()
