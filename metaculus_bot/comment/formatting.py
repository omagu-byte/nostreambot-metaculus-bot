"""Comment formatting shared by TemplateForecaster's section overrides."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from forecasting_tools import MetaculusQuestion, ReasonedPrediction

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.markers import (
    STACKED_MARKER_FALSE,
    STACKED_MARKER_TRUE,
    STACKER_OUTCOME_FALLBACK_LLM,
    STACKER_OUTCOME_FALLBACK_MEAN,
    STACKER_OUTCOME_FALLBACK_MEDIAN,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    STACKER_OUTCOME_SKIPPED_CONFIG_OFF,
    TOOLS_USED_MARKER_FALSE,
    TOOLS_USED_MARKER_TRUE,
    format_forecasters_used_marker,
    format_stacker_skip_reason_marker,
)
from metaculus_bot.comment.trimming import trim_comment, trim_section
from metaculus_bot.performance_analysis.parsing import (
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
)
from metaculus_bot.question_types import question_type_of
from metaculus_bot.research.section_format import PROVIDER_SECTION_HEADERS, detect_providers
from metaculus_bot.tool_runner import _feature_enabled as _tool_runner_feature_enabled

logger = logging.getLogger(__name__)

_FORECASTER_BULLET_RE = re.compile(r"(?m)^(\*Forecaster\s+\d+(?:\s+\([^)]*\))?\*:.*)$")
_PROVIDER_METHOD_LABELS = {
    "asknews": "AskNews",
    "exa": "Exa",
    "firecrawl": "Firecrawl",
    "financial_data": "financial and economic data",
    "gemini_search": "Google Search via Gemini",
    "native_search": "OpenRouter native web search",
    "nimble": "Nimble Agent Search",
    "openrouter": "OpenRouter research",
    "perplexity": "Perplexity",
    "prediction_market": "prediction-market data",
    "resolution_source": "resolution-source documents",
    "timeseries_anchor": "historical time-series data",
    "ydc": "You.com Search",
}


def _research_method_note(research_text: str) -> str:
    """Describe only research sources whose sections reached the forecast prompt."""
    providers = detect_providers(research_text)
    providers.sort(key=lambda name: research_text.find(PROVIDER_SECTION_HEADERS[name]))
    labels = [_PROVIDER_METHOD_LABELS.get(name, name) for name in providers]
    if not labels:
        return "No external research source contributed; forecasts use the question and model reasoning alone."
    source_text = ", ".join(labels[:-1]) + f" and {labels[-1]}" if len(labels) > 1 else labels[0]
    return f"Shared research from {source_text} was provided to each forecaster."


def format_research_summary_with_models(
    base_text: str,
    predictions: Sequence[ReasonedPrediction],
    report_number: int,
    research_text: str = "",
) -> str:
    """Inject model names and source-derived method notes, then trim to section limit."""
    model_names_by_index: dict[int, str] = {}
    for forecaster_number, forecast in enumerate(predictions, start=1):
        model_name = extract_model_display_name_from_reasoning(forecast.reasoning)
        if model_name is not None:
            model_names_by_index[forecaster_number] = model_name
    text = annotate_forecaster_bullets_with_models(base_text, model_names_by_index)
    method_note = _research_method_note(research_text)
    if method_note:
        text = _FORECASTER_BULLET_RE.sub(lambda match: f"{match.group(1)}\n  _Research method: {method_note}_", text)
    return trim_section(text, f"report_{report_number}_summary")


def format_main_research_section(base_text: str, report_number: int) -> str:
    """Trim the main research section to the configured section limit."""
    return trim_section(base_text, f"report_{report_number}_research")


def format_forecaster_rationales_section(base_text: str, report_number: int) -> str:
    """Trim the forecaster rationales section to the configured section limit."""
    return trim_section(base_text, f"report_{report_number}_rationales")


def _forecasters_used_suffix(n_used: int | None, n_configured: int | None) -> str:
    """The FORECASTERS_USED marker line (with a leading newline) when both counts
    are known, else ``""``.

    Emitted on every comment whose caller knows the ensemble size (the production
    ``_create_unified_explanation`` always does), so a comment with fewer than N
    bullets is self-describing — a dropped model is distinguishable from a roster
    change. Absent (rather than a fake ``0/0``) when a caller doesn't supply the
    counts, matching how pre-marker comments read as "unknown".
    """
    if n_used is None or n_configured is None:
        return ""
    return f"\n{format_forecasters_used_marker(n_used, n_configured)}"


def _stacker_outcome_markers(stacker_outcome: str) -> tuple[str, str]:
    """The (STACKER_OUTCOME, legacy STACKED) marker pair for one outcome.

    Raises on an unknown outcome rather than defaulting: a new outcome that silently
    published as ``STACKED: false`` would misreport whether the stacker ran.
    """
    match stacker_outcome:
        case "primary":
            return STACKER_OUTCOME_PRIMARY, STACKED_MARKER_TRUE
        case "fallback_llm":
            return STACKER_OUTCOME_FALLBACK_LLM, STACKED_MARKER_TRUE
        case "fallback_median":
            return STACKER_OUTCOME_FALLBACK_MEDIAN, STACKED_MARKER_FALSE
        case "fallback_mean":
            return STACKER_OUTCOME_FALLBACK_MEAN, STACKED_MARKER_FALSE
        case "skipped":
            return STACKER_OUTCOME_SKIPPED, STACKED_MARKER_FALSE
        case "skipped_config_off":
            return STACKER_OUTCOME_SKIPPED_CONFIG_OFF, STACKED_MARKER_FALSE
        case other:
            raise ValueError(f"Unknown stacker outcome {other!r}")


def build_unified_explanation(
    base_text: str,
    question: MetaculusQuestion,
    aggregation_strategy: AggregationStrategy,
    stacker_outcome: str | None,
    *,
    skip_reason: str | None = None,
    n_used: int | None = None,
    n_configured: int | None = None,
) -> str:
    """Build the final Metaculus comment with stacker/tools/ensemble markers appended.

    For non-stacking strategies, trims and returns (plus the ensemble marker). For
    STACKING / CONDITIONAL_STACKING, also appends STACKER_OUTCOME, legacy STACKED,
    and TOOLS_USED markers. ``skip_reason`` is additive: the skip paths in
    stacking_route supply it, and a STACKER_SKIP_REASON marker then rides directly
    under STACKER_OUTCOME so a plain ``skipped`` no longer conflates
    spread-below-threshold with the single-forecaster short-circuit; when ``None``
    (every non-skip outcome, and comments published before the field) the comment
    is unchanged. ``n_used`` / ``n_configured`` (contributed / configured
    forecasters) are keyword-only and additive: when both are supplied a
    FORECASTERS_USED marker rides the comment tail; when omitted the comment is
    unchanged (back-compat with callers that don't track ensemble size).
    """
    ensemble_suffix = _forecasters_used_suffix(n_used, n_configured)
    method_note = _research_method_note(base_text)
    aggregation_note = aggregation_strategy.value.replace("_", " ")
    if stacker_outcome is not None:
        aggregation_note += f" (outcome: {stacker_outcome.replace('_', ' ')})"
    final_method_note = (
        f"\n\n### Forecast Method\n{method_note}\n"
        f"Final forecast: {aggregation_note} aggregation of the individual forecasts above."
    )
    if aggregation_strategy not in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
        return trim_comment(f"{base_text}{final_method_note}{ensemble_suffix}")

    assert stacker_outcome is not None, (
        "stacker_outcome must be provided for STACKING/CONDITIONAL_STACKING strategies; "
        "every reachable code path in _aggregate_predictions sets it. Missing entry = real bug."
    )

    outcome_marker, legacy_marker = _stacker_outcome_markers(stacker_outcome)
    qtype = question_type_of(question)

    skip_reason_suffix = "" if skip_reason is None else f"\n{format_stacker_skip_reason_marker(skip_reason)}"
    tools_marker = TOOLS_USED_MARKER_TRUE if _tool_runner_feature_enabled(qtype) else TOOLS_USED_MARKER_FALSE
    return trim_comment(
        f"{base_text}{final_method_note}\n{outcome_marker}{skip_reason_suffix}\n{legacy_marker}\n{tools_marker}{ensemble_suffix}\n"
    )


__all__ = [
    "build_unified_explanation",
    "format_forecaster_rationales_section",
    "format_main_research_section",
    "format_research_summary_with_models",
]
