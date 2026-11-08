"""Send classification prompts to a chat model and score the responses."""

from __future__ import annotations

import logging
import re
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI, OpenAIError

from app.config import Settings
from app.dataset import select_cases
from app.errors import EvaluationError, MalformedCompletionError, MissingAPIKeyError
from app.metrics import compare_to_baseline, compute_metrics
from app.redaction import redact
from app.schemas import (
    CATEGORIES,
    Category,
    EvaluateRequest,
    EvaluateResponse,
    ResultRecord,
    RunRecord,
)
from app.storage import ResultStore

logger = logging.getLogger("app.evaluator")

SYSTEM_PROMPT = (
    "You classify customer support messages into exactly one category. "
    "Reply with a single lowercase word and nothing else. "
    "Allowed categories: billing, technical, account, general."
)

_CATEGORY_PATTERN = re.compile(r"\b(billing|technical|account|general)\b")


def build_openai_client(settings: Settings) -> OpenAI:
    """Construct the official client. Callers must not log the returned object."""

    if not settings.api_key_configured:
        raise MissingAPIKeyError("OPENAI_API_KEY is not configured")
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def parse_category(text: str | None) -> Category | None:
    """Extract a category from a model reply.

    An exact first token wins. Otherwise the first allowed word in the reply
    is used. Replies that name no category stay unmatched and score as incorrect.
    """

    if text is None:
        return None
    lowered = text.strip().lower()
    if not lowered:
        return None
    first_token = re.split(r"\s+", lowered, maxsplit=1)[0]
    first_clean = re.sub(r"[^a-z]", "", first_token)
    if first_clean in CATEGORIES:
        return first_clean  # type: ignore[return-value]
    match = _CATEGORY_PATTERN.search(lowered)
    if match is None:
        return None
    return match.group(1)  # type: ignore[return-value]


def request_completion(client: Any, settings: Settings, prompt: str) -> str:
    """Call chat completions and return the assistant text."""

    completion = client.chat.completions.create(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise MalformedCompletionError("model returned an unreadable completion") from exc
    if content is None:
        return ""
    return str(content)


class EvaluationService:
    """Run the golden dataset, or an inline set, and persist the outcome."""

    def __init__(self, settings: Settings, store: ResultStore) -> None:
        self.settings = settings
        self.store = store

    def evaluate(self, client: Any, request: EvaluateRequest) -> EvaluateResponse:
        cases = select_cases(
            request.cases,
            dataset_path=self.settings.dataset_path,
            case_ids=request.case_ids,
        )
        previous_baseline = self.store.get_baseline_run()
        previous_results = self.store.get_results(previous_baseline.id) if previous_baseline else None
        previous_metrics = (
            compute_metrics(
                [item.expected_category for item in previous_results],
                [item.predicted_category for item in previous_results],
            )
            if previous_results
            else None
        )

        run_id = uuid.uuid4().hex
        pending: list[ResultRecord] = []
        call_failures = 0
        for case in cases:
            raw_response: str | None = None
            predicted: Category | None = None
            error: str | None = None
            try:
                raw_response = request_completion(client, self.settings, case.input)
                predicted = parse_category(raw_response)
            except (OpenAIError, MalformedCompletionError) as exc:
                call_failures += 1
                error = redact(str(exc), self.settings.openai_api_key)
                logger.error(
                    "model call failed case_id=%s error=%s",
                    case.id,
                    redact(traceback.format_exc(), self.settings.openai_api_key),
                )
            pending.append(
                ResultRecord(
                    run_id=run_id,
                    case_id=case.id,
                    input=case.input,
                    expected_category=case.expected_category,
                    predicted_category=predicted,
                    raw_response=raw_response,
                    correct=predicted == case.expected_category,
                    description=case.description,
                    error=error,
                )
            )

        if call_failures == len(cases):
            raise EvaluationError("Every model call failed. No results were stored.")

        metrics = compute_metrics(
            [item.expected_category for item in pending],
            [item.predicted_category for item in pending],
        )
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run = RunRecord(
            id=run_id,
            model=self.settings.openai_model,
            created_at=created_at,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            is_baseline=request.save_as_baseline,
            case_count=len(pending),
        )
        self.store.save_run(run, pending, as_baseline=request.save_as_baseline)
        regression = compare_to_baseline(
            metrics,
            pending,
            previous_metrics,
            previous_results,
            baseline_run_id=previous_baseline.id if previous_baseline else None,
            drop_threshold=self.settings.regression_drop_threshold,
            current_run_id=run_id,
        )
        if regression.regressed:
            logger.warning(
                "regression detected run_id=%s baseline_run_id=%s case_regressions=%s",
                run_id,
                regression.baseline_run_id,
                [item.case_id for item in regression.case_regressions],
            )
        else:
            logger.info(
                "evaluation stored run_id=%s cases=%s accuracy=%s baseline=%s",
                run_id,
                len(pending),
                metrics.accuracy,
                request.save_as_baseline,
            )
        return EvaluateResponse(
            run_id=run_id,
            model=self.settings.openai_model,
            saved_as_baseline=request.save_as_baseline,
            metrics=metrics,
            regression=regression,
            results=pending,
        )
