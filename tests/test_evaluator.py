"""Evaluation flow with a fake chat client. No API key and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openai import OpenAIError

from app.config import PROJECT_ROOT, Settings
from app.dataset import load_dataset
from app.errors import DatasetError, EvaluationError, MissingAPIKeyError
from app.evaluator import EvaluationService, build_openai_client, parse_category
from app.schemas import CATEGORIES, EvaluateRequest, TestCase
from app.storage import ResultStore
from tests.fakes import EmptyCompletionClient, FailingClient, ScriptedClient


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    dataset = tmp_path / "cases.jsonl"
    values: dict[str, object] = {
        "openai_api_key": "",
        "openai_model": "gpt-test",
        "app_env": "test",
        "database_path": tmp_path / "eval.sqlite3",
        "dataset_path": dataset,
        "log_level": "INFO",
        "regression_drop_threshold": 0.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_cases(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sample_rows() -> list[dict[str, str]]:
    return [
        {
            "id": "bill-001",
            "input": "I need a refund.",
            "expected_category": "billing",
            "description": "Refunds are billing.",
        },
        {
            "id": "tech-001",
            "input": "The API returns 500.",
            "expected_category": "technical",
            "description": "Server errors are technical.",
        },
    ]


def test_golden_dataset_has_required_coverage() -> None:
    cases = load_dataset(PROJECT_ROOT / "data" / "golden_dataset.jsonl")
    assert len(cases) >= 20
    assert {case.expected_category for case in cases} == set(CATEGORIES)
    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        assert case.input
        assert case.description


def test_parse_category_accepts_exact_and_wrapped_replies() -> None:
    assert parse_category("Billing") == "billing"
    assert parse_category("technical.") == "technical"
    assert parse_category("Category: account") == "account"
    assert parse_category("I think this is general, not billing") == "general"
    assert parse_category("accounting question") is None
    assert parse_category("   ") is None
    assert parse_category(None) is None


def test_dataset_rejects_bad_json_unknown_category_and_duplicates(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Invalid JSON"):
        load_dataset(broken)

    unknown = tmp_path / "unknown.jsonl"
    _write_cases(
        unknown,
        [
            {
                "id": "x-1",
                "input": "hello",
                "expected_category": "sales",
                "description": "not a supported category",
            }
        ],
    )
    with pytest.raises(DatasetError, match="Invalid test case"):
        load_dataset(unknown)

    duplicates = tmp_path / "duplicates.jsonl"
    row = _sample_rows()[0]
    _write_cases(duplicates, [row, row])
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(duplicates)


def test_placeholder_key_does_not_build_a_client(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key="your_openai_api_key_here")
    with pytest.raises(MissingAPIKeyError, match="not configured"):
        build_openai_client(settings)
    assert "your_openai_api_key_here" not in repr(settings)


def test_evaluation_stores_responses_and_detects_regression(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key="sk-test-secret-value")
    _write_cases(settings.dataset_path, _sample_rows())
    store = ResultStore(settings.database_path)
    store.initialize()
    service = EvaluationService(settings, store)

    first = service.evaluate(
        ScriptedClient(["billing", "technical"]),
        EvaluateRequest(save_as_baseline=True),
    )
    assert first.saved_as_baseline is True
    assert first.metrics.accuracy == 1.0
    assert first.regression.baseline_available is False
    assert first.results[0].raw_response == "billing"
    assert store.get_baseline_run() is not None
    assert store.get_baseline_run().id == first.run_id  # type: ignore[union-attr]

    second_client = ScriptedClient(["general", "Technical"])
    second = service.evaluate(second_client, EvaluateRequest())
    assert second.regression.regressed is True
    assert [item.case_id for item in second.regression.case_regressions] == ["bill-001"]
    assert second.results[1].predicted_category == "technical"
    assert second.results[1].correct is True
    sent_messages = second_client.calls[0]["messages"]
    assert sent_messages[1]["content"] == "I need a refund."
    assert "sk-test-secret-value" not in json.dumps(second.model_dump())

    reloaded = ResultStore(settings.database_path)
    stored = reloaded.get_results(second.run_id)
    assert [item.case_id for item in stored] == ["bill-001", "tech-001"]
    assert reloaded.get_baseline_run().id == first.run_id  # type: ignore[union-attr]


def test_inline_cases_do_not_require_the_dataset_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = ResultStore(settings.database_path)
    store.initialize()
    service = EvaluationService(settings, store)
    case = TestCase(
        id="gen-001",
        input="What are your hours?",
        expected_category="general",
        description="Hours are general.",
    )
    report = service.evaluate(
        ScriptedClient(["general"]),
        EvaluateRequest(cases=[case], save_as_baseline=True),
    )
    assert report.metrics.support == 1
    assert report.results[0].correct is True


def test_unknown_case_filter_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_cases(settings.dataset_path, _sample_rows())
    store = ResultStore(settings.database_path)
    store.initialize()
    service = EvaluationService(settings, store)
    with pytest.raises(DatasetError, match="Unknown case ids"):
        service.evaluate(ScriptedClient([]), EvaluateRequest(case_ids=["missing"]))


def test_total_model_failure_stores_nothing_and_redacts_the_key(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    secret = "sk-test-secret-value"
    settings = _settings(tmp_path, openai_api_key=secret)
    _write_cases(settings.dataset_path, _sample_rows())
    store = ResultStore(settings.database_path)
    store.initialize()
    service = EvaluationService(settings, store)
    client = FailingClient(OpenAIError(f"unauthorized {secret}"))
    with pytest.raises(EvaluationError, match="No results were stored"):
        service.evaluate(client, EvaluateRequest(save_as_baseline=True))
    assert store.get_latest_run() is None
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_partial_failure_is_stored_as_incorrect(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_cases(settings.dataset_path, _sample_rows())
    store = ResultStore(settings.database_path)
    store.initialize()

    class MixedClient:
        def __init__(self) -> None:
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs: object):
            self.calls += 1
            if self.calls == 1:
                raise OpenAIError("temporary outage")
            message = type("Message", (), {"content": "technical"})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    service = EvaluationService(settings, store)
    report = service.evaluate(MixedClient(), EvaluateRequest())
    assert report.results[0].correct is False
    assert report.results[0].error == "temporary outage"
    assert report.results[1].correct is True
    assert store.get_results(report.run_id)[0].predicted_category is None


def test_empty_completion_fails_the_case(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_cases(settings.dataset_path, [_sample_rows()[0]])
    store = ResultStore(settings.database_path)
    store.initialize()
    service = EvaluationService(settings, store)
    with pytest.raises(EvaluationError):
        service.evaluate(EmptyCompletionClient(), EvaluateRequest())
    assert store.get_latest_run() is None
