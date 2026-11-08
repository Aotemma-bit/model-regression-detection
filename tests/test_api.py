"""HTTP API tests. The OpenAI client is replaced and no key is required."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_model_client, get_settings
from tests.fakes import ScriptedClient


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "",
        "openai_model": "gpt-test",
        "app_env": "test",
        "database_path": tmp_path / "api.sqlite3",
        "dataset_path": tmp_path / "cases.jsonl",
        "log_level": "WARNING",
        "regression_drop_threshold": 0.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "id": "bill-001",
            "input": "Refund my invoice.",
            "expected_category": "billing",
            "description": "Refunds are billing.",
        },
        {
            "id": "acct-001",
            "input": "Reset my password.",
            "expected_category": "account",
            "description": "Password resets are account.",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_health_hides_the_api_key(tmp_path: Path) -> None:
    secret = "sk-test-secret-value"
    settings = _settings(tmp_path, openai_api_key=secret)
    app.dependency_overrides[get_settings] = lambda: settings
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["openai_configured"] is True
    assert body["model"] == "gpt-test"
    assert secret not in response.text


def test_evaluate_requires_a_configured_key(tmp_path: Path) -> None:
    settings = _settings(tmp_path, openai_api_key="your_openai_api_key_here")
    _write_dataset(settings.dataset_path)
    app.dependency_overrides[get_settings] = lambda: settings
    response = TestClient(app).post("/evaluate", json={})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    assert "your_openai_api_key_here" not in response.text


def test_evaluate_results_and_metrics_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_dataset(settings.dataset_path)
    scripted = ScriptedClient(["billing", "account", "general", "account"])
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_client] = lambda: scripted
    client = TestClient(app)

    created = client.post("/evaluate", json={"save_as_baseline": True})
    assert created.status_code == 200
    created_body = created.json()
    assert created_body["metrics"]["accuracy"] == 1.0
    assert created_body["saved_as_baseline"] is True
    assert created_body["regression"]["baseline_available"] is False

    regressed = client.post("/evaluate", json={})
    assert regressed.status_code == 200
    regressed_body = regressed.json()
    assert regressed_body["regression"]["regressed"] is True
    assert regressed_body["regression"]["case_regressions"][0]["case_id"] == "bill-001"

    listed = client.get("/results")
    assert listed.status_code == 200
    assert listed.json()["count"] == 2
    assert listed.json()["results"][0]["expected_category"] == "billing"
    assert listed.json()["is_baseline"] is False

    baseline = client.get("/results", params={"run_id": created_body["run_id"]})
    assert baseline.status_code == 200
    assert baseline.json()["is_baseline"] is True

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    metrics_body = metrics.json()
    assert metrics_body["run_id"] == regressed_body["run_id"]
    assert metrics_body["regression"]["baseline_run_id"] == created_body["run_id"]
    assert metrics_body["regression"]["regressed"] is True
    accuracy = next(item for item in metrics_body["regression"]["metric_regressions"] if item["metric"] == "accuracy")
    assert accuracy["current"] < accuracy["baseline"]

    missing = client.get("/results", params={"run_id": "does-not-exist"})
    assert missing.status_code == 404


def test_metrics_before_any_run_is_not_found(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    response = TestClient(app).get("/metrics")
    assert response.status_code == 404


def test_invalid_inline_case_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_client] = lambda: ScriptedClient(["billing"])
    response = TestClient(app).post(
        "/evaluate",
        json={
            "cases": [
                {
                    "id": "x",
                    "input": "hello",
                    "expected_category": "sales",
                    "description": "unsupported",
                }
            ]
        },
    )
    assert response.status_code == 400
