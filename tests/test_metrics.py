"""Metric math and regression rules. These tests never call a model."""

from __future__ import annotations

from app.metrics import compare_to_baseline, compute_metrics
from app.schemas import ResultRecord


def _result(
    case_id: str,
    expected: str,
    predicted: str | None,
    *,
    correct: bool | None = None,
    run_id: str = "run",
) -> ResultRecord:
    return ResultRecord(
        run_id=run_id,
        case_id=case_id,
        input=f"prompt {case_id}",
        expected_category=expected,  # type: ignore[arg-type]
        predicted_category=predicted,  # type: ignore[arg-type]
        raw_response=predicted,
        correct=predicted == expected if correct is None else correct,
        description=f"check {case_id}",
    )


def test_perfect_predictions_score_one() -> None:
    labels = ["billing", "technical", "account", "general"]
    score = compute_metrics(labels, labels)  # type: ignore[arg-type]
    assert score.accuracy == 1.0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.support == 4
    assert score.per_class["billing"].support == 1


def test_macro_scores_ignore_unused_classes() -> None:
    score = compute_metrics(
        ["billing", "billing"],
        ["billing", "technical"],
    )
    assert score.accuracy == 0.5
    assert score.precision == 0.5
    assert score.recall == 0.25
    assert score.f1 == round((2 / 3) / 2, 6)
    assert set(score.per_class) == {"billing", "technical"}
    assert score.per_class["billing"].recall == 0.5
    assert score.per_class["technical"].precision == 0.0


def test_unparsed_prediction_is_incorrect() -> None:
    score = compute_metrics(["account"], [None])
    assert score.accuracy == 0.0
    assert score.per_class["account"].recall == 0.0
    assert score.support == 1


def test_metric_drop_and_case_flip_are_regressions() -> None:
    baseline_results = [
        _result("a", "billing", "billing", run_id="base"),
        _result("b", "technical", "technical", run_id="base"),
    ]
    current_results = [
        _result("a", "billing", "general", run_id="now"),
        _result("b", "technical", "technical", run_id="now"),
    ]
    baseline = compute_metrics(
        [item.expected_category for item in baseline_results],
        [item.predicted_category for item in baseline_results],
    )
    current = compute_metrics(
        [item.expected_category for item in current_results],
        [item.predicted_category for item in current_results],
    )
    report = compare_to_baseline(
        current,
        current_results,
        baseline,
        baseline_results,
        baseline_run_id="base",
        drop_threshold=0.0,
        current_run_id="now",
    )
    assert report.regressed is True
    assert report.baseline_run_id == "base"
    accuracy = next(item for item in report.metric_regressions if item.metric == "accuracy")
    assert accuracy.regressed is True
    assert accuracy.delta < 0
    assert [item.case_id for item in report.case_regressions] == ["a"]
    assert report.case_regressions[0].baseline_prediction == "billing"
    assert report.case_regressions[0].current_prediction == "general"


def test_drop_inside_threshold_is_not_a_metric_regression() -> None:
    baseline_results = [_result("a", "billing", "billing"), _result("b", "technical", "technical")]
    current_results = [_result("a", "billing", "general"), _result("b", "technical", "technical")]
    baseline = compute_metrics(
        [item.expected_category for item in baseline_results],
        [item.predicted_category for item in baseline_results],
    )
    current = compute_metrics(
        [item.expected_category for item in current_results],
        [item.predicted_category for item in current_results],
    )
    report = compare_to_baseline(
        current,
        current_results,
        baseline,
        baseline_results,
        baseline_run_id="base",
        drop_threshold=0.7,
    )
    assert all(item.regressed is False for item in report.metric_regressions)
    assert report.regressed is True
    assert len(report.case_regressions) == 1


def test_missing_baseline_is_not_a_regression() -> None:
    current_results = [_result("a", "general", "general")]
    current = compute_metrics(["general"], ["general"])
    report = compare_to_baseline(
        current,
        current_results,
        None,
        None,
        baseline_run_id=None,
        drop_threshold=0.0,
    )
    assert report.baseline_available is False
    assert report.regressed is False
    assert report.metric_regressions == []


def test_previously_wrong_case_is_not_a_new_regression() -> None:
    baseline_results = [_result("a", "account", "billing")]
    current_results = [_result("a", "account", "general")]
    baseline = compute_metrics(["account"], ["billing"])
    current = compute_metrics(["account"], ["general"])
    report = compare_to_baseline(
        current,
        current_results,
        baseline,
        baseline_results,
        baseline_run_id="base",
        drop_threshold=0.0,
    )
    assert report.case_regressions == []
