"""Classification metrics and baseline regression checks.

Macro precision, recall, and F1 follow the usual multiclass definition: each
category that appears in the expected labels or the predictions contributes
equally. Categories with no support and no predictions are left out of the
average so a partial run is not punished for classes it did not exercise.
Accuracy always counts every example. Published scores are rounded to six
decimal places so baseline comparisons use the same numbers the API returns.
"""

from __future__ import annotations

from app.schemas import (
    CATEGORIES,
    CaseRegression,
    Category,
    ClassScore,
    MetricDelta,
    MetricScore,
    RegressionReport,
    ResultRecord,
)

_METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
_SCORE_DECIMALS = 6


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _published(value: float) -> float:
    return round(value, _SCORE_DECIMALS)


def compute_metrics(expected: list[Category], predicted: list[Category | None]) -> MetricScore:
    """Score one run. Predicted None counts as incorrect and not as a class."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("at least one prediction is required")

    per_class: dict[str, ClassScore] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for category in CATEGORIES:
        support = sum(label == category for label in expected)
        true_positive = sum(
            truth == category and guess == category for truth, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            truth != category and guess == category for truth, guess in zip(expected, predicted, strict=True)
        )
        false_negative = support - true_positive
        if support == 0 and true_positive == 0 and false_positive == 0:
            continue
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[category] = ClassScore(
            precision=_published(precision),
            recall=_published(recall),
            f1=_published(f1),
            support=support,
        )

    correct = sum(truth == guess for truth, guess in zip(expected, predicted, strict=True))
    class_count = len(precisions)
    return MetricScore(
        accuracy=_published(_ratio(correct, len(expected))),
        precision=_published(sum(precisions) / class_count),
        recall=_published(sum(recalls) / class_count),
        f1=_published(sum(f1_scores) / class_count),
        support=len(expected),
        per_class=per_class,
    )


def compare_to_baseline(
    current: MetricScore,
    current_results: list[ResultRecord],
    baseline: MetricScore | None,
    baseline_results: list[ResultRecord] | None,
    *,
    baseline_run_id: str | None,
    drop_threshold: float,
    current_run_id: str | None = None,
) -> RegressionReport:
    """Compare a run with the saved baseline.

    A metric regresses when its drop is greater than ``drop_threshold``.
    A case regresses when the baseline got it right and the current run did not.
    """

    if baseline is None or baseline_results is None or baseline_run_id is None:
        return RegressionReport(
            baseline_available=False,
            baseline_run_id=None,
            regressed=False,
            metric_regressions=[],
            case_regressions=[],
            detail="No baseline has been saved yet. Re-run evaluation with save_as_baseline set to true.",
        )

    deltas: list[MetricDelta] = []
    for name in _METRIC_NAMES:
        before = getattr(baseline, name)
        after = getattr(current, name)
        drop = _published(before - after)
        deltas.append(
            MetricDelta(
                metric=name,
                baseline=before,
                current=after,
                delta=_published(after - before),
                regressed=drop > drop_threshold,
            )
        )

    previous_by_id = {item.case_id: item for item in baseline_results}
    case_regressions: list[CaseRegression] = []
    for item in current_results:
        previous = previous_by_id.get(item.case_id)
        if previous is None:
            continue
        if previous.correct and not item.correct:
            case_regressions.append(
                CaseRegression(
                    case_id=item.case_id,
                    expected_category=item.expected_category,
                    baseline_prediction=previous.predicted_category,
                    current_prediction=item.predicted_category,
                    description=item.description,
                )
            )

    same_run = current_run_id is not None and current_run_id == baseline_run_id
    detail = "This run is the saved baseline, so there is no earlier baseline to compare." if same_run else None
    regressed = any(item.regressed for item in deltas) or bool(case_regressions)
    return RegressionReport(
        baseline_available=True,
        baseline_run_id=baseline_run_id,
        regressed=regressed,
        metric_regressions=deltas,
        case_regressions=case_regressions,
        detail=detail,
    )
