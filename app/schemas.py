"""Pydantic models shared by the API, storage, and metrics."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["billing", "technical", "account", "general"]
CATEGORIES: tuple[Category, ...] = ("billing", "technical", "account", "general")


class TestCase(BaseModel):
    """One classification example with the category the model should return."""

    __test__: ClassVar[bool] = False

    id: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1)
    expected_category: Category
    description: str = Field(min_length=1)


class EvaluateRequest(BaseModel):
    """Optional inline cases. When omitted, the golden dataset is evaluated."""

    cases: list[TestCase] | None = None
    case_ids: list[str] | None = None
    save_as_baseline: bool = False

    @model_validator(mode="after")
    def cases_and_ids_are_exclusive(self) -> EvaluateRequest:
        if self.cases is not None and self.case_ids is not None:
            raise ValueError("pass cases or case_ids, not both")
        if self.case_ids is not None and len(self.case_ids) == 0:
            raise ValueError("case_ids must not be empty")
        if self.cases is not None and len(self.cases) == 0:
            raise ValueError("cases must not be empty")
        return self


class ClassScore(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MetricScore(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    per_class: dict[str, ClassScore]


class MetricDelta(BaseModel):
    metric: Literal["accuracy", "precision", "recall", "f1"]
    baseline: float
    current: float
    delta: float
    regressed: bool


class CaseRegression(BaseModel):
    case_id: str
    expected_category: Category
    baseline_prediction: Category | None
    current_prediction: Category | None
    description: str


class RegressionReport(BaseModel):
    baseline_available: bool
    baseline_run_id: str | None
    regressed: bool
    metric_regressions: list[MetricDelta]
    case_regressions: list[CaseRegression]
    detail: str | None = None


class ResultRecord(BaseModel):
    run_id: str
    case_id: str
    input: str
    expected_category: Category
    predicted_category: Category | None
    raw_response: str | None
    correct: bool
    description: str
    error: str | None = None


class RunRecord(BaseModel):
    id: str
    model: str
    created_at: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    is_baseline: bool
    case_count: int


class EvaluateResponse(BaseModel):
    run_id: str
    model: str
    saved_as_baseline: bool
    metrics: MetricScore
    regression: RegressionReport
    results: list[ResultRecord]


class ResultsResponse(BaseModel):
    run_id: str
    model: str
    created_at: str
    is_baseline: bool
    count: int
    results: list[ResultRecord]


class MetricsResponse(BaseModel):
    run_id: str
    model: str
    created_at: str
    is_baseline: bool
    metrics: MetricScore
    regression: RegressionReport


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str
    model: str
    openai_configured: bool
    version: str
