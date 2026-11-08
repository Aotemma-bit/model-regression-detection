"""FastAPI application for model regression detection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import __version__
from app.config import Settings, load_settings
from app.errors import DatasetError, EvaluationError, MissingAPIKeyError, RunNotFoundError
from app.evaluator import EvaluationService, build_openai_client
from app.metrics import compare_to_baseline, compute_metrics
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    MetricsResponse,
    ResultsResponse,
)
from app.storage import ResultStore

logger = logging.getLogger("app.main")


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("app").setLevel(numeric)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = load_settings()
    configure_logging(settings.log_level)
    logger.info("service started env=%s model=%s", settings.app_env, settings.openai_model)
    yield


app = FastAPI(
    title="Model Regression Detection",
    version=__version__,
    summary="Classify a golden dataset and compare the run with a saved baseline.",
    lifespan=lifespan,
)


def get_settings() -> Settings:
    return load_settings()


def get_store(settings: Settings = Depends(get_settings)) -> ResultStore:
    store = ResultStore(settings.database_path)
    store.initialize()
    return store


def get_model_client(settings: Settings = Depends(get_settings)) -> Any:
    return build_openai_client(settings)


def _parse_evaluate_request(raw: bytes) -> EvaluateRequest:
    if not raw.strip():
        return EvaluateRequest()
    try:
        return EvaluateRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise _bad_request(exc) from exc


def _bad_request(exc: ValidationError) -> DatasetError:
    message = exc.errors()[0]["msg"]
    return DatasetError(message)


@app.exception_handler(MissingAPIKeyError)
async def missing_api_key(_request: Request, exc: MissingAPIKeyError) -> JSONResponse:
    logger.error("evaluation rejected because the API key is not configured")
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(DatasetError)
async def bad_dataset(_request: Request, exc: DatasetError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(EvaluationError)
async def evaluation_failed(_request: Request, exc: EvaluationError) -> JSONResponse:
    logger.error("evaluation failed error=%s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(RunNotFoundError)
async def run_missing(_request: Request, exc: RunNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        model=settings.openai_model,
        openai_configured=settings.api_key_configured,
        version=__version__,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: Request,
    settings: Settings = Depends(get_settings),
    store: ResultStore = Depends(get_store),
    client: Any = Depends(get_model_client),
) -> EvaluateResponse:
    payload = _parse_evaluate_request(await request.body())
    service = EvaluationService(settings, store)
    return service.evaluate(client, payload)


@app.get("/results", response_model=ResultsResponse)
def results(
    run_id: str | None = Query(default=None),
    store: ResultStore = Depends(get_store),
) -> ResultsResponse:
    run = store.require_run(run_id)
    stored = store.get_results(run.id)
    return ResultsResponse(
        run_id=run.id,
        model=run.model,
        created_at=run.created_at,
        is_baseline=run.is_baseline,
        count=len(stored),
        results=stored,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics(
    run_id: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    store: ResultStore = Depends(get_store),
) -> MetricsResponse:
    run = store.require_run(run_id)
    stored = store.get_results(run.id)
    current = compute_metrics(
        [item.expected_category for item in stored],
        [item.predicted_category for item in stored],
    )
    baseline_run = store.get_baseline_run()
    if baseline_run is None:
        baseline_results = None
        baseline_metrics = None
        baseline_id = None
    else:
        baseline_results = store.get_results(baseline_run.id)
        baseline_metrics = compute_metrics(
            [item.expected_category for item in baseline_results],
            [item.predicted_category for item in baseline_results],
        )
        baseline_id = baseline_run.id
    regression = compare_to_baseline(
        current,
        stored,
        baseline_metrics,
        baseline_results,
        baseline_run_id=baseline_id,
        drop_threshold=settings.regression_drop_threshold,
        current_run_id=run.id,
    )
    return MetricsResponse(
        run_id=run.id,
        model=run.model,
        created_at=run.created_at,
        is_baseline=run.is_baseline,
        metrics=current,
        regression=regression,
    )
