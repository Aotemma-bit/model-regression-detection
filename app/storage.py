"""SQLite storage for evaluation runs and the promoted baseline."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from app.errors import RunNotFoundError
from app.schemas import Category, ResultRecord, RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    is_baseline INTEGER NOT NULL DEFAULT 0,
    case_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    input_text TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    predicted_category TEXT,
    raw_response TEXT,
    correct INTEGER NOT NULL,
    description TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
"""


class ResultStore:
    """File-backed store. Each public method opens a short transaction."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.executescript(_SCHEMA)

    def save_run(
        self,
        run: RunRecord,
        results: list[ResultRecord],
        *,
        as_baseline: bool,
    ) -> None:
        with self._lock, self._connect() as connection:
            if as_baseline:
                connection.execute("UPDATE runs SET is_baseline = 0")
            connection.execute(
                """
                INSERT INTO runs (
                    id, model, created_at, accuracy, precision_score, recall_score,
                    f1_score, is_baseline, case_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.model,
                    run.created_at,
                    run.accuracy,
                    run.precision,
                    run.recall,
                    run.f1,
                    1 if as_baseline else 0,
                    run.case_count,
                ),
            )
            connection.executemany(
                """
                INSERT INTO results (
                    run_id, case_id, input_text, expected_category, predicted_category,
                    raw_response, correct, description, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.run_id,
                        item.case_id,
                        item.input,
                        item.expected_category,
                        item.predicted_category,
                        item.raw_response,
                        1 if item.correct else 0,
                        item.description,
                        item.error,
                    )
                    for item in results
                ],
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row) if row is not None else None

    def get_latest_run(self) -> RunRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        return _run_from_row(row) if row is not None else None

    def get_baseline_run(self) -> RunRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE is_baseline = 1 ORDER BY created_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def get_results(self, run_id: str) -> list[ResultRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [_result_from_row(row) for row in rows]

    def require_run(self, run_id: str | None = None) -> RunRecord:
        run = self.get_run(run_id) if run_id else self.get_latest_run()
        if run is None:
            if run_id:
                raise RunNotFoundError(f"Unknown run id: {run_id}")
            raise RunNotFoundError("No evaluation runs have been stored yet")
        return run

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        model=row["model"],
        created_at=row["created_at"],
        accuracy=row["accuracy"],
        precision=row["precision_score"],
        recall=row["recall_score"],
        f1=row["f1_score"],
        is_baseline=bool(row["is_baseline"]),
        case_count=row["case_count"],
    )


def _result_from_row(row: sqlite3.Row) -> ResultRecord:
    predicted = row["predicted_category"]
    return ResultRecord(
        run_id=row["run_id"],
        case_id=row["case_id"],
        input=row["input_text"],
        expected_category=row["expected_category"],
        predicted_category=predicted if predicted is None else _as_category(predicted),
        raw_response=row["raw_response"],
        correct=bool(row["correct"]),
        description=row["description"],
        error=row["error"],
    )


def _as_category(value: str) -> Category:
    if value not in {"billing", "technical", "account", "general"}:
        raise ValueError(f"stored category is invalid: {value}")
    return value  # type: ignore[return-value]
