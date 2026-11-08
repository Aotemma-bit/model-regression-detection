"""Load classification cases from JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.errors import DatasetError
from app.schemas import TestCase

MAX_CASES_PER_RUN = 200


def load_dataset(path: Path) -> list[TestCase]:
    """Read a JSONL golden dataset and reject malformed rows."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"Dataset file not found: {dataset_path}")

    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    try:
        raw_text = dataset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Could not read dataset: {dataset_path}") from exc

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"Line {line_number} must be a JSON object")
        try:
            case = TestCase.model_validate(payload)
        except ValidationError as exc:
            raise DatasetError(f"Invalid test case on line {line_number}: {exc.errors()[0]['msg']}") from exc
        if case.id in seen_ids:
            raise DatasetError(f"Duplicate case id {case.id!r} on line {line_number}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetError(f"Dataset is empty: {dataset_path}")
    return cases


def select_cases(
    cases: list[TestCase] | None,
    *,
    dataset_path: Path,
    case_ids: list[str] | None,
) -> list[TestCase]:
    """Choose inline cases or a subset of the golden dataset."""

    if cases is None:
        selected = load_dataset(dataset_path)
        if case_ids is not None:
            wanted = set(case_ids)
            selected = [case for case in selected if case.id in wanted]
            missing = wanted - {case.id for case in selected}
            if missing:
                ordered = ", ".join(sorted(missing))
                raise DatasetError(f"Unknown case ids: {ordered}")
    else:
        seen: set[str] = set()
        for case in cases:
            if case.id in seen:
                raise DatasetError(f"Duplicate case id {case.id!r}")
            seen.add(case.id)
        selected = list(cases)

    if len(selected) > MAX_CASES_PER_RUN:
        raise DatasetError(f"A single run supports at most {MAX_CASES_PER_RUN} cases")
    return selected
