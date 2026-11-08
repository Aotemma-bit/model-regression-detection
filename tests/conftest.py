"""Keep API dependency overrides from leaking between tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    from app.main import app

    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
