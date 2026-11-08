"""Remove configured secrets from messages before they are logged or stored."""

from __future__ import annotations

_MIN_SECRET_LENGTH = 8


def redact(text: str, secret: str | None) -> str:
    """Replace an exact secret with a marker.

    Short values are left unchanged so ordinary words in model output are not
    stripped. The marker is stable so tests can assert the secret is gone.
    """

    if not text or not secret:
        return text
    cleaned = secret.strip()
    if len(cleaned) < _MIN_SECRET_LENGTH or cleaned not in text:
        return text
    return text.replace(cleaned, "[REDACTED]")
