"""In-memory chat clients used by tests. Nothing here talks to the network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class ScriptedCompletions:
    def __init__(self, owner: ScriptedClient) -> None:
        self.owner = owner

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if not self.owner.outputs:
            raise RuntimeError("scripted client has no responses left")
        self.owner.calls.append(kwargs)
        text = self.owner.outputs.pop(0)
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedClient:
    """Return queued assistant strings from chat.completions.create."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=ScriptedCompletions(self))


class FailingCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        raise self.error


class FailingClient:
    def __init__(self, error: Exception) -> None:
        self.chat = SimpleNamespace(completions=FailingCompletions(error))


class EmptyCompletionClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(choices=[])
