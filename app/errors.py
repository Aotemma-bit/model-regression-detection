"""Domain errors raised by the evaluation service."""


class MissingAPIKeyError(RuntimeError):
    """OPENAI_API_KEY is absent or still set to a placeholder."""


class DatasetError(ValueError):
    """The golden dataset or an inline case list cannot be evaluated."""


class EvaluationError(RuntimeError):
    """The model could not be evaluated and nothing was stored."""


class RunNotFoundError(LookupError):
    """No stored evaluation run matches the request."""


class MalformedCompletionError(RuntimeError):
    """The model SDK returned a payload without readable text."""
