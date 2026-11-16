"""Typed exceptions raised across the ``promptopt`` package.

Every exception the library raises deliberately (as opposed to letting a
dependency's own exception propagate) is defined here so callers can catch
one module instead of guessing at exception types scattered across the
codebase.

:seealso: ``lib_structure.md`` -> ``errors.py`` in the package layout table.
"""

from __future__ import annotations


class PromptOptimizerError(Exception):
    """Base class for every exception raised deliberately by ``promptopt``.

    :param message: Human-readable description of what went wrong.
    :type message: str
    """

    def __init__(self, message: str) -> None:
        """Initialize the error with a human-readable message.

        :param message: Human-readable description of what went wrong.
        :type message: str
        """
        super().__init__(message)
        self.message = message


class BudgetExceeded(PromptOptimizerError):
    """Raised when a run, stage, or request would exceed a configured ceiling.

    :param message: Human-readable description of which budget was exceeded.
    :type message: str
    :param scope: What the ceiling applies to, e.g. ``"run"`` or a stage name.
    :type scope: str
    :param limit: The configured ceiling that was hit.
    :type limit: float
    :param spent: The amount already spent (tokens or USD) at the time of the check.
    :type spent: float
    """

    def __init__(self, message: str, *, scope: str, limit: float, spent: float) -> None:
        """Initialize the error with budget context.

        :param message: Human-readable description of which budget was exceeded.
        :type message: str
        :param scope: What the ceiling applies to, e.g. ``"run"`` or a stage name.
        :type scope: str
        :param limit: The configured ceiling that was hit.
        :type limit: float
        :param spent: The amount already spent at the time of the check.
        :type spent: float
        """
        super().__init__(message)
        self.scope = scope
        self.limit = limit
        self.spent = spent


class RefusalError(PromptOptimizerError):
    """Raised when the model declines to answer (``stop_reason == "refusal"``).

    :param message: Human-readable description of the refusal.
    :type message: str
    :param model: The model that issued the refusal.
    :type model: str
    """

    def __init__(self, message: str, *, model: str) -> None:
        """Initialize the error with the offending model's identifier.

        :param message: Human-readable description of the refusal.
        :type message: str
        :param model: The model that issued the refusal.
        :type model: str
        """
        super().__init__(message)
        self.model = model


class ConvergenceStalled(PromptOptimizerError):
    """Raised when the optimization loop cannot make further progress.

    Distinct from a normal stopping criterion (see ``loop/convergence.py``):
    this signals an abnormal stall, e.g. every expansion attempt produced
    zero novel examples across repeated iterations.

    :param message: Human-readable description of the stall.
    :type message: str
    :param iteration: The iteration index at which the stall was detected.
    :type iteration: int
    """

    def __init__(self, message: str, *, iteration: int) -> None:
        """Initialize the error with the iteration at which the stall occurred.

        :param message: Human-readable description of the stall.
        :type message: str
        :param iteration: The iteration index at which the stall was detected.
        :type iteration: int
        """
        super().__init__(message)
        self.iteration = iteration


class StructuredOutputError(PromptOptimizerError):
    """Raised when a structured (schema-validated) LLM response fails to parse.

    :param message: Human-readable description of the parse failure.
    :type message: str
    :param raw_text: The raw model output that failed to parse, if available.
    :type raw_text: str | None
    """

    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        """Initialize the error with the raw text that failed to parse.

        :param message: Human-readable description of the parse failure.
        :type message: str
        :param raw_text: The raw model output that failed to parse, if available.
        :type raw_text: str | None
        """
        super().__init__(message)
        self.raw_text = raw_text


class ConfigurationError(PromptOptimizerError):
    """Raised when ``Settings`` or a caller-supplied config value is invalid."""
