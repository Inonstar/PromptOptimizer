"""The ``LLMClient`` protocol: the one seam every LLM-backed module calls through.

Nothing outside :mod:`promptopt.llm.anthropic` imports the ``anthropic`` SDK
directly (see ``lib_structure.md`` -> "What backs each part"). Every other
module -- spec extraction, judging, expansion, the DSPy adapter -- depends
only on this protocol, so a fake implementation can stand in for tests and a
second real implementation could be added without touching a caller.

:seealso: ``lib_structure.md`` -> "Package layout" (``llm/base.py``)
"""

from __future__ import annotations

from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from promptopt.llm.routing import ModelSpec
from promptopt.types import Citation, TokenUsage

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn in a conversation sent to :meth:`LLMClient.complete`.

    System instructions are not a turn -- they are passed via the separate
    ``system`` parameter on every :class:`LLMClient` method, matching the
    Anthropic Messages API's own shape.

    :ivar role: Who sent this turn.
    :type role: Role
    :ivar content: The turn's text content.
    :type content: str
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str


class CompletionResult(BaseModel):
    """The result of an :meth:`LLMClient.complete` call.

    :ivar text: The model's final reply text (excludes any thinking block).
    :type text: str
    :ivar model: The model that actually served the request.
    :type model: str
    :ivar stop_reason: Why generation stopped, e.g. ``"end_turn"``, ``"max_tokens"``.
    :type stop_reason: str
    :ivar usage: Token accounting for the request.
    :type usage: TokenUsage
    :ivar thinking: The model's extended-thinking transcript, if requested and returned.
    :type thinking: str | None
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str
    stop_reason: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    thinking: str | None = None


class StructuredResult(BaseModel, Generic[ResponseModelT]):
    """The result of an :meth:`LLMClient.complete_structured` call.

    :ivar value: The parsed, schema-validated response.
    :type value: ResponseModelT
    :ivar model: The model that actually served the request.
    :type model: str
    :ivar stop_reason: Why generation stopped.
    :type stop_reason: str
    :ivar usage: Token accounting for the request.
    :type usage: TokenUsage
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    value: ResponseModelT
    model: str
    stop_reason: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class SearchResult(BaseModel):
    """The result of an :meth:`LLMClient.search` call.

    Backs the optional domain-research pass: a synthesized answer plus the
    citations it was built from, so a :class:`~promptopt.types.DomainBrief`
    can be inspected for authority rather than taken on faith.

    :ivar text: The synthesized answer.
    :type text: str
    :ivar citations: Sources the model cited while researching.
    :type citations: list[Citation]
    :ivar model: The model that actually served the request.
    :type model: str
    :ivar usage: Token accounting for the request, including server-side search cost.
    :type usage: TokenUsage
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    citations: tuple[Citation, ...] = Field(default_factory=tuple)
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


@runtime_checkable
class LLMClient(Protocol):
    """The Claude access seam every stage of the loop calls through.

    Implementations: :class:`promptopt.llm.anthropic.AnthropicClient` (real)
    and a fake used across the unit test suite (see ``tests/support``). Both
    are exercised by the shared contract test suite so they can't silently
    diverge.
    """

    def complete(
        self,
        *,
        messages: list[Message],
        model_spec: ModelSpec,
        system: str | None = None,
        max_tokens: int = 4096,
        thinking: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> CompletionResult:
        """Run a free-text completion.

        :param messages: The conversation so far, oldest turn first.
        :type messages: list[Message]
        :param model_spec: Which model/effort tier to use.
        :type model_spec: ModelSpec
        :param system: System instructions, if any.
        :type system: str | None
        :param max_tokens: Maximum output tokens to generate.
        :type max_tokens: int
        :param thinking: Whether to enable adaptive extended thinking.
        :type thinking: bool
        :param stop_sequences: Custom stop sequences, if any.
        :type stop_sequences: list[str] | None
        :returns: The completion result.
        :rtype: CompletionResult
        :raises RefusalError: If the model declines to answer.
        :raises BudgetExceeded: If completing this request would exceed a configured ceiling.
        """
        ...

    def complete_structured(
        self,
        *,
        messages: list[Message],
        response_model: type[ResponseModelT],
        model_spec: ModelSpec,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> StructuredResult[ResponseModelT]:
        """Run a completion constrained to a pydantic schema.

        :param messages: The conversation so far, oldest turn first.
        :type messages: list[Message]
        :param response_model: The pydantic model the response must conform to.
        :type response_model: type[ResponseModelT]
        :param model_spec: Which model/effort tier to use.
        :type model_spec: ModelSpec
        :param system: System instructions, if any.
        :type system: str | None
        :param max_tokens: Maximum output tokens to generate.
        :type max_tokens: int
        :returns: The parsed, schema-validated result.
        :rtype: StructuredResult[ResponseModelT]
        :raises RefusalError: If the model declines to answer.
        :raises StructuredOutputError: If the response cannot be parsed into ``response_model``.
        :raises BudgetExceeded: If completing this request would exceed a configured ceiling.
        """
        ...

    def search(
        self,
        *,
        query_context: str,
        model_spec: ModelSpec,
        max_searches: int = 5,
        system: str | None = None,
    ) -> SearchResult:
        """Run a completion backed by the server-side ``web_search`` tool.

        :param query_context: What to research, in natural language.
        :type query_context: str
        :param model_spec: Which model/effort tier to use.
        :type model_spec: ModelSpec
        :param max_searches: Ceiling on ``web_search`` tool invocations for this call.
        :type max_searches: int
        :param system: System instructions, if any.
        :type system: str | None
        :returns: The synthesized, cited result.
        :rtype: SearchResult
        :raises RefusalError: If the model declines to answer.
        :raises BudgetExceeded: If completing this request would exceed a configured ceiling.
        """
        ...

    def count_tokens(
        self,
        *,
        messages: list[Message],
        model_spec: ModelSpec,
        system: str | None = None,
    ) -> int:
        """Count the input tokens a request would use, without generating.

        :param messages: The conversation to count.
        :type messages: list[Message]
        :param model_spec: Which model to count tokens for (tokenization is model-specific).
        :type model_spec: ModelSpec
        :param system: System instructions, if any.
        :type system: str | None
        :returns: The input token count.
        :rtype: int
        """
        ...
