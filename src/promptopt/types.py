"""Core data model shared by every ``promptopt`` module.

Every model here is a frozen ``pydantic.BaseModel``: instances are immutable
(mutating methods return a new instance via ``model_copy``), they serialize
to JSONL for free, and they validate at every module boundary that passes
one around. See ``lib_structure.md`` -> "Core data model (``types.py``)" for
the field-level rationale behind each type.

:seealso: ``lib_structure.md``
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    """Generate a short random identifier for a new model instance.

    :returns: A UUID4 hex string.
    :rtype: str
    """
    return uuid.uuid4().hex


def _utc_now() -> float:
    """Return the current time as a Unix timestamp.

    Factored out so tests can monkeypatch a single function instead of
    freezing every call site's notion of "now".

    :returns: Seconds since the Unix epoch.
    :rtype: float
    """
    return time.time()


class FrozenModel(BaseModel):
    """Base class for every immutable value type in the core data model.

    :cvar model_config: Forbids extra fields, freezes instances after
        construction, and validates on assignment (which is a no-op once
        frozen, but keeps the config self-documenting).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


RuleKind = Literal["constraint", "format", "behavior", "scope"]
RuleSource = Literal["extracted", "researched", "user"]
ExampleProvenance = Literal["seed", "spec_test", "perturbation", "paraphrase", "edge_case"]
DatasetSplit = Literal["train", "validation", "held_out"]
OutputType = Literal["exact", "numeric", "categorical", "structured", "free_text", "open_ended"]
StopReason = Literal[
    "target_reached",
    "plateau",
    "dry_expansion",
    "max_iterations",
    "max_runtime",
    "budget",
    "regression_guard",
]


class Citation(FrozenModel):
    """A single source backing a claim in a :class:`DomainBrief`.

    :ivar url: The source URL.
    :type url: str
    :ivar title: The source's title, if known.
    :type title: str | None
    :ivar snippet: The excerpt that supports the claim, if captured.
    :type snippet: str | None
    """

    url: str
    title: str | None = None
    snippet: str | None = None


class Prompt(FrozenModel):
    """A versioned prompt template under optimization.

    :ivar id: Stable identifier shared across all versions of one prompt lineage.
    :type id: str
    :ivar template: The prompt text, with ``{variable}``-style placeholders.
    :type template: str
    :ivar input_variables: Names of the placeholders ``template`` expects.
    :type input_variables: list[str]
    :ivar version: Monotonically increasing version number within the lineage.
    :type version: int
    :ivar parent_version: The version this one was derived from, or ``None`` for
        the original prompt.
    :type parent_version: int | None
    :ivar metadata: Free-form provenance, e.g. which optimizer produced this version.
    :type metadata: dict[str, str]
    """

    id: str = Field(default_factory=_new_id)
    template: str
    input_variables: tuple[str, ...] = Field(default_factory=tuple)
    version: int = 0
    parent_version: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def render(self, **values: str) -> str:
        """Render the template by substituting ``input_variables``.

        :param values: One keyword per entry in ``input_variables``.
        :type values: str
        :returns: The rendered prompt text.
        :rtype: str
        :raises KeyError: If a declared input variable has no supplied value.
        """
        missing = [name for name in self.input_variables if name not in values]
        if missing:
            raise KeyError(f"missing values for input variables: {missing}")
        return self.template.format(**values)

    def content_hash(self) -> str:
        """Compute a stable hash of the template text.

        Used to key caches (e.g. judgement caching by ``(prompt_hash, example_id)``)
        without relying on ``id``/``version``, which are lineage bookkeeping, not
        content identity.

        :returns: A hex-encoded SHA-256 digest of ``template``.
        :rtype: str
        """
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()


class Rule(FrozenModel):
    """A single behavioral rule extracted (or supplied) for a prompt.

    :ivar id: Stable identifier for this rule.
    :type id: str
    :ivar text: The rule stated as a plain-language assertion.
    :type text: str
    :ivar kind: What aspect of behavior the rule governs.
    :type kind: RuleKind
    :ivar source: Where the rule came from.
    :type source: RuleSource
    """

    id: str = Field(default_factory=_new_id)
    text: str
    kind: RuleKind
    source: RuleSource


class InvertedRule(FrozenModel):
    """The "what would violate this rule" transform of a :class:`Rule`.

    Inverted rules are the seed for targeted, adversarial test generation:
    each one describes an input shape that should make the prompt fail the
    rule it was inverted from.

    :ivar id: Stable identifier for this inverted rule.
    :type id: str
    :ivar rule_id: The :class:`Rule` this was inverted from.
    :type rule_id: str
    :ivar violation_description: Plain-language description of a violating input/output.
    :type violation_description: str
    """

    id: str = Field(default_factory=_new_id)
    rule_id: str
    violation_description: str


class DomainBrief(FrozenModel):
    """Cited domain research produced by the optional research pass.

    :ivar domain: The identified problem domain, e.g. "ICD-10 medical coding".
    :type domain: str
    :ivar key_concepts: Core concepts a correct response must respect.
    :type key_concepts: list[str]
    :ivar terminology: Domain-specific terms and their meanings.
    :type terminology: dict[str, str]
    :ivar realistic_input_patterns: Shapes of input real users in this domain send.
    :type realistic_input_patterns: list[str]
    :ivar known_edge_cases: Edge cases specific to the domain, not the prompt wording.
    :type known_edge_cases: list[str]
    :ivar common_pitfalls: Mistakes generic (non-domain-aware) responses tend to make.
    :type common_pitfalls: list[str]
    :ivar sources: Citations backing the above.
    :type sources: list[Citation]
    """

    domain: str
    key_concepts: tuple[str, ...] = Field(default_factory=tuple)
    terminology: dict[str, str] = Field(default_factory=dict)
    realistic_input_patterns: tuple[str, ...] = Field(default_factory=tuple)
    known_edge_cases: tuple[str, ...] = Field(default_factory=tuple)
    common_pitfalls: tuple[str, ...] = Field(default_factory=tuple)
    sources: tuple[Citation, ...] = Field(default_factory=tuple)


class Spec(FrozenModel):
    """The testable specification extracted from a prompt.

    :ivar prompt_version: The :class:`Prompt` version this spec was extracted from.
    :type prompt_version: int
    :ivar intent: Plain-language statement of what the prompt is trying to do.
    :type intent: str
    :ivar rules: The extracted (and/or researched/user-supplied) rules.
    :type rules: list[Rule]
    :ivar inverted_rules: The violation-seeking inversion of ``rules``.
    :type inverted_rules: list[InvertedRule]
    :ivar domain_brief: Domain research backing ``rules``, or ``None`` if research was off.
    :type domain_brief: DomainBrief | None
    """

    prompt_version: int
    intent: str
    rules: tuple[Rule, ...] = Field(default_factory=tuple)
    inverted_rules: tuple[InvertedRule, ...] = Field(default_factory=tuple)
    domain_brief: DomainBrief | None = None


class Example(FrozenModel):
    """A single dataset input, with optional expected output and provenance.

    :ivar id: Stable identifier for this example.
    :type id: str
    :ivar inputs: The prompt's input variables and their values for this example.
    :type inputs: dict[str, str]
    :ivar expected: The expected/gold output, if one exists.
    :type expected: str | None
    :ivar provenance: How this example was produced.
    :type provenance: ExampleProvenance
    :ivar parent_id: The example this one was derived from (perturbation/paraphrase), if any.
    :type parent_id: str | None
    :ivar generation: The loop iteration in which this example was created.
    :type generation: int
    :ivar target_rule_ids: Rule(s) this example was generated to exercise.
    :type target_rule_ids: list[str]
    """

    id: str = Field(default_factory=_new_id)
    inputs: dict[str, str]
    expected: str | None = None
    provenance: ExampleProvenance
    parent_id: str | None = None
    generation: int = 0
    target_rule_ids: tuple[str, ...] = Field(default_factory=tuple)


class Dataset(FrozenModel):
    """A collection of examples with coverage accounting and split assignment.

    :ivar examples: The examples in this dataset.
    :type examples: list[Example]
    :ivar coverage: Map of rule id -> number of examples targeting that rule.
    :type coverage: dict[str, int]
    :ivar splits: Map of example id -> assigned split.
    :type splits: dict[str, DatasetSplit]
    """

    examples: tuple[Example, ...] = Field(default_factory=tuple)
    coverage: dict[str, int] = Field(default_factory=dict)
    splits: dict[str, DatasetSplit] = Field(default_factory=dict)

    def split(self, name: DatasetSplit) -> tuple[Example, ...]:
        """Return the examples assigned to a given split.

        :param name: Which split to return.
        :type name: DatasetSplit
        :returns: The examples in that split, in dataset order.
        :rtype: tuple[Example, ...]
        """
        ids = {example_id for example_id, split_name in self.splits.items() if split_name == name}
        return tuple(example for example in self.examples if example.id in ids)


class TokenUsage(FrozenModel):
    """Token accounting for a single LLM request.

    :ivar input_tokens: Prompt tokens billed at the standard input rate.
    :type input_tokens: int
    :ivar output_tokens: Completion tokens billed at the output rate. This is
        the inclusive, authoritative total used for billing; ``thinking_tokens``
        is a read-only breakdown of a subset of it, not an addend.
    :type output_tokens: int
    :ivar thinking_tokens: Of ``output_tokens``, how many were spent on internal
        reasoning (extended/adaptive thinking) rather than the visible reply.
    :type thinking_tokens: int
    :ivar cache_creation_input_tokens: Input tokens written to the prompt cache.
    :type cache_creation_input_tokens: int
    :ivar cache_read_input_tokens: Input tokens served from the prompt cache.
    :type cache_read_input_tokens: int
    """

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens billed for the request, cache activity included.

        ``thinking_tokens`` is excluded since it is already counted within
        ``output_tokens``.

        :returns: Sum of every billed token count on this usage record.
        :rtype: int
        """
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


class CostUsage(FrozenModel):
    """Token usage and accrued dollar cost attributed to one model.

    This is the unit :mod:`promptopt.llm.budget` accumulates per stage/run:
    one entry per distinct model, so a run mixing tiers (e.g. Haiku for
    example generation, Opus for spec extraction) reports cost per tier
    rather than one opaque total.

    :ivar model: The model identifier this usage was billed against.
    :type model: str
    :ivar usage: Token accounting for requests to this model.
    :type usage: TokenUsage
    :ivar cost_usd: Dollar cost computed from ``usage`` and the model's price table.
    :type cost_usd: float
    """

    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0


class Generation(FrozenModel):
    """One prompt-under-test execution against one example.

    :ivar example_id: The :class:`Example` this generation ran against.
    :type example_id: str
    :ivar prompt_version: The :class:`Prompt` version that produced ``output``.
    :type prompt_version: int
    :ivar output: The raw model output text.
    :type output: str
    :ivar usage: Token accounting for the generation request.
    :type usage: TokenUsage
    :ivar latency_ms: Wall-clock latency of the request, in milliseconds.
    :type latency_ms: float
    :ivar trace_id: The observability span this generation was recorded under.
    :type trace_id: str | None
    """

    example_id: str
    prompt_version: int
    output: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    trace_id: str | None = None


class Judgement(FrozenModel):
    """The verdict a :class:`~promptopt.evaluate.judges.base.JudgeStrategy` returns.

    :ivar example_id: The :class:`Example` this judgement is for.
    :type example_id: str
    :ivar score: Score in ``[0, 1]``; ``1`` is a perfect pass.
    :type score: float
    :ivar passed: Whether the output cleared the strategy's pass threshold.
    :type passed: bool
    :ivar failure_reason: Plain-language explanation, empty when ``passed`` is ``True``.
    :type failure_reason: str
    :ivar violated_rule_ids: Rule(s) the output violated, if any were identifiable.
    :type violated_rule_ids: list[str]
    :ivar strategy: Name of the :class:`JudgeStrategy` implementation that scored this.
    :type strategy: str
    :ivar judge_model: The model used to judge, or ``None`` for deterministic judges.
    :type judge_model: str | None
    :ivar deterministic: Whether the judge is repeatable with zero LLM calls.
    :type deterministic: bool
    """

    example_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failure_reason: str = ""
    violated_rule_ids: tuple[str, ...] = Field(default_factory=tuple)
    strategy: str
    judge_model: str | None = None
    deterministic: bool


class EvalReport(FrozenModel):
    """Aggregated results of one evaluation pass over a dataset (split).

    :ivar overall_pass_rate: Fraction of judgements with ``passed=True``.
    :type overall_pass_rate: float
    :ivar mean_score: Mean of every judgement's ``score``.
    :type mean_score: float
    :ivar per_rule_pass_rate: Map of rule id -> pass rate among judgements citing it.
    :type per_rule_pass_rate: dict[str, float]
    :ivar failure_clusters: Map of cluster label -> example ids in that cluster.
    :type failure_clusters: dict[str, list[str]]
    :ivar judgements: Every judgement produced during this pass.
    :type judgements: list[Judgement]
    """

    overall_pass_rate: float = Field(ge=0.0, le=1.0)
    mean_score: float = Field(ge=0.0, le=1.0)
    per_rule_pass_rate: dict[str, float] = Field(default_factory=dict)
    failure_clusters: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    judgements: tuple[Judgement, ...] = Field(default_factory=tuple)


class IterationRecord(FrozenModel):
    """A checkpointable summary of one closed-loop iteration.

    :ivar iteration: Zero-based iteration index.
    :type iteration: int
    :ivar prompt_version: The :class:`Prompt` version evaluated this iteration.
    :type prompt_version: int
    :ivar dataset_size: Number of examples in the dataset at this iteration.
    :type dataset_size: int
    :ivar dataset_composition: Map of provenance -> example count at this iteration.
    :type dataset_composition: dict[str, int]
    :ivar eval_report: The evaluation results for this iteration.
    :type eval_report: EvalReport
    :ivar optimizer_outcome: Free-form summary of what the optimizer changed.
    :type optimizer_outcome: str
    :ivar token_spend: Total tokens spent during this iteration.
    :type token_spend: int
    :ivar started_at: Unix timestamp when the iteration began.
    :type started_at: float
    """

    iteration: int
    prompt_version: int
    dataset_size: int
    dataset_composition: dict[str, int] = Field(default_factory=dict)
    eval_report: EvalReport
    optimizer_outcome: str = ""
    token_spend: int = 0
    started_at: float = Field(default_factory=_utc_now)


class RunState(FrozenModel):
    """The full checkpoint unit for a closed-loop run.

    :ivar run_id: Stable identifier for this run.
    :type run_id: str
    :ivar history: Every completed iteration, in order.
    :type history: list[IterationRecord]
    :ivar current_prompt: The prompt at the current point in the run.
    :type current_prompt: Prompt
    :ivar current_dataset: The dataset at the current point in the run.
    :type current_dataset: Dataset
    :ivar rng_seed: The seed used for every stochastic step, for reproducibility.
    :type rng_seed: int
    :ivar stop_reason: Why the run ended, or ``None`` while still running.
    :type stop_reason: StopReason | None
    :ivar best_prompt: The best prompt seen by held-out score, which may not be
        ``current_prompt`` if later iterations regressed.
    :type best_prompt: Prompt | None
    """

    run_id: str = Field(default_factory=_new_id)
    history: tuple[IterationRecord, ...] = Field(default_factory=tuple)
    current_prompt: Prompt
    current_dataset: Dataset
    rng_seed: int
    stop_reason: StopReason | None = None
    best_prompt: Prompt | None = None
