"""Token and cost accounting, with per-run and per-stage ceilings.

:class:`BudgetTracker` is the single place that turns raw
:class:`~promptopt.types.TokenUsage` into dollars and enforces
``Settings.budgets``. Every :class:`~promptopt.llm.base.LLMClient`
implementation should route its usage through one shared tracker instance
per run so ``max_tokens``/``max_cost_usd`` are enforced across every stage,
not per-call.

Per-token prices are deliberately *not* hardcoded here: model pricing
changes over time and inventing numbers in source would silently misprice
every run. Callers supply a :class:`ModelPricing` table (e.g. sourced from
``Settings`` or the ``bench.py`` output in M6); a model with no entry still
gets accurate token accounting, just with ``cost_usd == 0.0``.

:seealso: ``lib_structure.md`` -> "Package layout" (``llm/budget.py``), "Cost guard"
"""

from __future__ import annotations

import threading
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from promptopt.config import BudgetSettings
from promptopt.errors import BudgetExceeded
from promptopt.llm.routing import Stage
from promptopt.types import CostUsage, TokenUsage


class ModelPricing(BaseModel):
    """Dollar cost per million tokens, by category, for one model.

    :ivar input_per_million_usd: Cost per million standard input tokens.
    :type input_per_million_usd: float
    :ivar output_per_million_usd: Cost per million output tokens (thinking included).
    :type output_per_million_usd: float
    :ivar cache_write_per_million_usd: Cost per million prompt-cache-write tokens.
    :type cache_write_per_million_usd: float
    :ivar cache_read_per_million_usd: Cost per million prompt-cache-read tokens.
    :type cache_read_per_million_usd: float
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_million_usd: float
    output_per_million_usd: float
    cache_write_per_million_usd: float = 0.0
    cache_read_per_million_usd: float = 0.0


def _cost_of(usage: TokenUsage, pricing: ModelPricing | None) -> float:
    """Compute the dollar cost of one usage record.

    :param usage: The token usage to price.
    :type usage: TokenUsage
    :param pricing: The model's price table, or ``None`` if unpriced.
    :type pricing: ModelPricing | None
    :returns: The dollar cost, or ``0.0`` if ``pricing`` is ``None``.
    :rtype: float
    """
    if pricing is None:
        return 0.0
    return (
        usage.input_tokens * pricing.input_per_million_usd
        + usage.output_tokens * pricing.output_per_million_usd
        + usage.cache_creation_input_tokens * pricing.cache_write_per_million_usd
        + usage.cache_read_input_tokens * pricing.cache_read_per_million_usd
    ) / 1_000_000


class BudgetTracker:
    """Accumulates token/cost usage for a run and enforces its ceilings.

    Thread-safe: :meth:`record` is called from every worker thread a
    concurrent evaluation or expansion batch spins up (see
    :mod:`promptopt.llm.progress`), so accounting and ceiling checks happen
    under a single lock.

    :param settings: The budget ceilings to enforce.
    :type settings: BudgetSettings
    :param pricing: Per-model price tables. Models absent from this mapping
        are still tracked in tokens but cost ``$0.00``.
    :type pricing: dict[str, ModelPricing] | None
    """

    def __init__(self, settings: BudgetSettings, pricing: dict[str, ModelPricing] | None = None) -> None:
        """Initialize an empty tracker for one run.

        :param settings: The budget ceilings to enforce.
        :type settings: BudgetSettings
        :param pricing: Per-model price tables, keyed by model identifier.
        :type pricing: dict[str, ModelPricing] | None
        """
        self._settings = settings
        self._pricing = pricing or {}
        self._lock = threading.Lock()
        self._run_usage: dict[str, CostUsage] = {}
        self._stage_usage: dict[Stage, dict[str, CostUsage]] = defaultdict(dict)

    def record(self, stage: Stage, model: str, usage: TokenUsage) -> CostUsage:
        """Record usage from one completed request and enforce ceilings.

        :param stage: Which :class:`~promptopt.llm.routing.Stage` issued the request.
        :type stage: Stage
        :param model: The model that served the request.
        :type model: str
        :param usage: Token accounting for the request.
        :type usage: TokenUsage
        :returns: The priced usage just recorded.
        :rtype: CostUsage
        :raises BudgetExceeded: If recording this usage pushes the run or the
            stage past a configured ceiling. The usage is recorded before the
            check, so the tracker's totals reflect the request that tipped it over.
        """
        cost = _cost_of(usage, self._pricing.get(model))
        with self._lock:
            self._run_usage[model] = _merge(self._run_usage.get(model), model, usage, cost)
            self._stage_usage[stage][model] = _merge(self._stage_usage[stage].get(model), model, usage, cost)
            run_tokens = self.total_tokens
            run_cost = self.total_cost_usd
            stage_tokens = self._stage_total_tokens(stage)
            stage_cost = self._stage_total_cost_usd(stage)

        self._check(scope="run", tokens=run_tokens, cost=run_cost, max_tokens=self._settings.max_tokens,
                     max_cost=self._settings.max_cost_usd)
        self._check(
            scope=stage.value,
            tokens=stage_tokens,
            cost=stage_cost,
            max_tokens=self._settings.max_tokens_per_stage.get(stage),
            max_cost=self._settings.max_cost_usd_per_stage.get(stage),
        )
        return CostUsage(model=model, usage=usage, cost_usd=cost)

    @staticmethod
    def _check(
        *, scope: str, tokens: int, cost: float, max_tokens: int | None, max_cost: float | None
    ) -> None:
        """Raise :class:`BudgetExceeded` if either ceiling for ``scope`` is broken.

        :param scope: Human-readable scope name for the error, e.g. ``"run"``.
        :type scope: str
        :param tokens: Tokens spent so far in this scope.
        :type tokens: int
        :param cost: Dollars spent so far in this scope.
        :type cost: float
        :param max_tokens: Token ceiling for this scope, or ``None`` for no ceiling.
        :type max_tokens: int | None
        :param max_cost: Dollar ceiling for this scope, or ``None`` for no ceiling.
        :type max_cost: float | None
        :raises BudgetExceeded: If a configured ceiling was broken.
        """
        if max_tokens is not None and tokens > max_tokens:
            raise BudgetExceeded(
                f"{scope} token budget exceeded: {tokens} > {max_tokens}",
                scope=scope,
                limit=max_tokens,
                spent=tokens,
            )
        if max_cost is not None and cost > max_cost:
            raise BudgetExceeded(
                f"{scope} cost budget exceeded: ${cost:.4f} > ${max_cost:.4f}",
                scope=scope,
                limit=max_cost,
                spent=cost,
            )

    @property
    def total_tokens(self) -> int:
        """Total tokens spent so far across every model in this run.

        :returns: Sum of ``total_tokens`` across every tracked model.
        :rtype: int
        """
        return sum(entry.usage.total_tokens for entry in self._run_usage.values())

    @property
    def total_cost_usd(self) -> float:
        """Total dollar cost spent so far across every model in this run.

        :returns: Sum of ``cost_usd`` across every tracked model.
        :rtype: float
        """
        return sum(entry.cost_usd for entry in self._run_usage.values())

    def usage_by_model(self) -> list[CostUsage]:
        """Return accumulated usage, one entry per model billed in this run.

        :returns: One :class:`~promptopt.types.CostUsage` per distinct model.
        :rtype: list[CostUsage]
        """
        return list(self._run_usage.values())

    def _stage_total_tokens(self, stage: Stage) -> int:
        """Total tokens spent so far within one stage.

        :param stage: The stage to total.
        :type stage: Stage
        :returns: Sum of ``total_tokens`` across every model billed in ``stage``.
        :rtype: int
        """
        return sum(entry.usage.total_tokens for entry in self._stage_usage[stage].values())

    def _stage_total_cost_usd(self, stage: Stage) -> float:
        """Total dollar cost spent so far within one stage.

        :param stage: The stage to total.
        :type stage: Stage
        :returns: Sum of ``cost_usd`` across every model billed in ``stage``.
        :rtype: float
        """
        return sum(entry.cost_usd for entry in self._stage_usage[stage].values())


def _merge(existing: CostUsage | None, model: str, usage: TokenUsage, cost: float) -> CostUsage:
    """Fold a new usage record into an existing per-model accumulator.

    :param existing: The accumulator so far, or ``None`` if this is the first record.
    :type existing: CostUsage | None
    :param model: The model identifier being accumulated.
    :type model: str
    :param usage: The new usage to fold in.
    :type usage: TokenUsage
    :param cost: The dollar cost of ``usage``.
    :type cost: float
    :returns: A new accumulator with ``usage`` and ``cost`` added.
    :rtype: CostUsage
    """
    if existing is None:
        return CostUsage(model=model, usage=usage, cost_usd=cost)
    merged_usage = TokenUsage(
        input_tokens=existing.usage.input_tokens + usage.input_tokens,
        output_tokens=existing.usage.output_tokens + usage.output_tokens,
        thinking_tokens=existing.usage.thinking_tokens + usage.thinking_tokens,
        cache_creation_input_tokens=(
            existing.usage.cache_creation_input_tokens + usage.cache_creation_input_tokens
        ),
        cache_read_input_tokens=existing.usage.cache_read_input_tokens + usage.cache_read_input_tokens,
    )
    return CostUsage(model=model, usage=merged_usage, cost_usd=existing.cost_usd + cost)
