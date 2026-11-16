"""``Settings`` -- the single configuration surface for the whole library.

Every knob mentioned across ``lib_structure.md`` (model tiers, effort,
budgets, backend selection, research, convergence) lives here as a
``pydantic-settings`` model, so a config field and its future CLI flag can't
drift apart (``cli.py`` generates flags from this model in M6) and so
precedence -- CLI flag > env var > config file > default -- has exactly one
place to be implemented.

:seealso: ``lib_structure.md`` -> "Package layout" (``config.py``), "argparse (``cli.py``)"
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from promptopt.llm.routing import ModelSpec, Stage


class ResearchSettings(BaseModel):
    """Settings for the optional domain-research pass.

    :ivar enabled: Whether to run :mod:`promptopt.spec.research` before spec
        extraction. Off by default -- this step can quietly dominate cost.
    :type enabled: bool
    :ivar max_searches: Ceiling on ``web_search`` tool invocations for the pass.
    :type max_searches: int
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_searches: int = 10


class BudgetSettings(BaseModel):
    """Ceilings enforced by :mod:`promptopt.llm.budget`.

    ``None`` means "no ceiling" for that dimension. Per-stage ceilings are
    optional refinements on top of the run-wide ones.

    :ivar max_tokens: Hard ceiling on total tokens (input + output) for a run.
    :type max_tokens: int | None
    :ivar max_cost_usd: Hard ceiling on total dollar cost for a run.
    :type max_cost_usd: float | None
    :ivar max_tokens_per_stage: Per-:class:`~promptopt.llm.routing.Stage` token ceilings.
    :type max_tokens_per_stage: dict[Stage, int]
    :ivar max_cost_usd_per_stage: Per-:class:`~promptopt.llm.routing.Stage` cost ceilings.
    :type max_cost_usd_per_stage: dict[Stage, float]
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_tokens_per_stage: dict[Stage, int] = Field(default_factory=dict)
    max_cost_usd_per_stage: dict[Stage, float] = Field(default_factory=dict)


class ConvergenceSettings(BaseModel):
    """Stopping criteria for the closed loop (``loop/convergence.py``, M5).

    Any one criterion triggers a stop; see ``lib_structure.md`` -> "Stopping
    criteria" for the semantics of each field.

    :ivar target_pass_rate: Stop once held-out pass rate reaches this.
    :type target_pass_rate: float | None
    :ivar min_improvement: Epsilon below which improvement counts as a plateau.
    :type min_improvement: float
    :ivar patience: Number of iterations improvement is averaged over for plateau detection.
    :type patience: int
    :ivar max_iterations: Hard ceiling on loop iterations.
    :type max_iterations: int
    :ivar max_runtime_seconds: Hard ceiling on total wall-clock run time.
    :type max_runtime_seconds: float | None
    :ivar max_regressions: Consecutive held-out score drops before stopping.
    :type max_regressions: int
    """

    model_config = ConfigDict(extra="forbid")

    target_pass_rate: float | None = None
    min_improvement: float = 0.01
    patience: int = 2
    max_iterations: int = 10
    max_runtime_seconds: float | None = None
    max_regressions: int = 2


class BackendSettings(BaseModel):
    """Which pluggable backend implementation each protocol seam should use.

    Local (JSONL/SQLite) backends are the zero-external-services default;
    LangFuse is opt-in via the ``promptopt[langfuse]`` extra.

    :ivar observability: Which :mod:`promptopt.observability` implementation to use.
    :type observability: str
    :ivar dataset_store: Which :mod:`promptopt.dataset.store` implementation to use.
    :type dataset_store: str
    :ivar data_dir: Directory local backends persist state under.
    :type data_dir: str
    """

    model_config = ConfigDict(extra="forbid")

    observability: str = Field(default="local", pattern="^(local|langfuse)$")
    dataset_store: str = Field(default="jsonl", pattern="^(jsonl)$")
    data_dir: str = ".promptopt"


class CacheSettings(BaseModel):
    """Settings for the on-disk LLM response cache (``llm/cache.py``).

    :ivar enabled: Whether to consult/populate the cache at all.
    :type enabled: bool
    :ivar directory: Directory the cache database is stored under.
    :type directory: str
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    directory: str = ".promptopt/cache"


class ProgressSettings(BaseModel):
    """Settings for tqdm progress reporting (``llm/progress.py``).

    :ivar enabled: ``None`` auto-detects (a bar is shown only on a TTY);
        ``True``/``False`` force it on/off, e.g. for CI logs.
    :type enabled: bool | None
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None


class Settings(BaseSettings):
    """Root configuration for a ``promptopt`` run.

    Values are read with precedence CLI flag > env var > config file >
    default once ``cli.py`` exists (M6); for now, precedence is env var >
    default, with every env var prefixed ``PROMPTOPT_`` and nested fields
    delimited with ``__`` (e.g. ``PROMPTOPT_BUDGETS__MAX_COST_USD=5.0``).

    :ivar anthropic_api_key: API key for the Anthropic SDK. Falls back to the
        SDK's own ``ANTHROPIC_API_KEY`` env var handling when unset here.
    :type anthropic_api_key: str | None
    :ivar target_model: The model the prompt under optimization will actually
        run on in production. Required before :attr:`~promptopt.llm.routing.Stage.TARGET`
        can be resolved -- "not ours to pick" (see ``lib_structure.md``).
    :type target_model: str | None
    :ivar model_tiers: Per-:class:`~promptopt.llm.routing.Stage` overrides of
        :data:`promptopt.llm.routing.DEFAULT_MODEL_TIERS`.
    :type model_tiers: dict[Stage, ModelSpec]
    :ivar research: Domain-research settings.
    :type research: ResearchSettings
    :ivar budgets: Token/cost ceilings.
    :type budgets: BudgetSettings
    :ivar convergence: Loop stopping criteria.
    :type convergence: ConvergenceSettings
    :ivar backends: Pluggable backend selection.
    :type backends: BackendSettings
    :ivar cache: LLM response cache settings.
    :type cache: CacheSettings
    :ivar progress: tqdm progress reporting settings.
    :type progress: ProgressSettings
    """

    model_config = SettingsConfigDict(
        env_prefix="PROMPTOPT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    anthropic_api_key: str | None = None
    target_model: str | None = None
    model_tiers: dict[Stage, ModelSpec] = Field(default_factory=dict)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    budgets: BudgetSettings = Field(default_factory=BudgetSettings)
    convergence: ConvergenceSettings = Field(default_factory=ConvergenceSettings)
    backends: BackendSettings = Field(default_factory=BackendSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    progress: ProgressSettings = Field(default_factory=ProgressSettings)

    @model_validator(mode="after")
    def _check_research_search_budget(self) -> "Settings":
        """Reject a nonsensical research configuration.

        :returns: ``self``, unchanged, once validated.
        :rtype: Settings
        :raises ValueError: If research is enabled with a non-positive search budget.
        """
        if self.research.enabled and self.research.max_searches <= 0:
            raise ValueError("research.max_searches must be positive when research.enabled is True")
        return self
