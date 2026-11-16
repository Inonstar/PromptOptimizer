"""Stage -> :class:`ModelSpec` resolution: the model-tiering seam.

Not every stage of the loop needs a frontier model, and per-token cost
differs roughly 5x between tiers (see ``lib_structure.md`` -> "Model
tiering"). This module defines the :class:`Stage` enum, the ``Effort``
literal the Anthropic API accepts, the immutable :class:`ModelSpec` a stage
resolves to, and :func:`resolve_model`, which looks a stage up in
``Settings.model_tiers`` with the defaults below as a fallback.

:seealso: ``lib_structure.md`` -> "Model tiering"
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from promptopt.errors import ConfigurationError

if TYPE_CHECKING:
    from promptopt.config import Settings

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Stage(str, enum.Enum):
    """A named point in the loop that resolves to its own model/effort tier.

    :cvar RESEARCH: Optional domain research pass (:mod:`promptopt.spec.research`).
    :cvar SPEC_EXTRACTION: Prompt -> :class:`~promptopt.types.Spec` extraction.
    :cvar INVERTED_RULES: Rule -> :class:`~promptopt.types.InvertedRule` transform.
    :cvar EXAMPLE_GENERATION: Rule-targeted input synthesis for seeding.
    :cvar PERTURBATION_PARAPHRASE: Rewriting an existing failing example.
    :cvar EDGE_CASE_SYNTHESIS: Creative edge-case generation from inverted rules.
    :cvar JUDGING: LLM-backed judging (``semantic``/``rubric`` strategies only).
    :cvar OPTIMIZATION: MIPROv2's instruction-proposal step.
    :cvar TARGET: The prompt-under-test's own deployment model. Not tiered by
        us -- resolved straight from ``Settings.target_model`` -- but modeled
        as a stage so every caller asks :func:`resolve_model` uniformly.
    """

    RESEARCH = "research"
    SPEC_EXTRACTION = "spec_extraction"
    INVERTED_RULES = "inverted_rules"
    EXAMPLE_GENERATION = "example_generation"
    PERTURBATION_PARAPHRASE = "perturbation_paraphrase"
    EDGE_CASE_SYNTHESIS = "edge_case_synthesis"
    JUDGING = "judging"
    OPTIMIZATION = "optimization"
    TARGET = "target"


class ModelSpec(BaseModel):
    """An immutable ``(model, effort)`` pair a :class:`Stage` resolves to.

    :ivar model: The Anthropic model identifier, e.g. ``"claude-sonnet-5"``.
    :type model: str
    :ivar effort: The ``output_config.effort`` level to request, or ``None``
        to omit it and let the API use its default.
    :type effort: Effort | None
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    effort: Effort | None = None


# Defaults from lib_structure.md -> "Model tiering". A starting hypothesis,
# not a guess kept forever: `bench.py` (M6) sweeps alternatives and this
# table is meant to be updated from its output.
DEFAULT_MODEL_TIERS: dict[Stage, ModelSpec] = {
    Stage.RESEARCH: ModelSpec(model="claude-sonnet-5", effort="medium"),
    Stage.SPEC_EXTRACTION: ModelSpec(model="claude-opus-5", effort="high"),
    Stage.INVERTED_RULES: ModelSpec(model="claude-sonnet-5", effort="medium"),
    Stage.EXAMPLE_GENERATION: ModelSpec(model="claude-haiku-4-5"),
    Stage.PERTURBATION_PARAPHRASE: ModelSpec(model="claude-haiku-4-5"),
    Stage.EDGE_CASE_SYNTHESIS: ModelSpec(model="claude-sonnet-5", effort="medium"),
    Stage.JUDGING: ModelSpec(model="claude-sonnet-5", effort="medium"),
    Stage.OPTIMIZATION: ModelSpec(model="claude-opus-5", effort="xhigh"),
}


def resolve_model(stage: Stage, settings: "Settings") -> ModelSpec:
    """Resolve a :class:`Stage` to the :class:`ModelSpec` it should run with.

    Resolution order: an explicit override in ``settings.model_tiers``, then
    the built-in default for every stage except :attr:`Stage.TARGET`, which
    always comes from ``settings.target_model`` since it is "not ours to
    pick" (see ``lib_structure.md``).

    :param stage: The stage to resolve.
    :type stage: Stage
    :param settings: The active settings, consulted for overrides.
    :type settings: Settings
    :returns: The resolved model/effort pair.
    :rtype: ModelSpec
    :raises ConfigurationError: If ``stage`` is :attr:`Stage.TARGET` and
        ``settings.target_model`` was never configured.
    """
    if stage is Stage.TARGET:
        if settings.target_model is None:
            raise ConfigurationError(
                "settings.target_model must be set before resolving Stage.TARGET -- "
                "the target model is the user's deployment choice, not a tiering default."
            )
        return ModelSpec(model=settings.target_model)

    override = settings.model_tiers.get(stage)
    if override is not None:
        return override
    return DEFAULT_MODEL_TIERS[stage]
