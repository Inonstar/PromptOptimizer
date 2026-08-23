# PromptOptimizer — Library Architecture

## Context

[OVERVIEW.md](OVERVIEW.md) describes a closed-loop system that takes any prompt and returns a better one: generate a dataset, evaluate outputs to find failures, expand the dataset adversarially against the *current* prompt's weaknesses, rewrite the prompt, repeat until convergence. The novel part is the automatic adversarial dataset expansion — the dataset gets more complete and more adversarial each iteration rather than just larger.

This document defines the package architecture — module boundaries, the data model that flows between them, which library backs each part, and the build order — so implementation can start against a fixed set of seams.

Integration decisions settled up front:

| Decision | Choice | Why it matters |
|---|---|---|
| PromptPex | Native Python implementation of the method, behind a `TestGenerator` protocol. No Node, no subprocess adapter. | PromptPex is a Microsoft **GenAIScript/TypeScript** tool shipped as a dockerized CLI — no Python API, nothing on PyPI. Porting the method keeps the library pure Python; the protocol seam preserves the option to add other generators later. |
| Spec/test generation deps | None — build it | Survey below: nothing in Python does prompt → spec → rules → inverted rules → tests. DeepEval's `Synthesizer` covers only input synthesis and generic difficulty evolutions, defaults to OpenAI, and has no notion of targeting the *current prompt's* observed failures — which is the project's whole differentiator. |
| LLM access | Own `LLMClient` protocol + Anthropic SDK implementation, wrapped as a `dspy.LM` adapter | Keeps direct access to adaptive thinking, prompt caching, structured outputs, and `stop_reason: "refusal"` handling, while letting DSPy share one client, one budget tracker, and one trace context. |
| LangFuse | Optional backend behind `Tracer` / `ScoreStore` / `DatasetStore` protocols; local JSONL+SQLite is the default | The library runs with zero external services out of the box; LangFuse plugs in via config for teams that want it. Tests don't need a mock server. |

### Prior-art survey (why the spec layer is built, not borrowed)

| Candidate | Verdict |
|---|---|
| [PromptPex](https://github.com/microsoft/promptpex) (Microsoft) | Node/GenAIScript + Docker CLI. No Python bindings, no PyPI package. Exports to OpenAI Evals format. Method is portable; the tool is not. |
| [`promptfoo` on PyPI](https://pypi.org/project/promptfoo/) | Not a Python SDK — a pip wrapper that shells to `npx promptfoo@latest`, requires Node 20+. Same Node dependency as PromptPex with no added benefit. |
| [DeepEval `Synthesizer`](https://deepeval.com/docs/golden-synthesizer) | Real overlap on *input synthesis* (`generate_goldens_from_scratch` + `StylingConfig`) and *generic* difficulty evolutions (`EvolutionConfig`). No spec extraction, no failure-targeted expansion; needs a `DeepEvalBaseLLM` subclass to run on Claude. Revisit only if native expansion underperforms at M3. |
| RAGAS testset generation | Requires a document corpus. RAG-specific — wrong fit for a prompt-agnostic system. |
| LLMart, PromptAttack | Jailbreak / robustness toolkits. Different problem: attacking the model, not testing behavioral conformance to a prompt's own spec. |

## Package layout

Src layout, `promptopt` as the import name.

```
src/promptopt/
  types.py          core dataclasses shared by every module
  config.py         Settings (pydantic-settings): models, effort, budgets, backends
  errors.py         typed exceptions (BudgetExceeded, RefusalError, ConvergenceStalled)

  llm/
    base.py         LLMClient protocol: complete / complete_structured / search / count_tokens
    anthropic.py    AnthropicClient — the only place the Anthropic SDK is imported
    routing.py      Stage -> ModelSpec(model, effort) resolution; the model-tiering seam
    dspy_lm.py      ClaudeLM(dspy.BaseLM) adapter so DSPy reuses the same client
    budget.py       token + cost accounting, per-run and per-stage ceilings
    cache.py        on-disk response cache keyed by (model, effort, prompt hash)
    progress.py     tqdm progress bars driven by client-level request events

  prompts/
    template.py     Prompt: template text, input variables, render(), hash
    registry.py     append-only version history + diff between versions

  spec/
    models.py       Spec, Rule, InvertedRule, DomainBrief
    research.py     optional domain research pass -> DomainBrief (web search)
    extract.py      prompt (+ DomainBrief) -> Spec (Claude, structured output)
    rules.py        rule -> inverted rule (the "what would violate this" transform)
    base.py         TestGenerator protocol
    native.py       the PromptPex method, reimplemented in Python on Claude

  dataset/
    store.py        DatasetStore protocol + JSONL impl
    seed.py         initial input synthesis from the Spec
    expand.py       adversarial expansion: perturbation, paraphrase, edge-case synthesis
    dedupe.py       near-duplicate removal + coverage accounting
    sampling.py     train / validation / held-out split with provenance stratification

  evaluate/
    judges/
      base.py        JudgeStrategy protocol + Judgement construction
      exact.py       exact / normalized string match
      numeric.py     numbers with absolute or relative tolerance, ranges, units
      categorical.py label-set membership, confusion tracking
      structured.py  JSON/schema conformance, then field-wise delegation
      semantic.py    free text: equivalence grade vs expected output
      rubric.py      no gold answer — grade against Spec rules directly
      composite.py   combine strategies (e.g. format AND semantic)
    routing.py      pick the judge from the Spec's output type + config override
    runner.py       concurrent generate-then-judge over a dataset
    metrics.py      aggregation, per-rule pass rates, failure clustering

  observability/
    base.py         Tracer / ScoreStore protocols
    local.py        JSONL + SQLite (default)
    langfuse.py     LangFuse implementation (optional extra)

  optimize/
    base.py         Optimizer protocol: optimize(prompt, dataset, metric) -> Prompt
    dspy_bridge.py  Prompt <-> dspy.Signature/Module, metric adapter
    mipro.py        MIPROv2 driver
    baseline.py     no-op + single-shot rewrite, for A/B against the real optimizer

  loop/
    controller.py   the closed loop
    convergence.py  stopping criteria
    state.py        checkpoint / resume

  report.py         markdown + HTML run report
  bench.py          model x effort sweep per stage — which tier is good enough where
  cli.py            argparse CLI (subcommands + settings flags)
```

## Core data model (`types.py`)

Everything below is a frozen pydantic model so it serializes to JSONL for free and validates at module boundaries.

- **`Prompt`** — `id`, `template`, `input_variables`, `version`, `parent_version`, `metadata`. Immutable; the optimizer returns a new one.
- **`Rule`** — `id`, `text`, `kind` (`constraint` | `format` | `behavior` | `scope`), `source` (`extracted` | `researched` | `user`).
- **`DomainBrief`** — `domain`, `key_concepts`, `terminology`, `realistic_input_patterns`, `known_edge_cases`, `common_pitfalls`, `sources: list[Citation]`. Optional; `None` when research is off.
- **`Spec`** — `prompt_version`, `intent`, `rules: list[Rule]`, `inverted_rules: list[InvertedRule]`, `domain_brief: DomainBrief | None`.
- **`Example`** — `id`, `inputs: dict[str, str]`, `expected: str | None`, `provenance` (`seed` | `spec_test` | `perturbation` | `paraphrase` | `edge_case`), `parent_id`, `generation: int`, `target_rule_ids`.
- **`Dataset`** — examples + coverage map (`rule_id -> example count`) + split assignment.
- **`Generation`** — `example_id`, `prompt_version`, `output`, `usage`, `latency_ms`, `trace_id`.
- **`OutputType`** — `exact` | `numeric` | `categorical` | `structured` | `free_text` | `open_ended`. Inferred from the `Spec`'s format rules, overridable in config.
- **`Judgement`** — `example_id`, `score: float` (0–1), `passed: bool`, `failure_reason: str`, `violated_rule_ids: list[str]`, `strategy: str`, `judge_model: str | None` (`None` for deterministic judges), `deterministic: bool`.
- **`EvalReport`** — overall pass rate, mean score, per-rule pass rate, failure clusters, `list[Judgement]`.
- **`IterationRecord`** — iteration index, prompt version, dataset size + composition, `EvalReport`, optimizer outcome, token spend.
- **`RunState`** — full history of `IterationRecord`s, current prompt, current dataset, RNG seed. This is the checkpoint unit.

## Data flow — one iteration of the loop

`loop/controller.py` owns this sequence; every step is a protocol call, so each is independently swappable and testable.

1. **Bootstrap (iteration 0 only).** Optionally `spec.research` builds a `DomainBrief` first (see below). `spec.extract` then derives a `Spec` from the prompt, conditioned on that brief. `spec.native` turns rules + inverted rules into targeted test inputs. `dataset.seed` rounds out coverage for any rule with too few examples.
2. **Evaluate.** `evaluate.runner` generates outputs for every example with the current prompt, then the judge selected by `evaluate.routing` scores each one. Concurrency and retries live in the runner; judges are pure per-example calls. Scores go to the `ScoreStore`; spans go to the `Tracer`.
3. **Discover failures.** `evaluate.metrics` clusters failures by `violated_rule_ids` and by judge-written failure reason, producing a ranked list of weak rules — this is the targeting signal for expansion, and the piece that makes expansion adversarial to the *current* prompt rather than generic.
4. **Expand.** For each weak cluster, `dataset.expand` produces (a) perturbations and paraphrases of the failing examples and (b) fresh edge cases from the inverted rules for that cluster. `dataset.dedupe` drops near-duplicates and updates the coverage map before merging.
5. **Optimize.** `optimize.mipro` runs MIPROv2 over the training split. The DSPy metric is a thin adapter around the *same* judge used in step 2, so the optimizer and the observability layer see an identical signal — the key integration detail from OVERVIEW.md. Judgements are cached by `(prompt_hash, example_id)` so a metric re-evaluation is free.
6. **Converge.** `loop.convergence` checks stopping criteria against the held-out split (never the split the optimizer saw). Then checkpoint `RunState` and loop.

Stopping criteria — all configurable, any one triggers:

| Criterion | Setting | Notes |
|---|---|---|
| Target reached | `target_pass_rate` | Held-out pass rate ≥ target |
| Plateau | `min_improvement`, `patience` | Improvement over the last *N* iterations below epsilon |
| Dry expansion | — | Expansion produced no novel failures (every new example passes) |
| Iteration cap | `max_iterations` | Hard ceiling on loop count |
| Wall-clock cap | `max_runtime` | Total elapsed run time; checked between iterations *and* inside long stages so a single slow evaluation can't blow past it |
| Budget | `max_tokens`, `max_cost_usd` | Enforced by `llm/budget.py` |
| Regression guard | `max_regressions` | Held-out score drops for *N* consecutive iterations — stop and return the best prompt seen, not the last |

Held-out score is what decides — a prompt that only improves on the training split is a regression. Every stop records its reason in `RunState` so the report says *why* the run ended.

### Domain research (optional, iteration 0)

Toggled by `--research` / `Settings.research.enabled`. Before spec extraction, the library asks Claude to identify the prompt's problem domain and research it using the server-side `web_search` tool, producing a `DomainBrief`: real terminology, realistic input shapes, known edge cases, common pitfalls, with citations.

The brief feeds two places: **spec extraction**, so rules can encode domain constraints a generic reading of the prompt would miss; and **seeding/expansion**, so generated examples look like real inputs from that field rather than plausible-sounding generic text. For a prompt about, say, ICD-10 coding or lease abstraction, this is the difference between examples that exercise the actual problem and examples that only exercise the prompt's surface wording.

Guardrails, because this step can quietly dominate cost and can hallucinate authority: it is off by default, capped by `research.max_searches`, produces a `DomainBrief` that is cached and reused across all iterations of a run, and every researched rule is tagged `source="researched"` so the report can show which constraints came from the web rather than from the prompt itself. `DomainBrief` is a plain artifact on disk — inspectable and hand-editable before a run proceeds.

## What backs each part

- **Anthropic SDK** (`llm/anthropic.py`) — the only module importing `anthropic`. Adaptive thinking (`thinking={"type": "adaptive"}`), `output_config={"effort": ...}` per stage, `client.messages.parse()` with pydantic models for every structured step (research, spec extraction, judging, expansion), streaming with `get_final_message()` whenever `max_tokens` is large, and the server-side `web_search` tool for the research pass. Prompt caching with `cache_control` on the stable prefix — the spec and judge rubric are byte-identical across a whole evaluation pass, so this is where most of the cost savings live. `stop_reason == "refusal"` is checked before reading `content` and surfaced as a typed error, not a crash.
- **tqdm** (`llm/progress.py`) — every request the client issues emits a start/finish event; the runner and expander attach `tqdm` bars to those events, so concurrent evaluation shows live per-stage progress (`judging 47/200`) instead of going silent for minutes. Auto-disabled when not a TTY so CI logs stay clean.
- **DSPy (MIPROv2)** (`optimize/`) — instruction and few-shot search. `dspy_bridge` converts a `Prompt` into a `dspy.Signature` + `Module` and back, and adapts the judge into a DSPy metric function.
- **pydantic / pydantic-settings** — the data model and config, and the schemas passed to `messages.parse()`.
- **argparse** (`cli.py`) — stdlib, no CLI framework dependency. Subcommands `optimize`, `eval`, `expand`, `spec`, `research`, `bench`, `report`, `resume`. Settings are generated into flags from the pydantic `Settings` model, so a config field and its `--flag` can't drift apart; precedence is CLI flag > env var > config file > default.
- **tenacity** — retry policy in `llm/anthropic.py` on top of the SDK's own retries, for judge/expansion calls that fail validation.
- **LangFuse** — optional extra (`pip install promptopt[langfuse]`), used only inside `observability/langfuse.py`. Together with the optional embeddings extra below, these are the only optional dependencies; everything else is pure Python with no Node or Docker runtime anywhere.

## Judging

"Did this output pass?" means different things for different prompts, and an LLM rubric grade is the wrong tool for most of them. Grading `42` against `42` with a language model is slow, expensive, and — worse — non-deterministic on a question that has an exact answer. Each output type gets a judge built for it, behind one `JudgeStrategy` protocol so the loop and the DSPy metric never care which one ran.

| Output type | Strategy | How it scores | LLM? |
|---|---|---|---|
| `exact` | `ExactMatchJudge` | Normalized string equality (case, whitespace, punctuation policy configurable) | No |
| `numeric` | `NumericJudge` | Parse the number out of the response, compare with absolute or relative tolerance; supports estimation bands and unit normalization | No |
| `categorical` | `CategoricalJudge` | Membership in the label set from the Spec's enum rule; feeds a confusion matrix so `metrics.py` can report which labels the prompt confuses | No |
| `structured` | `StructuredJudge` | Parse + JSON-Schema validate, then delegate field-by-field to the judge matching each field's type | Only for free-text fields |
| `free_text` | `SemanticJudge` | Equivalence grade against the expected output — does the response convey the same content? | Yes |
| `open_ended` | `RubricJudge` | No gold answer exists; grade directly against the `Spec` rules and return which ones were violated | Yes |
| mixed | `CompositeJudge` | Conjunction with per-component weights — e.g. format conformance must pass *and* semantic score ≥ threshold | Depends |

**Selection.** `evaluate/routing.py` infers `OutputType` from the Spec's `kind="format"` rules ("respond with exactly one of: yes, no" → `categorical`; "return valid JSON matching…" → `structured`; "answer with a single integer" → `numeric`), falling back to `free_text` when an example carries an `expected` value and `open_ended` when it doesn't. Config overrides the inference per run, and the chosen strategy is recorded on every `Judgement` so a report can show what actually graded each example.

**Why this matters beyond correctness.** Deterministic judges are free, instant, and perfectly repeatable. Judging is the highest-volume stage in the system (dataset × iterations × optimizer trials), so every prompt that routes to `exact`/`numeric`/`categorical` removes the dominant cost line *and* removes judge noise from the optimizer's metric — the single biggest source of a bad optimization decision. Model tiering below only applies to the two LLM-backed strategies.

**Semantic similarity has a dependency question.** Anthropic ships no embedding API, so `SemanticJudge` defaults to an LLM equivalence grade (no extra dependency, better at "same meaning, different words" than cosine distance). An optional `promptopt[embeddings]` extra adds local `sentence-transformers` for cheap bulk similarity, which `dataset/dedupe.py` also uses; without it, dedupe falls back to MinHash over token shingles. Neither path requires a network service.

## Model tiering

Not every stage needs a frontier model, and per-token cost differs ~5× between tiers. `llm/routing.py` resolves a `Stage` to a `ModelSpec(model, effort)`, every stage is overridable from config or CLI, and `bench.py` sweeps the alternatives so the defaults below are a starting hypothesis rather than a guess we keep forever.

| Stage | Default | Candidates to benchmark | Reasoning |
|---|---|---|---|
| Domain research | `claude-sonnet-5`, effort `medium` | opus-5 | Search + synthesis; breadth matters more than depth. |
| Spec extraction | `claude-opus-5`, effort `high` | sonnet-5 | Runs once per prompt but everything downstream is built on it — a missed rule is a permanently invisible failure mode. Highest quality-per-dollar case in the system. |
| Inverted rules | `claude-sonnet-5`, effort `medium` | opus-5, haiku-4-5 | Mechanical transform of an existing rule. |
| Example generation from rules | `claude-haiku-4-5` | sonnet-5 | The easy, high-volume stage. Generating an input that exercises a stated rule is close to templating. |
| Perturbation / paraphrase | `claude-haiku-4-5` | sonnet-5 | Same shape — rewrite an existing example. |
| Edge-case synthesis | `claude-sonnet-5`, effort `medium` | opus-5, haiku-4-5 | Genuinely creative; where weak models produce boring near-duplicates. |
| Judging (`semantic` / `rubric` only) | `claude-sonnet-5`, effort `medium` | opus-5, haiku-4-5 | Highest LLM call volume, so the dominant cost line — but it *is* the metric, and a noisy judge silently corrupts the optimizer. Benchmark hardest here. The deterministic judges use no model at all. |
| Optimization (MIPROv2 proposer) | `claude-opus-5`, effort `xhigh` | sonnet-5 | Instruction search is the core intelligence task. |
| **Target model** (running the prompt under test) | **user's choice** | — | Not ours to pick. The user is optimizing a prompt *for a specific deployment model*; optimizing against a different one produces a prompt that doesn't transfer. Configured separately from every stage above. |

`bench.py` runs the M6 fixture prompts across a stage × model × effort grid and reports quality, token spend, wall-clock, and cost per configuration. For judging specifically, "quality" is agreement with a fixed opus-5 reference judge plus a small hand-labeled set — cheaper judges are acceptable only where agreement stays high, since disagreement here propagates into every optimization decision.

## Public API surface

The library is the product; the CLI is a thin wrapper over it.

```python
from promptopt import Optimizer, Settings

opt = Optimizer(Settings())     # local backends, tiered models, sane budgets, research off
result = opt.run("Summarize the article in exactly three bullets.\n\n{article}")

result.prompt          # the improved prompt
result.report          # per-iteration history, pass rates, failure clusters
result.dataset         # the grown adversarial dataset, reusable as a regression suite
```

Each stage is also usable standalone (`research_domain`, `extract_spec`, `generate_tests`, `evaluate`, `expand`) so a user can adopt one piece without the loop.

## Build order

- **M0 — Foundations.** `pyproject.toml` (src layout, pylint + mypy + pytest), `types.py`, `config.py`, `errors.py`, `llm/` complete — including `routing.py` and `progress.py` — plus a fake client for tests. Verify: a round-trip structured call against Claude, visible tqdm progress on a concurrent batch, and a fully offline test suite using the fake.
- **M1 — Spec + seed data.** `prompts/`, `spec/` (models, research, extract, rules, base, native). Verify: `promptopt spec <prompt>` prints extracted rules and inverted rules; `promptopt research <prompt>` produces a cited `DomainBrief`; running extraction with and without the brief shows the brief adds domain rules a generic reading misses; generated tests actually target the rules they claim to.
- **M2 — Evaluation.** `evaluate/` (all judge strategies + routing) + `observability/local.py`. Verify: `promptopt eval` on a deliberately weak prompt reports a low pass rate with failure reasons that name real defects; routing picks the right strategy for each of the six output types from Spec format rules alone; deterministic judges return identical results across repeated runs and issue zero LLM calls.
- **M3 — Expansion.** `dataset/` complete. Verify: expansion seeded from a known failure cluster yields new examples that the current prompt also fails at a materially higher rate than random seeds — this is the core claim of the project and needs its own measurement, not just a smoke test.
- **M4 — Optimization.** `optimize/` with the DSPy bridge and MIPROv2. Verify: held-out pass rate improves over the M2 baseline on at least two unrelated prompts, and `baseline.py` (single-shot rewrite) is beaten.
- **M5 — The loop.** `loop/` with convergence and checkpointing. Verify: a full run terminates on *each* stopping criterion independently — including `max_iterations` and `max_runtime` — and records the correct reason; a run that regresses returns the best prompt seen, not the last; killing mid-run and resuming from the checkpoint reproduces the same trajectory.
- **M6 — Surface, benchmarks, LangFuse.** `cli.py`, `report.py`, `bench.py`, `observability/langfuse.py`. Verify: the argparse CLI exposes every setting and honors flag > env > file precedence; `promptopt bench` produces the stage × model × effort table and the defaults in the model-tiering section are updated from its output rather than left as assumptions; a LangFuse-backed run passes the same backend contract tests as the local implementation.

## Verification

Test style: **every test is a `unittest.TestCase` subclass**, run under `pytest`. Shared fixtures live in `setUp` / `setUpClass` and base classes rather than pytest fixture functions; the contract suites below are mixin `TestCase` classes that each backend's test module inherits from. `pytest` is the runner (markers, `-k`, parallelism, reporting); `unittest` is the authoring style.

- **Unit** — every module tested against the fake `LLMClient`; no network in the default `pytest` run.
- **Contract** — one shared test suite run against every implementation of each protocol (`DatasetStore`, `Tracer`, `ScoreStore`, `TestGenerator`, `JudgeStrategy`, `Optimizer`), so the local and LangFuse backends can't silently diverge and every judge honors the same `Judgement` invariants. This is also what keeps the door open for a second `TestGenerator` (e.g. a DeepEval-backed one) without reworking the loop.
- **Integration** (`-m live`, opt-in) — real Claude calls for research, spec extraction, judging, and one short optimization run, with a hard token budget.
- **Static** — `pylint` and `mypy` clean in CI, with the pylint config checked in so disables are explicit and reviewed rather than scattered inline.
- **End-to-end benchmark** — a small fixture set of 3–5 prompts with known defects (missing format constraint, ambiguous scope, no edge-case handling). The loop must find the defect in the failure clusters and fix it in the rewritten prompt. This doubles as the regression suite for the whole system.
- **Cost guard** — `llm/budget.py` asserted in tests: a run that exceeds its configured ceiling raises `BudgetExceeded` rather than continuing.
- **Model-tier regression** — the M6 fixture set is re-run whenever a stage's default model changes, so a cost-driven downgrade that quietly hurts final prompt quality is caught rather than shipped.
