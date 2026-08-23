# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PromptOptimizer (`promptopt`) is a fully automatic, prompt-agnostic system that takes any prompt and iteratively improves it without per-prompt customization. It runs a closed loop: generate a dataset of (input, output) examples, evaluate outputs to find failures, expand the dataset adversarially against the *current* prompt's weaknesses (not just larger, but more targeted), then rewrite the prompt with DSPy (MIPROv2) using LangFuse-compatible scores as the metric, and repeat until convergence.

See [OVERVIEW.md](OVERVIEW.md) for the product-level pitch and [lib_structure.md](lib_structure.md) for the full architecture, data model, build order, and design rationale — read that file before implementing any module, it is the source of truth this CLAUDE.md summarizes.

**Status:** the package (`src/promptopt/`) has not been scaffolded yet — only the design docs (`OVERVIEW.md`, `lib_structure.md`) exist. The folder layout below is the *planned* target layout from `lib_structure.md`; build it in the order given there (M0 → M6).

## Planned folder structure

Src layout, `promptopt` as the import name (`src/promptopt/`):

- **`llm/`** — LLM access layer. Defines the `LLMClient` protocol, the Anthropic SDK implementation (the only module allowed to import `anthropic`), model-tier routing, the DSPy adapter, budget/cost accounting, response caching, and progress reporting.
- **`prompts/`** — The `Prompt` value type (template, variables, render, hash) and an append-only version registry with diffing between versions.
- **`spec/`** — Turns a prompt into a testable specification: optional domain research, spec extraction, rule inversion, and the native (pure-Python) reimplementation of the PromptPex method behind a `TestGenerator` protocol.
- **`dataset/`** — Dataset lifecycle: the `DatasetStore` protocol, initial seeding from the spec, adversarial expansion (perturbation/paraphrase/edge-case synthesis), near-duplicate dedupe, and train/validation/held-out splitting.
- **`evaluate/`** — Runs the prompt over the dataset and judges the outputs; aggregates results into per-rule pass rates and failure clusters that drive expansion.
  - **`evaluate/judges/`** — One `JudgeStrategy` implementation per output type (exact, numeric, categorical, structured, semantic, rubric, composite), all behind a shared protocol so callers never need to know which one ran.
- **`observability/`** — Tracing and score storage behind `Tracer` / `ScoreStore` protocols, with a local JSONL+SQLite default and an optional LangFuse-backed implementation.
- **`optimize/`** — Prompt optimization: the `Optimizer` protocol, the DSPy bridge (`Prompt` ⇄ `dspy.Signature`/`Module`, metric adapter), the MIPROv2 driver, and a no-op/single-shot baseline for A/B comparison.
- **`loop/`** — The closed-loop controller that sequences bootstrap → evaluate → discover failures → expand → optimize → converge, plus stopping-criteria logic and checkpoint/resume state.

Top-level files inside `src/promptopt/` (not folders, so not itemized further): `types.py` (shared dataclasses/pydantic models), `config.py` (`Settings`), `errors.py` (typed exceptions), `report.py`, `bench.py`, `cli.py`.

## Code style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Prefer clean, clear, readable code while still being efficient — favor obvious naming and simple control flow over cleverness; don't sacrifice performance where it matters (hot paths, batch/concurrent LLM calls) to get there.
- Max line length: **100 characters**.
- Use type hints on all function/method signatures and module-level variables where the type isn't obvious.
- Docstrings use **reST style** (`:param:`, `:type:`, `:returns:`, `:rtype:`, `:raises:`), on every public and private module, class, and function.
- Code must be **pylint-clean**; any disable must be scoped as narrowly as possible and justified with a comment.
- **No code duplication** — if logic is repeated (even across modules), extract it into a shared function/module rather than copy-pasting. Prefer the existing protocol seams in `lib_structure.md` (e.g. `LLMClient`, `JudgeStrategy`, `Optimizer`) over ad hoc reimplementations.

## Testing

- **All code must be tested**, including helper functions and private (`_`-prefixed) functions — not just the public API.
- Every test covers both **typical and edge cases** (empty input, boundary values, error paths), not just the happy path.
- Test classes inherit from `unittest.TestCase`, or `unittest.IsolatedAsyncioTestCase` for async code. `pytest` is the runner; `unittest` is the authoring style — write test classes, not bare pytest functions.
- Per `lib_structure.md`'s verification plan: unit tests run against the fake `LLMClient` (no network by default); contract test suites are shared mixin `TestCase` classes run against every implementation of a protocol (`DatasetStore`, `Tracer`, `ScoreStore`, `TestGenerator`, `JudgeStrategy`, `Optimizer`); integration tests that make real Claude calls are marked `live` and opt-in only.

## Common commands

Run these from the repository root, with the project's virtualenv active.

### Run the full test suite

```bash
pytest
```

### Run a single test file, class, or method

```bash
pytest tests/path/to/test_module.py
pytest tests/path/to/test_module.py::TestClassName
pytest tests/path/to/test_module.py::TestClassName::test_method_name
```

### Skip opt-in live (real-API) tests

```bash
pytest -m "not live"
```

### Run pylint across the entire project

```bash
pylint src
```

(Include `tests` too once test modules exist and a pylint config is checked in, e.g. `pylint src tests`.)

## After finishing each feature

Before considering any feature, fix, or refactor done, you **must** run both of the following and confirm they pass clean:

1. `pytest` — the full test suite.
2. `pylint src` — the full lint check.

Do not report a feature as complete if either command fails or reports new warnings/errors.
