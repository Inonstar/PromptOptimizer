# Automatic Prompt Evaluator & Optimizer

A fully automatic, prompt-agnostic system that takes any prompt and iteratively improves it without per-prompt customization. You hand it a prompt; it returns a better one.

## How It Works

The system runs a closed loop. Starting from an initial prompt, it generates a dataset of (input, output) examples, evaluates the outputs to find failures, and then expands the dataset with new failure-prone examples before rewriting the prompt to handle them. This repeats until the prompt converges.

The novel angle is the automatic adversarial dataset expansion. Rather than testing against a fixed set of inputs, the system reasons about what kinds of inputs cause the *current* prompt to fail and actively generates more of them each iteration. The dataset therefore grows more complete and more adversarial over time, not just larger.

## Tech Stack

- **Evaluation** — LangFuse with custom LLM-as-judge evaluators that score outputs and return failure reasons.
- **Failure-case discovery** — PromptPex (Microsoft), which extracts the prompt's own spec to generate targeted edge-case tests.
- **Dataset expansion** — LLM-driven perturbation that takes failing examples and generates variations and paraphrases to multiply coverage.
- **Prompt optimization** — DSPy (MIPROv2), using LangFuse scores as the metric function to iteratively rewrite the prompt.
- **LLM backbone** — Claude (Anthropic).

A key integration detail: DSPy's metric function is bridged to LangFuse scores, so the optimizer sees exactly the same evaluation signal as the observability layer.

## What Makes It Different

Tools like DSPy, TextGrad, and EvidentlyAI's optimizers already exist, but none combine spec-based failure discovery (PromptPex) with LLM-driven perturbation for dataset expansion. The result works for any prompt without domain-specific configuration, and its dataset becomes progressively more adversarial and complete with each iteration.