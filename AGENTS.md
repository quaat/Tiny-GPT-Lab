# Implementation Agent Contract — Tiny GPT Lab

> Applies to every Codex/Claude implementation or review task in `quaat/nanogpt`. Read **this file and the current relevant milestone**, not every research document on every task. The human maintainer is the final approver.

## Mission and boundaries

- Build a **real, randomly initialized causal Transformer** for an educational research demonstration. PyTorch tensors/autograd/optimizers are allowed; no pretrained weights, proprietary-model inference at runtime, imported Transformer model implementations, or copy-pasted nanoGPT source.
- The language model learns **next-token prediction** and narrow text→JSON conversion. Rule-based postprocessors may **reject** invalid outputs but must not secretly repair or generate missing model predictions. The rules baseline is a separate, clearly labeled evaluation artifact.
- Every trained output, attention diagram, score, and timing must come from a **real, identified run**; illustrative examples are labeled as specifications, not results. Report negative results and known limitations.
- Model-generated JSON is **untrusted**. Validate independently; never execute allocations, access real company systems, or write to external databases. Later RDF/SHACL changes require a separate approval.
- Do not read or use confidential user, employee, organizational, or project data. Synthetic example names and projects only.

## Where the contracts live

| Need | Read |
|---|---|
| Thesis, reference model, training defaults, milestones | [`plan.md`](plan.md) |
| Splits, baseline information diet, locked suites, metrics, freeze procedure | [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) |
| RDF/SHACL extension once approved | [`docs/semantic-extension.md`](docs/semantic-extension.md) |
| Current APIs and expected behavior | Actual `src/`, `tests/`, and affected configs |

Code and tests that exist in the repository must be inspected before editing. A plan is not evidence that an API has already been implemented.

## Minimal two-agent workflow

1. **Implementer:** Select one milestone slice, inspect dependencies, state its interface/acceptance checks, implement it with direct tests and learner-facing explanation, then report exact tests and outcomes.
2. **Adversarial reviewer:** Independently check the diff against acceptance criteria, especially future-token causality, target-only loss masking, data provenance/leakage, checkpoint safety, score denominators, and claims. Add or request failing tests where useful. Do not assert that unrun tests passed.
3. **Human maintainer:** Approves merges and any changed task schema, benchmark definitions/thresholds, training-template information diet, or neural/symbolic safety boundary. Only the human authors/approves the independent curated evaluation suite.

Limited parallelism: after tokenizer/target-schema interfaces stabilize, one implementer can work on **model** modules and another on **synthetic data generation**, with different files/worktrees and an explicit handoff. Do not run large parallel agent swarms for a small codebase. One independent review still precedes merge.

A work request should contain: **milestone; owned files; expected public API; one-line learner outcome; acceptance tests; maximum compute budget**. Do not create multi-page planning artifacts or ADRs for ordinary internal changes.

## Research integrity and data provenance

- Agents may create template and synthetic datasets **only if recorded as LLM-assisted**. Never call them independently human-authored, manually curated, or independent evidence.
- **Curated test cases:** The human maintainer creates them *after* the generator and rules baseline are frozen and without consulting model outputs. Agents must not write, label, augment, paraphrase, or silently replace that suite. If not supplied, mark H2/H3 **not evaluated**.
- The rules baseline may inspect only training templates, training examples and documented task schemas before its code is frozen. It may not inspect held-out family sentences, negative categories reserved for tests, or future curated text. Record allowed inputs and baseline commit SHA. The LLM agent's own linguistic prior cannot be eliminated; disclose this limitation.
- Never inspect locked test cases to choose hyperparameters, extend training, change instructions, or modify parsing heuristics. Make changes from training/validation diagnostics; requests to revise test contracts go to the human, with a new benchmark version.
- The release verifier must recompute SHA-256 hashes of canonical test manifests and verify the *approved* pass-criteria hash. No agent may quietly update either lock to make a run pass.

## Mandatory tests and decision gates

Prior to a full training run, prove:

- Byte-tokenizer round-trip and special-token IDs; UTF-8 and explicit invalid-byte behavior.
- Causal invariance: changing future input tokens does not alter earlier logits in eval mode.
- Attention mask finite and correctly directed; RoPE dimensions and offsets tested.
- Target-only shift/masking: first JSON token and EOS counted, prompt and pad excluded.
- All `<pad>` tokens occur **after** `<eos>` (strict right-padding) in valid encoded sequences.
- Embedding/LM-head weights **share storage**, not merely equal initialization.
- Greedy generation is equivalent to teacher-forced next-token argmax when driven by the same preceding prefix, with model in eval mode and no truncation.
- Model overfits one tiny training batch; save/load yields equivalent logits with validated JSON metadata and `torch.load(..., weights_only=True)`.
- Strict parser rejects duplicate keys, malformed JSON, extra text, out-of-range values, and unexpected fields; no silent success fallback.
- Split builder has no cross-split normalized duplicates or prohibited template/slot leakage.

Run bounded CPU lint/type/unit tests in CI; aim for **≤5 minutes** total. No unbounded training, new infrastructure, network-connected services, or large checkpoint artifacts without human agreement. Benchmark actual CPU performance rather than relying on FLOP estimates. The full `tiny` model is a legitimate CPU target if measured runtime permits; `cpu-smoke` stays a CI-only correctness check.

## Release/PR reporting (concise)

Every reviewable change includes: changed files and purpose; relevant tests and **actual observed results**; remaining failures; training data/checkpoint provenance where applicable; any new dependency or changed contract. Avoid elaborate progress logs and redundant evidence packages. A single reproducible evaluation report suffices for the experimental release.

**Stop and ask for human approval** only for genuine boundary changes: intent schema, curated benchmark, threshold/hash lock, model/rules information diet, validation semantics, production side effects, or scope expansion to RDF. Otherwise, complete the small assigned task and submit for adversarial review.
