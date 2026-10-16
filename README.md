# Tiny GPT Lab

**Build a language model from scratch. Teach it a small language. Test what it actually learns.**

Tiny GPT Lab is a follow-along Python project for understanding how GPT-style language models work. Starting with token probabilities, we will build a **decoder-only Transformer from randomly initialized weights**, train it to interpret a narrow set of natural-language requests, and measure how well it handles sentences it has never seen.

The goal is not to build a useful general-purpose chatbot. It is to make the mechanics of language modeling **visible, runnable, and falsifiable**.

> **Project status: planned / pre-implementation.** The architecture, tutorial, and evaluation protocol are specified, but this README does not claim that a working model, checkpoints, benchmark results, or the CLI already exist. Setup and execution instructions below are targets for the initial implementation and must be verified as milestones land.

## What will you learn?

By the end of the core tutorial, you should be able to explain and experiment with:

- How text becomes token IDs and how a model predicts the next token.
- Why embeddings, context, and learned parameters matter.
- How causal self-attention, multi-head attention, and rotary positional embeddings (RoPE) work.
- How a small Transformer is trained using backpropagation and teacher forcing.
- How the same next-token objective can generate **structured interpretations**, not just prose.
- Why training accuracy, plausible examples, and well-formed JSON do **not** by themselves establish language understanding.

You will also learn how to build fair baselines and tests that can *disprove* your expectations.

## The target demonstration

The first language domain is deliberately small: fictional project-resource allocation. The trained model should translate everyday requests into a strict JSON contract.

**Illustrative target behavior — not measured model output:**

```text
Input:  Assign Ingrid to Coral for 12 hours in week 42.
Output: {"intent":"allocate","person":"Ingrid","project":"Coral","hours":12,"week":42}

Input:  Who is allocated to Coral in week 42?
Output: {"intent":"query_allocations","project":"Coral","week":42}

Input:  Assign Ingrid to Coral in week 42.
Output: {"intent":"needs_clarification","for_intent":"allocate","missing":["hours"]}

Input:  What's the weather tomorrow?
Output: {"intent":"unsupported"}
```

The model learns to **propose** these representations. A strict JSON parser and intent-schema validator then decide whether a proposal is structurally acceptable. Neither a well-formed proposal nor a validation pass guarantees that the original sentence was correctly understood.

Initially, interpretation is **read-only**: it does not update any real planning system.

## How the model works

```text
Natural-language request
        |
        v
Byte tokenizer -> token embeddings
        |
        v
4 x causal Transformer blocks
    - multi-head self-attention
    - RoPE positions
    - feed-forward network
    - normalization + residual connections
        |
        v
Tied output head -> next-token probabilities
        |
        v
Autoregressive JSON generation
        |
        v
Strict parsing + schema validation
        |
        v
Untrusted structured interpretation
```

The reference `tiny` configuration targets **fewer than one million trainable parameters**: 4 layers, embedding width 128, 4 attention heads, feed-forward width 512, and a vocabulary of 256 byte values plus special tokens. The actual parameter count and training performance will be reported from the implementation, not assumed. A smaller `cpu-smoke` configuration is reserved for correctness checks in CI.

We use PyTorch for tensor computation, automatic differentiation, and optimization, while implementing the educational architecture ourselves. **No pretrained weights, hosted model APIs, imported Transformer model classes, or downloaded tokenizers** are used in the core experiment.

## Follow-along lessons

Lessons are designed as readable, executable **Jupytext percent-format `.py` files**, with optional generated notebooks. Each lesson introduces a concept, demonstrates an observable result, and includes an exercise.

| Lesson | Topic | What you build or observe |
|---|---|---|
| 01 | Text and tokens | Byte tokenizer; encode/decode and probability counts |
| 02 | Learning to predict | Trainable bigram model, loss, gradients, and sampling |
| 03 | Why context matters | Token embeddings and a fixed-window MLP |
| 04 | Attention | Causal masking, Q/K/V, and attention visualizations |
| 05 | Tiny GPT | Multi-head Transformer, RoPE, training, and generation |
| 06 | Language interpretation | Synthetic instruction-to-JSON data and target-only loss |
| 07 | Does it generalize? | Held-out tests, rule-based baseline, and failure analysis |
| 08 | Interactive demo | Type a request and inspect generation and validation |

Engineering and teaching progress is tracked through milestones **M0–M7** in [`plan.md`](plan.md). The knowledge-graph demonstration is a separate, optional **M8**.

## Getting started

**Current state:** The repository is in its specification phase. The commands below describe the intended setup **once M0 has implemented and verified the Python package**; they are not advertised as working today.

The requirements will be pinned in `pyproject.toml`. You will need a compatible Python version, a local environment, and (for training) sufficient RAM and processing time. A GPU is **not assumed**; the project will benchmark the real `tiny` model on CPU before making workshop-time claims.

```bash
# After M0 and the packaging files are committed:
git clone https://github.com/quaat/nanogpt.git
cd nanogpt
python -m venv .venv
source .venv/bin/activate  # Linux/macOS; activate differently on Windows
python -m pip install -e .
python -m nanogpt_lab --help
```

The intended CLI commands are `tokenize`, `generate-data`, `train`, `evaluate`, and `interpret`. Exact flags, config paths, test commands, and reproducible training invocations will be documented **after their implementations exist**. Do not treat an illustrative command or sample output as evidence of a completed experiment.

## What counts as success?

The most important result is not a convincing-looking JSON example. It is an honest comparison between **memorization, generalization, and deterministic parsing**.

The experimental protocol includes:

- **Learning curves:** train independent models from scratch with 1,000, 5,000, and 20,000 unique examples, using the declared optimization budget.
- **In-distribution tests:** previously unseen instances from familiar template families.
- **Distribution shifts:** new names, number forms, slot orders, held-out phrasing families, and entirely withheld categories of unsupported requests.
- **Independent human-curated tests:** sentences written by the human maintainer *after* the generator and baseline are frozen, without model-generated suggestions or model-output feedback.
- **Fair rules comparison:** a transparent parser built from training templates and public schemas, frozen before the independent tests are written.

The preregistered hypotheses are deliberately falsifiable:

| Question | Planned success criterion |
|---|---|
| **H1: Sample efficiency** | At least **80%** exact semantic match on held-out IID examples, with at most 20k training examples and 5k updates |
| **H2: Unseen wording** | At least **50%** exact semantic match on independently human-authored, supported paraphrases |
| **H3: Value beyond rules** | The neural model beats the frozen training-template-only rules parser by at least **5 absolute percentage points** on the curated supported suite |

These are **targets, not achieved results**. If the human-curated suite is not available, H2 and H3 will be marked **not evaluated** rather than silently replaced with AI-authored examples. Test sets, criteria, and artifact identities are locked before the final evaluation. Failures will be published alongside successes.

See [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md) for the normative benchmark design, leakage controls, metrics, and release checks.

## Optional extension: natural language to a knowledge graph

Once the core model and evaluation are complete, a separate exercise will connect the interpreter to a small **RDF knowledge graph**:

```text
Natural language
  -> tiny GPT proposes JSON
  -> strict schema validation
  -> deterministic entity resolution (fictional catalog)
  -> JSON-to-RDF mapping
  -> SHACL validation
  -> local graph inspection / read-only query
```

The neural model performs language interpretation; **deterministic code** handles identity resolution, RDF creation, and constraints. A valid SHACL report shows that a graph conforms to its declared constraints, **not** that the model understood the original sentence. The extension will use only fictional data and will not mutate external systems.

This optional phase is specified in [`docs/semantic-extension.md`](docs/semantic-extension.md). RDFLib and pySHACL will be installed only through a future optional `semantic` dependency extra, not as mandatory core dependencies.

## Repository guide

The planned structure is intentionally small; directories will appear as milestones are implemented.

```text
nanogpt/
├── README.md                    # start here
├── plan.md                      # architecture, milestones, hypotheses
├── AGENTS.md                    # implementation and review guardrails
├── docs/
│   ├── evaluation-protocol.md   # locked benchmark design
│   └── semantic-extension.md    # optional RDF/SHACL phase
├── lessons/                     # 01–08 executable .py lessons (planned)
├── src/nanogpt_lab/             # educational implementation (planned)
├── tests/                       # correctness and smoke tests (planned)
├── data/                        # disclosed templates and fixtures (planned)
├── configs/                     # tiny and cpu-smoke profiles (planned)
└── pyproject.toml               # pinned dependencies (planned)
```

For contributors and coding agents, [`AGENTS.md`](AGENTS.md) defines the minimal implementer → adversarial reviewer → human approval process. The learner material lives here and in `lessons/`, not in agent governance documents.

## Project principles

1. **Build it, don't import it.** Write the model components locally; use external references for concepts, not copied implementations.
2. **Make claims reproducible.** Publish settings, seeds, dataset provenance, model artifacts, actual runtimes, and denominators.
3. **Prevent accidental cheating.** Do not tune on locked test sets or pass AI-written examples off as independent human assessment.
4. **Keep boundaries explicit.** Neural interpretation, schema validity, domain correctness, and graph conformance are separate questions.
5. **Show what fails.** A tiny model failing on novel syntax or names is a valuable lesson, not a reason to hide the test.

## Reading and inspiration

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer foundations.
- [TinyStories](https://arxiv.org/abs/2305.07759) — what small language models can learn from constrained data.
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — an accessible conceptual walkthrough.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) — practical experimental controls.
- [W3C SHACL](https://www.w3.org/TR/shacl/) — validation for the optional semantic extension.

**License:** To be selected by the repository maintainer before a public code release.
