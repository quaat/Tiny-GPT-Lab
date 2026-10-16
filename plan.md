# Tiny GPT Lab — Development Plan

> **Status:** Proposed v2, incorporating adversarial review · **Date:** 2026-10-16  
> **Repository:** `quaat/nanogpt` (independent educational project; **not** Andrej Karpathy's `nanoGPT`)  
> **Purpose:** Governing *short* plan for the human maintainer and implementation agents.  
> **Companion contracts:** [`AGENTS.md`](AGENTS.md), [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md), [`docs/semantic-extension.md`](docs/semantic-extension.md).

## 1. Thesis and findings

Build a real decoder-only causal Transformer **from randomly initialized weights** with elementary PyTorch operations; teach every component in a follow-along sequence; then test whether a sub-million-parameter model can convert narrow natural-language requests into structured JSON. The final core deliverable is an **evaluated educational experiment**, not a general-purpose assistant. A later, **optional** extension converts independently validated JSON into an ontology-backed RDF knowledge graph with SHACL constraints.

Findings and design consequences:

- A 4-layer, width-128, 4-head Transformer with a small byte vocabulary is plausibly below one million parameters. Report the **actual** count from the built model, not an inherited estimate. Changing positional embeddings changes that count.
- CPU training of the full reference model may be practical during a workshop. **Benchmark real step time and total-run time on a reference CPU at M0/M4; never promise a workshop duration from FLOP estimates alone.** `cpu-smoke` is for CI correctness, not the only CPU-capable profile.
- Good scores on generator-shaped examples do **not** prove language understanding. The core scientific value is a preregistered comparison between in-distribution performance, genuine distribution shifts, and a fair rules baseline.
- An LLM coding agent may author source code and synthetic templates; **that data is LLM-assisted by provenance**. Only a human-maintainer-authored, withheld curated suite can support the project's independent paraphrase/generalization claim (H2).
- Neural decoding proposes an interpretation; parsing, schema checks, catalog resolution, and future SHACL validation are independent. No model output can directly modify systems.

**Never present intended output, projected metrics, runtime, or illustrative attention as measured model results.** A negative experimental result is acceptable and must be reported.

## 2. Scope and architecture

```text
Utterance → byte tokenizer → token embeddings → 4 × pre-norm Transformer block
          → final norm → tied LM head → causal next-token generation
          → strict JSON parse → exact intent schema → untrusted interpretation

Optional later: validated interpretation → deterministic identity resolver
          → RDF staging graph → SHACL validation → local/read-only query demo
```

Core implementation uses **PyTorch tensors, autograd, and optimizers**, but implements byte tokenization, Q/K/V attention, rotary positions, MLP blocks, masked supervised loss, autoregressive generation, and evaluation in this repository. Do **not** use pretrained weights, model APIs, Transformers library model classes, downloaded tokenizers, or code copied from other nanoGPT projects. Third-party papers and tutorials are *conceptual references only*. A KV cache, `torch.compile`, fused attention, mixed precision, distributed training, and external inference services are intentionally omitted from the teaching baseline.

### Reference profiles (measure, don't assume)

| Setting | `tiny` — real experiment | `cpu-smoke` — CI |
|---|---:|---:|
| Transformer blocks | 4 | 2 |
| Embedding width | 128 | 64 |
| Attention heads | 4 | 4 |
| FFN width | 512 | 256 |
| Token IDs | 256 bytes + `<pad>`, `<bos>`, `<sep>`, `<eos>` | Same |
| Position representation | **RoPE**, implemented and tested | Same |
| Max tokens | Start with 256; adjust to measured P99 + margin before dataset freeze | 128 provisional |
| Dropout | 0.0 initially | 0.0 |
| Output | Token-embedding-tied LM head | Same |

Length policy: measure **prompt + output** lengths; choose context from real distributions and bounded memory; reject overlength cases instead of truncating. Keep in-range length-shift tests distinct from overlength rejection tests. RoPE avoids untrained absolute-position table rows but **does not guarantee** extrapolation to longer inputs. Learned absolute positions may be compared as an ablation only.

Training format: `<bos>utterance<sep>canonical JSON<eos>`, teacher forced. Cross-entropy includes **all target bytes and EOS**, but excludes prefix and right-padding. With causal attention and strictly right-padded sequences, valid token positions cannot attend to future pad tokens, so no separate pad-key attention mask is required. Test this rather than assume it.

### User-visible intent contract (v1)

| Situation | Exact canonical target shape |
|---|---|
| Fully specified allocation | `{"intent":"allocate","person":"Ingrid","project":"Coral","hours":12,"week":42}` |
| Allocation query | `{"intent":"query_allocations","project":"Coral","week":42}` |
| Clear out-of-domain / impossible / conflicting request | `{"intent":"unsupported"}` |
| In-domain incomplete request | `{"intent":"needs_clarification","for_intent":"allocate","missing":["hours"]}` |

`needs_clarification` is a **learned intent** with `for_intent` and an ordered `missing` list; it does not carry partial extracted slots. For v1 it covers incomplete `allocate` requests only. Missing-field order is the `allocate` schema order (`person`, `project`, `hours`, `week`). Ambiguous referents, negated commands, multiple actions, contradictions, and out-of-range values are labeled `unsupported`; schema/identity validation can separately fail. `hours` must be integer 1–40, `week` integer 1–53. Exact allowed keys per intent; strict parser rejects duplicates, extra prose, invalid UTF-8, and malformed JSON.

**Surface normalization:** person/project values use Unicode NFC and preserve their input spelling/case exactly (the model learns copying, not identity resolution); number words map to integers. Canonical JSON serializes UTF-8, fixed field order, no superfluous whitespace. Entity IDs, aliases, and case-insensitive matches belong only to the optional resolver, not the model's target. The dataset generator enforces these contracts.

### Learned rejection vs score-based rejection

**Primary:** train `unsupported` and `needs_clarification` targets, and report those intent outcomes. **Secondary experiment only:** threshold sequence log-probabilities on a dedicated validation/calibration split, freeze the threshold, then measure risk–coverage on locked tests. Do not call raw scores calibrated confidence. Independently reject schema-invalid and unknown-entity outputs, and never relabel them as correct learned abstentions.

## 3. Preregistered research questions and outcomes

- **H1 — sample efficiency:** At **at most 20,000 unique training examples and 5,000 optimizer updates**, the `tiny` model reaches **≥80% exact semantic match** on the held-out `iid` suite. This is an **aspirational, falsifiable threshold**, not a guarantee. Measure nested **1k / 5k / 20k** training sets, each with a fixed update budget, and plot learning curves. Report all failures.
- **H2 — generalization:** After the generator, baseline, and benchmark protocol are frozen, the human maintainer authors an **independent curated** suite without querying the model. H2 requires **≥50% exact semantic match on supported curated paraphrases**, with denominators and abstention errors shown. If the human suite is unavailable, H2 is **not evaluated**, never silently replaced by agent-authored text.
- **H3 — contribution beyond rules:** On the same **locked curated supported** suite, `tiny` exceeds a frozen, transparent **training-template-only** rules parser by **≥5 percentage points** in exact semantic match. The rules author can inspect training examples/templates and public schemas, but **not** held-out wording or curated samples before freezing its implementation. Report ties and losses as falsification; do not optimize the baseline to fail.
- **H4 — optional semantic integration:** The graph layer rejects invalid identities/constraints and provides reproducible provenance and exact expected triples. SHACL conformance **is not evidence that natural language was interpreted correctly**.

The full test inventory, controls, sample denominators, hash locking, error taxonomy, and decision procedure are normative in [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md). Pre-register thresholds **before any locked test run**. A pass on IID but failure on unseen syntax, names, or human paraphrases is a publishable teaching outcome.

## 4. Training and dataset defaults

Use a deterministic, seedable, versioned generator; agent-authored templates are explicitly marked **LLM-assisted**, not manually curated. Start with 20k unique training utterances, stratified **55% `allocate`, 25% `query_allocations`, 10% `needs_clarification`, 10% `unsupported`**. Include at least eight allocation and six query template families; incomplete allocations and several negative categories must be represented. Keep intent-balanced per-slice metrics even when the global mix is skewed. Separate validation and calibration from *all* final test suites.

Provisional training recipe, adjustable **using training/validation only** and recorded in configs:

| Hyperparameter | Baseline |
|---|---|
| Optimizer | AdamW, betas `(0.9, 0.95)`, eps `1e-8` |
| Peak learning rate | `3e-4`; linear warmup 100 steps, cosine decay to `3e-5` |
| Weight decay | `0.01` on matrix weights; none on biases/norms |
| Batch size | 32 sequences, no accumulation initially |
| Update limit | 5,000, or explicit shorter measured workshop budget |
| Grad clipping | Global norm 1.0 |
| Dropout | 0.0 (optional ablation, not default) |
| Seed | 42; record further seeds if replicated |
| Checkpoint choice | Best validation **loss** at fixed evaluation intervals (250 updates), no test-based selection |

For 1k, 5k, and 20k, use **nested, stratified subsets of one frozen training pool**; hold optimizer-step budget constant for primary comparison, record repeated-example exposure, then optionally add equal-epoch comparisons as a separately labeled analysis. Train from random weights **for every condition**. Benchmarked tokens/step, memory, CPU throughput, and full elapsed time are required before claiming workshop feasibility.

Checkpoint contract: save tensor-only `state_dict` with `torch.save`; load via `torch.load(..., weights_only=True)` with metadata in a separately validated JSON sidecar (model/tokenizer schema, run config, seed, commit, dataset fingerprint, framework and device). Do not load untrusted arbitrary serialized objects. Validate tensor names/shapes and config compatibility. Artifacts should not silently overwrite earlier experiments.

## 5. Incremental teaching and engineering milestones

Every lesson is **Jupytext percent-format `.py` source** in `lessons/` with optional generated `.ipynb`. Run the `.py` scripts in CI; do not require notebook execution in PR CI. Each lesson explains a concept, runs a real experiment, includes a learner exercise, and links to tested `src/` functions. **CI hard budget: 5 minutes** on a standard free runner (tune the test subset to fit; report actual CI time).

| Gate | Learner milestone | Implementation and objective completion criterion |
|---|---|---|
| **M0** | Reproducible environment | Lock verified dependencies, run lint/types/tests, benchmark representative CPU matrix operations; document *measured or unverified* full training feasibility. |
| **M1** | Bytes, probabilities | Byte tokenizer, UTF-8 round-trip, count-based bigram and sampling. |
| **M2** | Learning token transitions | Trainable bigram and loss/gradient descent; show overfitting and validation curves. |
| **M3** | Why context matters | Embeddings and **fixed-window concatenation MLP** before Q/K/V; demonstrate its fixed-context limitation. |
| **M4** | Context via attention | Transparent causal self-attention, RoPE, multi-head pre-norm Transformer; causality, gradient, parameter count, weight tying, one-batch overfit, and bounded greedy generation verified. Measure real `tiny` CPU step time here. |
| **M5** | Supervised language interpretation | Frozen schema, stratified generator, leak-free splits, output-only loss mask, reproducible train loop, benchmark baseline locked, 1k/5k/20k learning curves. No held-out test tuning. |
| **M6** | Do we understand? | Locked tests, exact/slot metrics, negative categories, rule comparison, failure gallery, verified model/checkpoint/run provenance. Human-authored curated suite or explicit H2/H3 `not evaluated`. |
| **M7** | Interactive demonstration | Local CLI (UI optional), typed natural language → raw generation → strict validation, real documented examples, self-guided lessons and clear limitation statement. |
| **M8 optional** | Text → knowledge graph | **Only after core gate and human approval.** See [`docs/semantic-extension.md`](docs/semantic-extension.md). |

**Correctness gate before training:** tests for causal future-token invariance; prefix, EOS and pad loss boundaries; right-padding invariant (`<pad>` never occurs before `<eos>`); actual weight-tied shared tensor storage; greedy generation vs teacher-forced argmax on the same prefix; checkpoint reload equivalence; malformed output rejection; and split/leak integrity. Tests cannot be waived by high accuracy.

## 6. Lean repository layout and ownership

```text
nanogpt/                           # quaat/nanogpt, not Karpathy's nanoGPT
├── plan.md                        # this ~200-line controlling roadmap
├── AGENTS.md                      # mandatory safety/agent instructions
├── README.md                      # learner narrative, setup, exercises
├── pyproject.toml                 # pinned dependencies + optional `semantic` extra
├── docs/
│   ├── evaluation-protocol.md     # preregistration, held-out suites and scoring
│   └── semantic-extension.md      # optional RDF/SHACL phase
├── lessons/                       # 01..08 percent-format .py, optional generated ipynb
├── src/nanogpt_lab/               # tokenizer, bigram, mlp, attention, model,
│                                 # schema, dataset, train, generate, evaluate
├── tests/                         # CPU correctness + bounded smoke tests
├── data/templates/                # disclosed synthetic templates
├── data/fixtures/                 # small examples; hidden curated suite imported at lock
├── configs/                       # tiny.yaml, cpu-smoke.yaml
└── scripts/verify_release.py      # manifest/hash/criteria verification
```

Implement only the files warranted by the current milestone; no placeholder scaffolds for M8. Depend on `rdflib` and `pyshacl` **only** through the future `semantic` optional extra, not the core install.

**Team of two agents + human:** an **implementer** produces bounded changes; a separate **adversarial reviewer** independently inspects correctness, leakage, and claims; the **human maintainer** approves research protocol changes, the curated suite, and merges. Optional *parallel* model vs. dataset-generator implementation is allowed after schema/API contracts freeze and only on separate files/worktrees. No eight-role standing organization, recurring ADR ceremony, or release-evidence bureaucracy; use one short decision note when a material contract changes.

## 7. Integration, acceptance, and release

The core release is complete when: (1) lesson scripts and CI run on a fresh machine; (2) the from-scratch reference architecture passes direct correctness tests; (3) all three sample-size conditions have reproducible training logs or clearly documented execution limits; (4) model and rule baseline are scored on **all available frozen suites**, with counts and failures; (5) locked manifests/thresholds pass `scripts/verify_release.py`; (6) the demo never invents a success by silent rule fallback; and (7) reviewer and human maintainer sign off on the claims. The project **may release with failed hypotheses**, clearly labeled; benchmark incompleteness is never a silent pass.

CLI contract to implement: `python -m nanogpt_lab {tokenize,generate-data,train,evaluate,interpret}` with explicit config/checkpoint arguments, stable nonzero exit codes, and no external side effects. Record exact commands, SHA-256 data manifests, git commit, checkpoint identity, seeds, settings, hardware, runtime, and failed experiments. A release verifier recomputes locked SHA-256 hashes and checks that the committed pass-criteria file matches its approved fingerprint; hashes detect drift, while human review remains the integrity boundary.

**First Codex/Claude instruction:** Read `AGENTS.md`, this plan, and only the document relevant to the assigned milestone. Inspect the repository before editing; create one small reviewable change and run its tests. First implement M0 and decide the stable tokenizer/schema interfaces. After interface approval, the model and template generator may proceed in parallel; the reviewer should challenge leakage and causality before any full training. No model checkpoints or benchmark scores are claimed to exist yet.

## 8. References

Conceptual background only; **no code reuse planned**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [TinyStories](https://arxiv.org/abs/2305.07759), [Karpathy's GPT walkthrough](https://www.youtube.com/watch?v=kCc8FmEb1nY), [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html). Future semantics: [W3C SHACL](https://www.w3.org/TR/shacl/). Check selected framework's actual version and APIs at implementation time.
