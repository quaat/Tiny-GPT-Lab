# Evaluation Protocol and Benchmark Lock — Tiny GPT Lab

> **Normative v1 design (pre-results).** This defines what can be claimed before any locked evaluation. No model has been trained or evaluated by this document. Research questions and threshold values are defined in [`../plan.md`](../plan.md); implementation rules are in [`../AGENTS.md`](../AGENTS.md).

## 1. Claims and comparators

- **H1:** With ≤20,000 unique train examples and ≤5,000 optimizer updates, reach ≥80% IID exact semantic match. Show the 1k/5k/20k learning curve regardless of outcome.
- **H2:** Reach ≥50% exact semantic match on **supported** genuinely human-authored curated paraphrases. If none are available, outcome is **not evaluated**.
- **H3:** Outperform the frozen *train-template-only* rules baseline on supported curated examples by ≥5 **absolute percentage points**, not a relative 5%. Baseline scores are reported everywhere, even where it wins.
- **H4 (optional):** Separately verify graph-side constraint/rejection behavior. Never use RDF/SHACL validity as a proxy for text-understanding correctness.

Reject all claims of language generalization based solely on training loss, attention visualizations, exact copying of familiar strings, or template-shaped examples. Include negative findings.

## 2. Task / target schema invariants

**Allowed model targets (canonical key order):**

```json
{"intent":"allocate","person":"Ingrid","project":"Coral","hours":12,"week":42}
{"intent":"query_allocations","project":"Coral","week":42}
{"intent":"needs_clarification","for_intent":"allocate","missing":["hours"]}
{"intent":"unsupported"}
```

`allocate`: `person`, `project` nonempty NFC Unicode surface strings preserving original casing; `hours` integer 1..40; `week` integer 1..53. `query_allocations`: `project` NFC/case-preserved and `week` 1..53. `needs_clarification`: only for incomplete, otherwise recognizable single-allocation requests, with `missing` being a **nonempty ordered subset** of (`person`, `project`, `hours`, `week`), no partial slot payload, and `for_intent=allocate`. An absent required query parameter in v1 is `unsupported` (out of supported clarification scope). Negation, unresolved ambiguity, conflicting slot mentions, multi-action and impossible-value requests map to `unsupported` when clearly detectable; the model may fail, which must remain visible.

No automatic lowercasing, synonym rewriting, entity alias resolution, punctuation stripping of names, or trusted correction is permitted between generated text and semantic-match scoring. JSON object keys are ordered canonically for **training text**, but semantic evaluator compares parsed typed objects; invalid syntax, duplicate keys, extra keys, hallucinated slots, or wrong values fail semantic exact match. Strict UTF-8 and JSON parsing; numeric strings are not silently converted into integers. NFC normalization occurs **at dataset creation**, not as evaluator repair of a mismatched model output.

**OOD semantics note:** Novel names test copying and role assignment, **not** known-identity resolution. Numbers never expressed in training words (e.g., a novel numeral phrase) test hard out-of-distribution extrapolation; accuracy may legitimately be near zero. Explicitly avoid interpreting that failure as proof the architecture cannot learn number composition under better supervision.

## 3. Dataset protocol and intent balance

Use seedable reproducible generation with stable example IDs and fields `generator_version`, `template_id`, `family_id`, `source` (`llm_assisted_synthetic`, `human_curated`), `seed`, `utterance`, `target`, `split`, `token_length`, and `difficulty_tags`.

**Main training pool:** 20,000 unique utterances with a stratified intent mix: `allocate` 11,000 (55%), `query_allocations` 5,000 (25%), `needs_clarification` 2,000 (10%), `unsupported` 2,000 (10%). Use nested 1,000 and 5,000 subset manifests maintaining intent proportions; ensure all families represented where feasible. Deduplicate within and across splits using an explicitly versioned normalized-utterance key (Unicode NFC + agreed whitespace normalization only, preserving case unless the data-generation contract explicitly fixes it). Reserve separate validation and calibration pools; their source families may overlap train, but their utterances may not.

**Template inventory minimum:** eight independent `allocate` families (different verb/syntax combinations); six `query_allocations` families (e.g., *who*, *which people*, *show allocations*, *list team members*); four incomplete-allocation families; negative categories including out-of-domain requests, impossible numeric values, multi-intent, contradictory instructions and negations. Keep exact inventory and synthetic grammar versioned and published as *LLM-assisted*, even if an AI coding agent writes plausible English. Preserve family IDs and generation scripts.

A suite should be large enough for honest slice metrics. Suggested synthetic suite starting sizes are **500 supported + 100 negatives IID**, **200 per named synthetic generalization suite**, and **100+ for held-out negative categories**; publish actual denominators and do not quietly reduce a hard suite after inspecting results. Human curated suite **size and composition are human-approved before first scoring**; target ≥100 independent supported paraphrases and ≥50 independent negatives, but smaller suites may be used with wide uncertainty intervals and explicit `limited_evidence` status. Curated size is not a pass/fail condition imposed retroactively.

## 4. Locked suite registry — do not collapse into one accuracy

| Suite ID | Held out from training | Meaning | Claim strength |
|---|---|---|---|
| `iid` | Specific utterances | Familiar grammar with new instances; baseline learning check | Low |
| `combination_control` | Whole `(person,project,hours,week)` tuples | **Easy copying/control**; expect approximately IID | Very low |
| `novel_names` | Specific names and project names absent from any training strings | Can the model copy new token sequences and assign them to roles? | Medium |
| `numeral_ood` | Particular spelled-out numeric forms/structures | Numerical linguistic extrapolation; note impossible-OOD cases | Medium, conditional |
| `slot_order_ood` | Entirely held-out slot orders, such as week before person | Structural composition beyond copied slots | Medium |
| `template_family_ood` | Whole generated template families | Synthetic paraphrase shift; may reflect author style | Limited |
| `curated_human_supported` | All human-authored examples; **not visible before freeze** | Independent wording/role generalization; sole H2/H3 evidence | Highest available |
| `negative_known` | Specific negative instances but familiar negative categories | Rejection behavior on trained negatives | Low |
| `negative_category_ood` | **Whole** negative categories absent from train | Does learned `unsupported` generalize? | Medium |
| `curated_human_negative` | Independently human-authored out-of-scope, ambiguous etc. | Critical qualitative evidence on rejection | High |
| `length_unicode_inrange` | Unseen but supported lengths/codepoints within chosen context | Robustness, not length extrapolation proof | Medium |
| `overlength_reject` | Prompts/targets beyond context | Deterministic and explicit rejection, **not** model accuracy | Correctness only |

For OOD construction, document which training features are absent and which lexical substrings overlap. `novel_names` must forbid the full name in any training **prompt or target**, even within a different intent. `slot_order_ood` must hold out structural order, not merely a newly worded familiar order. `numeral_ood` should disclose if component words such as “thirty” were absent entirely (near-impossible zero-shot scenario). Reserve negative domains by category (for example, train on cooking and weather questions, test on HR complaints/financial demands), not only by surface string. Always report confusion between `unsupported`, `needs_clarification`, invalid JSON, and model hallucinations.

### Leakage and cross-split assertions

- No normalized utterance appears in multiple splits or with contradictory targets.
- Template-family holdouts may not appear in training or validation when testing family generalization; changing synonym punctuation does not create a new held-out family.
- Train-only slot catalog excludes `novel_names` names and project strings. Curated content must not be fed into prompts, few-shot examples, rule-creation tools, or validation diagnostics.
- Negative-category holdouts exclude the category and its closely paraphrased prompts from **all train/validation/calibration records**; categorization must be reviewed by a human before unlock.
- Report overlap statistics for lexemes, entity pairs, template families, and canonical utterance hashes. Simple tuple disjointness is explicitly **not** accepted as proof of composition.

## 5. Human curated benchmark — precise anti-contamination workflow

1. Agent-authored generator and training templates are reviewed/frozen at a named Git commit. Document their LLM-assisted provenance.
2. Rules baseline **implementation and allowlisted inputs** (train templates, training examples, public schema only) are frozen and hashed **before any human curated sentences exist in the repository**. An LLM code author necessarily has prior language knowledge; disclose that asymmetry; the mechanical information diet remains restricted and inspectable.
3. Human maintainer independently writes the curated suite in a private document, **without reading model outputs or using a text-generation AI**. The human predefines supported/negative labels and canonical JSON, including ambiguous-case adjudication, and reviews for unwanted trivial duplicates. Model developers do not write or label this suite.
4. Human approves suite definition, size, case provenance, and preregistered success criteria. Commit/import `data/fixtures/curated_human_v1.jsonl` at first unlock. Generate canonical stable manifest and commit SHA-256 alongside *approved* pass-criteria SHA-256. The curated fixture and criteria lock must be frozen before any model or baseline evaluation against it.
5. Run the **single preregistered final evaluation** on the previously selected checkpoint and frozen baseline. Publish all outputs and errors. Repeated runs of exactly the same committed artifact for reproducibility are allowed; choosing a new model after inspecting cases constitutes **exploratory post-test** analysis, not fresh confirmatory H2/H3 evidence.

If an LLM creates, edits, or labels any human-suite item, it is **not** human-independent and must be moved to an explicitly tagged `llm_assisted_curated` exploratory suite. A missing human-curated fixture makes H2 and H3 `not_evaluated`; never synthesize one automatically. Private pre-unlock fixtures cannot be scored in general PR CI.

## 6. Rules baseline fairness

Implement transparent regex/grammar rules or a deterministic parser as a separate module; **never call it as a fallback in `interpret`**. The baseline author may know documented intent schemas, training templates, and a sample of training data; may not use withheld generated templates, curated text, test categories, external pretrained models, or test labels. It must be frozen by commit before the curated suite is authored. Document exact authoring inputs and scripts. Do not intentionally cripple it: include straightforward lexical features that a competent engineer would reasonably derive from training examples. Its expected near-perfect IID/template-combination score is **not** evidence of superiority or inferiority on real paraphrases.

Compare both systems on identical input examples, with transparent rejection handling. Any rule baseline changes after lock are tagged as **post hoc explorations** and cannot retroactively change H3.

## 7. Primary, secondary, and calibration metrics

**Primary per supported suite:** strict **exact semantic match** after parsed JSON and v1 schema validation (numerical values typed; exact surface-slot spelling); malformed output or erroneous learned rejection = incorrect. Separate `schema_valid_rate`, `json_valid_rate`, `intent_accuracy`, and per-slot accuracy (only where slot exists in ground truth). Report counts `[correct/total]`, percentages, per-intent denominators, and a bootstrap confidence interval where a suite is large enough to justify it.

**Negative/rejection:** report learned `unsupported` precision/recall and false accept for each negative category; in-domain incompletes are separately evaluated against `needs_clarification` (including `missing` exact match). Never aggregate all negatives into a single comforting ≤10% false-accept score. Show refusal failures, false rejections of supported requests, invalid JSON, schema invalid, and unknown identity as **different** outcomes.

**Secondary score threshold:** optional token-sequence mean/sum log-probability (definition must be declared). Calibrate threshold only with a distinct validation/calibration pool; freeze it before tests and report risk–coverage curves, accepted fraction, conditional exact-match accuracy, false-accept and false-reject separately. Do not describe an uncalibrated score as a probability of semantic correctness.

**Resource metrics:** measured model parameter count, train unique examples and exposures, tokens processed, step counts, seed, PyTorch/backend versions, CPU/GPU designation and model, peak memory where possible, elapsed training, average step duration (warmup excluded), generation speed. Record initial and final losses **but don't promote loss to language understanding**.

### Learning curves

Train from random initialization on **nested stratified** 1k, 5k and 20k samples with identical architecture, optimizer/schedule and maximum 5k updates. Checkpoint selection uses validation loss only; retain learning curves by step and examples-seen so repeated exposure is visible. Plot held-out **IID** exact match and every synthetic generalization slice vs. dataset size. Run curated suite **once per preregistered selected condition** only after lock; either preregister that H2/H3 evaluates the 20k condition or specify a validation-only selection rule before test unlock. Default: **20k checkpoint** is H2/H3 confirmatory candidate. Record all failed/unfinished training attempts; future multi-seed replication is a distinct approved study.

## 8. Release freeze and verification mechanism

Suggested canonical tracked files (created during implementation):

```text
data/manifests/train-pool-v1.json

data/manifests/validation-v1.json

data/manifests/test-iid-v1.json

data/manifests/test-ood-v1.json

data/manifests/curated-human-v1.json            # exists only after human unlock

docs/benchmark-criteria-v1.json                # exact H1/H2/H3 thresholds

docs/benchmark-lock-v1.json                    # expected SHA-256 hashes + versions

scripts/verify_release.py                     # recalc hashes and enforce approvals
```

Manifests use a defined canonical encoding: JSON UTF-8, Unicode NFC, deterministic key ordering/separators, stable record ordering, newline convention documented. On benchmark freeze, compute SHA-256 of the **exact committed canonical bytes**. `benchmark-lock-v1.json` records each manifest relative path and expected SHA-256 plus the criteria file hash, generator commit, baseline commit, approval date and human-approver field. `verify_release.py` must fail with nonzero exit status on missing manifest, mismatch, unapproved criterion change, undeclared model/evaluator artifact, or missing test score; allow `not_evaluated` only with an explicit reason. CI can verify public manifests; curated hash validation begins only after unlock. A checksum only detects changes relative to a trusted, approved lock; code-owner/human review must protect modifications to both lock and verifier.

**No automatic rebaselining.** If a benchmark truly must change, assign a new version (`v2`), obtain human approval, retain the prior suite/results, and describe changed hypotheses. Training and prediction artifacts identify their exact evaluation version.

## 9. Correctness checks independent of score

Test all of the following before full training: byte tokenizer roundtrip + malformed UTF-8; true causal logit invariance; RoPE apply/offset behavior; shifted loss including first JSON byte and EOS but excluding prefix and pads; right-padding `<pad>` never preceding EOS; real LM-head/embedding storage sharing; teacher-forced argmax/greedy consistency for identical prefixes; save/reload equivalence with JSON sidecar and `weights_only=True`; bounded generation; strict duplicate-key rejection; target-schema property checks; split leakage/family/negative-category controls; reproducible score computation at fixed seed/device.

A high test accuracy is not permission to waive any correctness failure.

## 10. Reporting template

For each model condition: repo commit, data generator/template commit, dataset versions/hashes, tokenizer and output schema version, model configuration/count, selected checkpoint ID and hash, device and measured runtime, seeds, training recipe, validation checkpoint-selection evidence, baseline commit, suite-by-suite `[correct / total]`, invalid JSON and rejection rates, confidence intervals, top recurring error types, and **sampled successes and failures drawn from actual checkpoint outputs**. Indicate which hypotheses were passed, failed, `not_evaluated`, or exploratory. State the limits: synthetic grammar bias, agent-authored template bias, possible baseline-author linguistic prior, and limits of tiny from-scratch models.
