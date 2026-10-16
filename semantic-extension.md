# Optional Extension — From Tiny GPT Interpretations to a Knowledge Graph

> **Status:** Post-core proposal; **do not implement before M7 release and human authorization**. The complete neural learning experiment stands on its own. This extension may be called **M8**. It must not change the H1/H2/H3 experimental contract.

## 1. Research and educational purpose

Demonstrate a modest neurosymbolic pipeline: a tiny, from-scratch language model predicts an intent and slots; an **independent deterministic layer** checks and maps that prediction into RDF; SHACL validates graph-local rules; the resulting staging graph can be inspected with SPARQL. This illustrates the relationship between probabilistic interpretation and explicit semantic constraints. It **does not** establish that the model has understood the utterance, because a coherent but incorrect JSON prediction may still pass all symbolic checks.

**Teaching outcome:** A learner traces one instruction through token prediction, JSON validation, identity lookup, ontology mapping, validation, and a provenance-linked graph query; then observes how deliberate failures are rejected at each boundary.

## 2. Non-goals and safety boundary

- No production planning database, real employee/project records, SharePoint, cloud service, external knowledge graph, or user-visible allocation mutation.
- No model-generated SPARQL or arbitrary RDF triples trusted as executable commands. The model's contract remains the same validated JSON intents as in `plan.md`.
- No claim that SHACL checks semantic agreement with the original sentence. That requires independent, human-authored interpretation ground truth.
- No general ontology induction, RAG, agent tools, inference loops, or model self-modification in v1.
- No new core dependency. Add `rdflib` and `pyshacl` only under `pyproject.toml` optional extra `semantic`, with tested pinned versions/lockfile and a small independent CI job.

## 3. End-to-end pipeline and explicit outcomes

```text
Natural-language request
    │
    ▼
Neural tokenizer → causal Transformer → raw generated JSON text
    │
    ▼
Strict UTF-8 + JSON parse / no duplicate fields
    ├── fail → `invalid_json`
    ▼
Typed v1 intent-schema validation
    ├── fail → `schema_invalid`
    ├── `unsupported` → `learned_unsupported` (never map to a graph)
    ├── `needs_clarification` → `learned_clarification` (never map to a graph)
    ▼
Deterministic identity resolver (read-only fictitious catalog)
    ├── fail → `unknown_entity` or `ambiguous_entity`
    ▼
JSON → deterministic RDF staging graph + provenance
    ├── fail → `mapping_error`
    ▼
SHACL validation of staged graph
    ├── fail → `shacl_invalid` + diagnostic report
    ▼
Read-only inspection / authorized local in-memory graph publish
    └── `staged_valid` / `published_local`
```

Every outcome is separate and machine-readable. No silent input coercion, default people/projects, fallback regex extraction, or auto-fixing of neural errors. Never substitute `unsupported` for invalid JSON or for a failed graph lookup when computing **learned** rejection metrics.

## 4. Minimal ontology, scopes, and namespace policy

Use a fictitious namespace such as `https://example.org/nanogpt-planning/` and a separately named shape graph. Design a deliberately small ontology rather than importing large existing ontologies for educational appearance:

| Class | Minimum properties | Why it exists |
|---|---|---|
| `ex:Person` | `ex:displayName` | Fictitious referenced worker |
| `ex:Project` | `ex:displayName` | Fictitious project |
| `ex:AllocationProposal` | `ex:forPerson`, `ex:forProject`, `ex:hours`, `ex:week` | Proposed allocation, not an actual assignment |
| `ex:InterpretationRun` | `ex:utteranceHash`, `ex:checkpointId`, `ex:taskSchemaVersion`, `ex:createdAt`, `ex:status` | Traceability of model-originated statement |

`ex:AllocationProposal ex:derivedFrom ex:InterpretationRun` connects every proposal to its origin. Use RDF datatypes (`xsd:integer`, `xsd:dateTime`) and explicit IRIs. **Do not model the proposed allocation as an accepted or authorized allocation.** `query_allocations` yields a structured read-only query plan, not a mutation or new `AllocationProposal`.

Choose one documented stable identifier policy for the demonstration. For fixtures, reserve catalog IDs such as `person/ingrid-01` and `project/coral-01`; do not use bare surface names as globally unique production identities. Synthetic resolution can use exact NFC names at first; case-folding/aliases are optional experiments and must report collisions. Names unseen by the model may be perfectly copied yet fail catalog resolution; this is an **identity coverage** failure, not an automatic interpretation failure.

## 5. SHACL contract

Version shapes in `ontology/planning-shapes.ttl` and keep examples in `tests/semantic/fixtures/`.

- `AllocationProposal` must have **exactly one** `ex:forPerson` pointing to an existing catalog `Person`, and exactly one `ex:forProject` pointing to a known `Project`.
- Exactly one integer `ex:hours` in `[1,40]` and one integer `ex:week` in `[1,53]`.
- Every proposal has exactly one `ex:derivedFrom` linked to a run containing required provenance properties.
- No unknown terms in the designated proposal shape, if enforcing a closed shape; explicitly allow known provenance/type terms.
- Validation is conducted over a graph containing the staged proposal **and the relevant catalog triples**; don't assume referential existence merely because an IRI has the correct syntax.

Tests must include: valid proposal; missing/duplicate person relation; unknown catalog entity; out-of-range hours/week; wrong datatypes; absent provenance; malicious unexpected predicate; wrong node types. Generate a SHACL diagnostic report for invalid input, but never call a SHACL failure proof that the original utterance was semantically incorrect.

## 6. Provenance and reproducibility

Record a privacy-preserving utterance SHA-256 and, for this toy offline demo only, optionally the literal synthetic input text. Record model checkpoint hash, git commit, tokenizer/schema versions, greedy decoding settings, catalog version, mapping version, shape version, validation result, and graph-content hash. Keep model and semantic evidence **separable** so the source of failure is identifiable.

A repeatable staging graph can mint IDs from a documented hash over canonical interpreted JSON and run/checkpoint identity; this is an educational **proposal identity**, not a persistent real-world allocation ID. Hash collisions and accidental deduplication are outside the intended small demo but deterministic serialization must be tested.

## 7. Implementation sequence and owner responsibilities

This is a **separate phase** with one implementer, one adversarial reviewer, and human approval:

| Work slice | Build | Acceptance |
|---|---|---|
| S0 — contracts | Versioned namespace, fictional entity catalog, ontology and target→RDF mapping table | Human approves naming/authorization boundary |
| S1 — mapping | `resolver.py`, `mapper.py`, strict outcomes and local in-memory RDF | Deterministic exact triple fixtures; unknown/ambiguous entity errors |
| S2 — constraints | SHACL shapes and validator with structured validation report | Positive/negative fixture matrix passing on CPU |
| S3 — inspection | Local `query_allocations` → **parameterized, deterministic SPARQL** query over permitted graph | Query never accepts raw model-authored SPARQL; stable expected rows |
| S4 — evidence | Traced CLI demo (`--show-graph`, `--show-provenance`), tests, docs, independent review | Full chain and every failure class can be reproduced |

Optional suggested CLI after authorization: `python -m nanogpt_lab interpret --checkpoint ... --text ... --semantic --show-trace`. The extension should be installable via `pip install '.[semantic]'` **once real packaging defines that extra**; this is a future acceptance contract, not a command claimed runnable today.

## 8. Evaluation — keep three different questions separate

1. **Linguistic correctness:** Did the neural model predict the human-ground-truth `intent` and slot values? Use the frozen M6 interpreter benchmark **unchanged**.
2. **Structural correctness:** Given an arbitrary JSON interpretation, did deterministic JSON→RDF mapping and SHACL validation accept exactly the well-formed allowed graphs? Test using independently authored fixture JSON, including incorrect-but-schema-valid content.
3. **System-level behavior:** Starting with real checkpoint predictions, what fraction are JSON-valid, schema-valid, catalog-resolvable, SHACL-valid, and **linguistically correct among all original requests**? Give each stage its own numerator/denominator and show errors.

A wrong model prediction such as swapping the person and project might still produce conformant RDF if both strings identify known resources. Do not erase this failure under a high SHACL pass rate. Also show legitimate novel-name predictions that fail identity resolution because the demonstration catalog is intentionally closed.

## 9. Future experiments (not M8 commitments)

- **Lexicon-backed entity linking:** explicit aliases, collision policies, identity evidence, and disambiguation; evaluate separately from model accuracy.
- **Ontology-informed supervision:** train the same tiny architecture with concept-tagged targets or structured relation triples, then compare to JSON-only training with identical split and compute budgets.
- **Constrained generation:** grammar- or schema-guided token decoding as an independent condition, measuring format gains **without implying improved intent accuracy**.
- **Neural proposal plus symbolic feedback:** demonstrate why SHACL violations are diagnostic signals but do not uniquely identify the correct interpretation.
- **Norwegian mini-domain:** new independently authored Norwegian corpus and test set, without assuming the English model can transfer zero-shot.
- **Semantic provenance ledger:** append-only local proposal records and versioned graph snapshots; no operational mutations.
- **Scaling/ablation study:** compare model width, RoPE vs. learned absolute positions, byte vs. compact tokenizer, corpus pretraining vs. supervised-from-scratch, and rule-informed supervision under honest held-out protocols.

## 10. Final semantic-extension release gate

All mapping/SHACL fixture tests pass; catalog collisions/unknowns reject explicitly; no model-authored query execution; no external side effects; provenance identifies all versions; semantic and linguistic correctness are reported independently; reviewer verifies the trust boundary; human approves the optional-phase merge. Any failure is published rather than hidden behind a graceful-looking user interface.

References: [W3C SHACL](https://www.w3.org/TR/shacl/), [RDF 1.1 Concepts](https://www.w3.org/TR/rdf11-concepts/), [RDFLib](https://rdflib.readthedocs.io/), [pySHACL](https://github.com/RDFLib/pySHACL).
