# nanogpt
A small educational project for building a GPT-style model from scratch

## 1. Executive summary

### The idea

Build a tiny but authentic **decoder-only, causal Transformer**—a miniature GPT-style language model—from randomly initialized weights. Present its construction as a sequence of follow-along, executable lessons. In the final core exercise, train it to interpret natural-language resource-allocation instructions and questions into canonical JSON.

The example domain is deliberately narrow: projects, people, hours, and calendar weeks. A participant should be able to type:

```text
Assign Ingrid to project Coral for 12 hours in week 42.
```

and aim to obtain:

```json
{"intent":"allocate","person":"Ingrid","project":"Coral","hours":12,"week":42}
```

A paraphrase such as “Book twelve hours of Ingrid's time for Coral in week 42” should ideally yield the same interpretation. A question such as “Who is working on Coral in week 42?” should yield a different intent:

```json
{"intent":"query_allocations","project":"Coral","week":42}
```

These are **target behaviors**, not claims about model performance. The model must learn token prediction from examples. A held-out benchmark must determine whether it generalizes beyond memorized sentences and templates.

### Central thesis

A small causal Transformer can be an effective teaching tool for demonstrating the mechanics of modern language modeling, and a useful experiment in whether carefully restricted language-to-structure mapping can be learned with fewer than roughly one million parameters. The project does **not** claim general language understanding, reliable open-domain interpretation, or reasoning comparable to a large language model.

### Two deliberately separate outcomes

- **Core deliverable (mandatory):** a runnable, fully documented tiny GPT-style model and an evaluated, narrowly scoped natural-language-to-JSON interpreter.
- **Semantic extension (later, optional):** deterministic conversion of validated interpretations to an ontology-backed RDF knowledge graph, followed by SHACL validation, traceable queries, and a small neurosymbolic demonstration.

### Why this is pedagogically useful

Participants will observe the transition from simple bigram probabilities to embeddings, self-attention, Transformer blocks, cross-entropy training, and autoregressive decoding. They will then see the distinction between minimizing training loss and learning reusable linguistic patterns. The knowledge-graph extension illustrates why neural interpretation and symbolic constraint checking are complementary, not interchangeable.

### Evidence and cautions

- The Transformer architecture originates in *Attention Is All You Need* (Vaswani et al., 2017).
- *TinyStories* (Eldan and Li, 2023) demonstrates coherent generation with language models substantially smaller than common commercial LLMs when training data is strongly constrained. It does **not** establish the specific accuracy of the interpreter proposed here.
- Synthetic text-to-JSON supervision is **not equivalent** to broad natural-language pretraining. This project should describe its later stage accurately as domain-specific conditional autoregressive training from random initialization; optional preliminary corpus pretraining is a separately measured experiment.

References are collected in §19.

---
