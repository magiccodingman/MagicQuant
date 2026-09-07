# Learning From Existing Quantizations (Tensor Configuration Sources)

MagicQuant does not invent tensor-level quantization strategies.

Instead, it **learns from existing quantizers** and reuses their decisions in a controlled, comparable way.

This is a core design principle:

> **MagicQuant is the judge, not the quantizer.**

---

## What “Provider” Means

When you see a model labeled as:

- `llama.cpp`
- `Unsloth`
- `MagicQuant`

it does **not** mean the same thing in each case.

### llama.cpp

A llama.cpp quant (e.g. `Q5_K`, `IQ4_XS`) is a **baseline quantization strategy**.

Importantly:

> These are not uniform quantizations.

Even when labeled “Q5_K” or “IQ4_XS”, llama.cpp does **not** apply that quant type to every tensor.

Instead, it:
- leaves some tensors at `F32` or `BF16`
- protects sensitive tensors at higher precision (e.g. `Q6_K`)
- applies the target quant only where safe

This means every baseline already contains a **mixed tensor configuration**, even if it is presented as a single quant name.

---

### Unsloth (Dynamic GGUF)

Unsloth Dynamic models follow a similar philosophy:

- identify which tensors are sensitive
- apply higher precision where needed
- compress more aggressively where safe

However, Unsloth often:
- uses a **wider range of quant types**
- adapts more aggressively across tensor groups
- produces more dynamic per-tensor assignments

This can result in lower KLD or different trade characteristics.

But critically:

> Unsloth is still making tensor-level decisions about what to protect and what to compress.

---

### MagicQuant

MagicQuant does **not** decide tensor sensitivity.

It does **not** determine:
- which tensors should be `Q6_K`
- which tensors must remain `F32`
- which tensors can be aggressively quantized

Instead, it:

> **Learns these decisions from existing quantizations and reuses them.**

---

## Step 1: Tensor Grouping

MagicQuant organizes model tensors into **coarse groups**:

- `embeddings`
- `lm_head`
- `attn_q`
- `attn_kv`
- `attn_output`
- `ffn_up_gate`
- `ffn_down`
- `moe_experts`
- `moe_router`

These groups are defined using naming patterns (typically regex-based).

This grouping layer is the main point where architecture awareness is required.

- Most architectures follow stable naming patterns
- New architectures may require minor updates
- Fallback logic exists, but explicit mapping is preferred

---

## Step 2: Learning Tensor Configurations

For each baseline (llama.cpp or external):

MagicQuant:

1. Loads the quantized model
2. Inspects every tensor
3. Records:
   - which quant type was applied
   - which tensors remained `F32` / `BF16`
   - how each tensor maps into a group

This produces a **learned tensor configuration per group**.

Example (conceptual):

| Group | Observed quant types |
|------|---------------------|
| attn_q | Q5_K, Q6_K, F32 |
| ffn_up_gate | IQ4_XS, Q5_K |
| embeddings | Q6_K |

This is not guessed.

It is **directly extracted** from real quantized models.

#### How MagicQuatn Learns & Builds Hybrids

```
        ┌───────────────────────┐
        │  llama.cpp Baselines  │
        │   (Q8, Q6_K, IQ4_XS)  │
        └──────────┬────────────┘
                   │
                   │ Extract tensor assignments
                   ▼
        ┌───────────────────────┐
        │  Tensor Config Cache  │
        │ (per-group mappings)  │
        └──────────┬────────────┘
                   │
                   │ Combine with
                   │
        ┌──────────▼────────────┐
        │   External Baselines  │
        │   (Unsloth, others)   │
        └──────────┬────────────┘
                   │
                   │ Learn + Normalize
                   ▼
        ┌────────────────────────────┐
        │ Unified Tensor Config Pool │
        └──────────┬─────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Hybrid Builder     │
        │ (mix tensor groups) │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Isolation Probing   │
        │ + Prediction Engine │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Real GGUF Build     │
        │ + Benchmark (KLD)   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Survivor Selection  │
        │ (dominance +        │
        │  nonlinear winners) │
        └─────────────────────┘
```


---

## Step 3: Normalization (Equal-Footing Comparison)

External baselines (e.g. Unsloth) are not compared as-is.

Instead, MagicQuant:

1. Learns the tensor configuration from the external model
2. Rebuilds that configuration internally
3. Applies it using MagicQuant’s own pipeline

This includes:

- starting from a controlled base (typically BF16)
- applying the learned tensor assignments
- using MagicQuant’s own imatrix
- normalizing precision (e.g. converting F16 → BF16 when needed)

This is extremely important:

> MagicQuant is not testing the original external artifact directly.

It is testing:

> **the tensor configuration choices under controlled conditions**

---

## What This Means for Comparisons

When MagicQuant reports:

> “X beat Unsloth Y”

it means:

- under normalized conditions
- using the same imatrix and base setup
- the **tensor configuration pattern** performed better

It does **not** mean:

- the original Unsloth release is worse
- Unsloth’s imatrix is worse
- Unsloth’s full pipeline is worse

In fact:

> External providers may outperform MagicQuant’s rebuilt versions in real usage.

This is why:

- MagicQuant links to original upstream models when appropriate
- it does not claim universal superiority over providers

---

## Step 4: Hybrid Construction

Once configurations are learned, MagicQuant builds hybrids by:

- selecting a **baseline configuration**
- replacing specific tensor groups with another learned configuration

Example:

| Group | Source |
|------|--------|
| embeddings | llama.cpp Q8_0 |
| attn_q | llama.cpp Q6_K |
| ffn_up_gate | Unsloth Q5_K_XL |

This does **not** mean:
- “apply Q6_K everywhere in attn_q”

It means:
- **apply the exact tensor pattern observed in that source**

Including:
- which tensors stayed high precision
- which were reduced
- how the quant was distributed

This is also why you may see things like a group tensor receiving `IQ2_M` for example. That's **NOT** a weight that can be assigned to a tensor. It is a full IQ2_M baseline, then learned tensors in that group and that learned IQ2_M group behavior applied to X.

This is also why adding new quantization tactics is incredibly easy. If it's in llama.cpp, MagicQuant can use it.

> Just a note. I've been asked about weirder quants that live in forked branches of llama.cpp and sadly I will be staying with only quants that live in llama.cpp officially. This makes maintenance easier and the models more accessible.

---

## Step 5: Isolation and Learning for Prediction

Each learned tensor group configuration is also used to:

- build isolated test models
- measure its effect on KLD
- feed the prediction system

This allows MagicQuant to estimate:

> “What happens if this group uses configuration A instead of B?”

without brute-forcing the entire combinatorial space.

---

## Historical Sources: Learn the Vocabulary, Not the Old Winner

An older source revision can contain tensor recipes that a newer revision no longer publishes.

That makes historical sources useful, but only at the correct level of abstraction.

MagicQuant should learn:

```text
which tensor assignments existed
which group recipes are independently testable
which quant families are available to the current search
```

It should not learn:

```text
this old final mixture won before
therefore replay the same mixture now
```

Replaying historical winners would bias the current frontier toward a previous model, imatrix, benchmark corpus, and search campaign. Instead, MagicQuant digests the tensor vocabulary, rebuilds those choices under the current controlled conditions, and makes every group behavior earn support again.

The principle is:

> **Learn what configurations exist. Relearn what they do.**

## Pin Every External Revision

A provider label such as “Unsloth Dynamic v2” or “v3” is not a reproducible source identity.

Repositories change. Files can be replaced while retaining familiar names. MagicQuant should therefore record:

- repository identity
- exact revision or commit hash
- artifact filename
- artifact checksum where practical

The repository configuration supports an immutable revision directly:

```yaml
baselines:
  custom_repositories:
    - repo_id: provider/model-gguf
      revision: exact-commit-or-repository-revision
      enabled: true
      allow_as_learning_baseline: true
```

Historical vocabulary sources can remain learning-only by disabling their use as global carriers and explicit group candidates. Their learned group recipes can then be considered through the controlled search without multiplying the carrier space or replaying the source artifact as a winner.

This matters both for future reruns and for cross-run comparisons. Numeric SQLite IDs are local bookkeeping values, not globally stable identities. Two runs can assign the same number to different source recipes.

For the wider evidence and identity rules, see [Pareto Archives, Release Curation, and Reproducibility](./Pareto-Archives-And-Reproducibility.md).

---

## Why This Approach Works

MagicQuant avoids one of the hardest problems in quantization:

> figuring out tensor sensitivity from scratch

Instead, it leverages:

- llama.cpp’s engineering
- Unsloth’s dynamic strategies
- any other valid baseline source

This makes the system:

- **more stable** across architectures
- **more realistic** in its search space
- **less likely to produce broken hybrids**

Because it never invents unsafe tensor behavior.

---

## Key Principle

MagicQuant does not ask:

> “What tensors should be compressed?”

It asks:

> “Given known safe tensor configurations, which combinations produce the best trade?”

---

## Summary

MagicQuant learns tensor configurations by:

1. Extracting real tensor assignments from baseline quantizations
2. Grouping them into meaningful tensor categories
3. Rebuilding them under controlled conditions for fair comparison
4. Reusing them to construct hybrid candidates
5. Measuring and validating their effects

It does not replace quantizers.

It builds on them.

> **MagicQuant does not decide how to quantize a tensor.  
> It decides which quantized configurations are worth keeping.**
