# Nonlinear Winners and Survivor Selection

MagicQuant does not keep a model merely because it is different.

A quantized model only survives if it represents a meaningful size/fidelity trade. This can happen in two major ways:

1. **Dominance** — the model is the same size or smaller than another candidate and has lower KLD.
2. **Nonlinear trade improvement** — the model sits between two surviving anchors and improves KLD more efficiently than the expected linear size/KLD trade between them.

This is what MagicQuant means when it says a hybrid “earned its place.”

---

## Step 1: Baseline Collapse

MagicQuant begins by collecting baseline candidates.

These may come from:

- llama.cpp baseline quantizations
- external/custom baseline sources such as Unsloth Dynamic GGUF
- other configured reference sources

Before hybrid discovery matters, MagicQuant first collapses obvious redundancy.

If one candidate is:

- the same size or smaller
- and has lower real KLD

then it strictly dominates the other candidate.

The dominated candidate is removed from the final survivor set.

This is intentionally brutal. MagicQuant is not trying to preserve every familiar quant name. It is trying to preserve the candidates that still represent useful choices.

For example, if a Q5-style candidate has lower KLD than another Q5-style candidate while being the same size or smaller, the weaker candidate does not need to remain in the final table.

---

## Step 2: Hybrid Prediction and Validation

After baseline collapse, MagicQuant searches for hybrid candidates.

A hybrid is a mixed tensor-group configuration. It may combine different quantization choices across major tensor groups such as (including but not limited to):

- `embeddings`
- `attn_q`
- `attn_kv`
- `attn_output`
- `ffn_up_gate`
- `ffn_down`

MagicQuant does not blindly build every possible hybrid. The search space is far too large, and most combinations are not worth testing.

Instead, MagicQuant uses isolated probing and rank-safe prediction to identify candidates that are likely to matter.

A predicted hybrid must still be validated. If the prediction says a hybrid may beat a surviving baseline or occupy a valuable trade space, MagicQuant builds the actual GGUF, benchmarks it, and compares the real result.

Predicted winners do not survive on prediction alone.

They survive only after real validation.

---

## Step 3: Strict Dominance Replacement

The simplest kind of winner is a strict dominance winner.

A candidate strictly dominates another candidate when it has:

- lower KLD
- and the same size or smaller file size

In this case, there is no meaningful tradeoff to preserve. The dominated artifact is simply worse in the measured comparison space.

This applies equally to:

- llama.cpp baselines
- external baselines (like Unsloth)
- MagicQuant hybrids

MagicQuant does not care which provider produced the winner. If the winner is a llama.cpp baseline, it survives. If the winner is an Unsloth baseline, it survives. If the winner is a MagicQuant hybrid, it survives.

The goal is not to make hybrids win.

The goal is to keep the best practical survivors.

---

## Step 4: Near-Baseline Premium Replacement

MagicQuant also allows one very strict exception to pure dominance.

A candidate may replace another candidate if it is slightly larger, but only within a very small configured size premium.

By default, this premium is limited to:

> **maximum +1% size**

This exists because some quantization families are naturally separated by very small file-size differences. For example, nearby low-bit formats may differ by roughly 1% in final file size.

However, being within 1% larger is not enough.

The larger candidate must also prove that the extra bytes are worth paying for. It must beat the expected linear KLD trade for that size increase.

In other words, the model cannot merely be “a little better because it is a little bigger.”

It has to be unusually efficient.

This rule is intentionally strict and fires rarely. Once the candidate is more than 1% larger, MagicQuant treats it as belonging in the interior space between anchors instead of allowing it to replace the smaller baseline directly.

---

## Step 5: Interior Trade-Space Discovery

After collapse and dominance checks, MagicQuant looks at the spaces between surviving anchors.

For example, suppose the final survivor set contains:

- a Q4-range survivor
- a Q5-range survivor

MagicQuant then asks:

> Is there a hybrid between these two sizes that improves KLD more efficiently than the normal size increase from Q4 to Q5?

This is the core of nonlinear winner selection.

---

## What “Nonlinear Winner” Means

A nonlinear winner is a candidate that beats the expected linear trade between two surviving anchors.

Imagine two surviving models:

| Anchor | Size | KLD |
|---|---:|---:|
| Lower anchor | smaller | higher KLD |
| Upper anchor | larger | lower KLD |

The upper anchor is larger, but it improves KLD.

MagicQuant calculates the expected linear trade between them:

> For each byte added between the lower and upper anchor, how much KLD improvement would we expect if the improvement were perfectly linear?

This creates a straight-line expectation between the two anchors.

A candidate inside that space is only interesting if it beats that line.

That means:

- it costs some additional bytes compared to the lower anchor
- but the KLD improvement is better than the linear expectation for those bytes

This is what MagicQuant calls a nonlinear trade improvement.

The candidate is not kept because it is “between Q4 and Q5.”

It is kept because it is better than the expected Q4-to-Q5 trade curve.

---

## Why This Matters

Many candidates can exist between two baselines.

That alone does not make them useful.

A hybrid that is slightly larger and slightly better may still be a bad trade if its improvement is only linear or worse than linear.

MagicQuant is not trying to fill every gap with another file.

It is trying to find candidates that improve the practical frontier.

A nonlinear winner represents a discovered region where the model paid bytes efficiently. It found more KLD improvement than the surrounding anchor trade would normally suggest.

That is why the model deserves to exist.

---

## Not a “Q4.5”

A MagicQuant hybrid between Q4 and Q5 should not be understood as a simple “Q4.5.”

It is not kept because it landed halfway between two quant levels.

It may land 30%, 40%, 50%, or even farther through the space depending on the search configuration and the available candidates.

The exact search window may evolve over time, but the principle does not change:

> The candidate must beat the expected linear size/KLD trade between its neighboring survivors.

If it does not beat that line, MagicQuant does not keep it as an interior winner.

---

## Why MagicQuant Does Not Pack the Graph

MagicQuant intentionally avoids flooding releases with near-duplicate models.

It usually selects only the best nonlinear candidate in a given subspace. In some cases, more than one candidate may be allowed, but the goal is not to publish a dense cloud of tiny variations.

The final release should remain useful.

A user should be able to look at the survivor table and understand that each model represents a meaningful choice.

MagicQuant avoids keeping candidates that only provide negligible improvement, redundant spacing, or linear trade behavior.

---

## The Evidence Archive Is Not the Release Menu

Meaningful spacing is a presentation decision. Strict dominance is a measured fact.

Those should be stored separately:

```text
full measured nondominated archive
    => every real point that survives strict dominance

curated release menu
    => a smaller set chosen for useful user-facing spacing
```

A candidate removed only by spacing can still be scientifically important. It may become relevant when compared with a future run, when a user wants finer high-fidelity choices, or when the curation policy changes.

The current spacing rule uses a fraction of the global survivor size span. This keeps a release clean, but a very wide frontier can cause useful points in a dense high-quality region to be hidden.

MagicQuant should therefore preserve the unspaced nondominated evidence before creating the release view.

For a complete source-aware comparison example and the reproducibility rules, see [Pareto Archives, Release Curation, and Reproducibility](./Pareto-Archives-And-Reproducibility.md).

---

## Baselines Still Often Win

MagicQuant hybrids do not always beat pure baselines.

In many size ranges, llama.cpp or external baselines already provide excellent tradeoffs. If the baseline is the best practical choice, MagicQuant keeps the baseline.

This is expected.

MagicQuant is not designed to prove that hybrids are always superior. It is designed to test whether useful hybrid trade spaces exist.

Sometimes they do.

Sometimes they do not.

When no nonlinear or dominant hybrid exists in a region, MagicQuant leaves that region alone.

---

## Summary

A MagicQuant survivor exists because it passed one or more strict tests:

- it strictly dominated another candidate
- it replaced a near-baseline candidate within a tiny size premium and beat the expected trade
- it occupied an interior space with nonlinear KLD improvement
- it survived final real validation after prediction

A nonlinear winner is not just a middle point.

It is a candidate that improves the size/fidelity frontier.

That is the core idea:

> MagicQuant does not keep hybrids because they are different.  
> It keeps them because the trade was better than expected.
