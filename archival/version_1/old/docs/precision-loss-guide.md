# **Precision Loss: A Practical Philosophy for MagicQuant**

## **Overview**

Quantization is a powerful way to shrink and accelerate large language models, but it always comes at a cost. That cost is **precision loss**: a measurable drift between how a quantized model behaves compared to its original BF16/F16 version.

> "Precision Loss" is the referred statement for the perplexity drift % (aka: PPL Delta percentage) which utilizes the llama.cpp tool for measurement.

**MagicQuant rejects any quantization with more than 5% measured precision loss.**
This does *not* mean >5% loss is useless. What it means is: it no longer meets the quality threshold this project was designed around.

MagicQuant is about producing **small, fast models that stay as close as possible to the original model under the project’s measurements**, especially for trustworthy or agentic automation.

---

# **What Is Precision Loss?**

Quantization does not remove knowledge directly; it changes the numerical representation of the model’s weights. As those weights shift, the model’s behavior can drift from the original.

In V1, MagicQuant used PPL delta as its primary numerical measure of that drift.

* **0% measured precision loss** → no PPL drift from the base model in the tested benchmark
* **higher measured precision loss** → greater PPL drift from the base model

A “better” answer from a quantized model is not evidence by itself that the quantization improved the model. The goal here is not to make the model *different*. The goal is to keep it **as close as possible** to the BF16/F16 original while reducing size and potentially improving speed.

---

# **Precision Loss Tiers (MagicQuant V1 Philosophy)**

You will see quantization discussed in terms like Q8, Q6_K, Q5, Q4.
Those labels are not enough on their own because different architectures can react differently to the same quantization scheme.

MagicQuant V1 therefore evaluated quantization primarily by **measured PPL delta**, rather than assuming quality strictly from bit width or quant name.

Below is the practical breakdown used by V1.

---

## **0.0% – 0.1%

Very Low Measured Drift**

This was the strictest V1 tier.

* Extremely small PPL delta from BF16/F16 in the tested benchmark
* Useful when the goal is to minimize measurable drift as much as possible
* Particularly interesting for very small models where quantization effects can be more visible
* Often more precision than most applications require

At this level, the measured PPL difference can become small relative to the resolution and noise of the evaluation itself.

---

## **0.1% – 1%

Low Measured Drift**

This was V1’s preferred range for serious use.

Models in this range showed only a small PPL delta from the original model under the V1 benchmark setup.

V1 treated this range as especially desirable for:

* complex reasoning workloads
* long-form tasks
* multi-step agentic workflows
* repeatable automation
* applications where fidelity to the original model matters

> For automation that must run cleanly for many cycles,
> **V1 preferred the 0.1% to 1% measured-loss range.**

---

## **1% – 3%**

Moderate Measured Drift**

Once the measured PPL delta crossed 1%, V1 treated the quantization as no longer near-lossless, but still potentially useful.

This range was considered:

* still solid for many workloads
* still coherent in testing
* appropriate for personal use and many development workflows
* a reasonable trade-off when size matters more

V1’s concern was that subtle degradation could become more noticeable as measured drift increased, especially during longer or more demanding tasks.

**MagicCodingMan's personal V1 limit:** *2–2.5% preferred, 3% absolute maximum.*

---

## **3% – 5%

High but Accepted Measured Drift**

This was the upper range allowed by MagicQuant V1.

The project treated this range as increasingly compromised relative to the base model and generally reserved it for cases where the size or speed trade-off justified the additional measured drift.

This range could still be useful for:

* tinkering
* casual chatting
* personal local LLM use
* situations where smaller size matters more than maximum fidelity

…but it no longer aligned with the project’s preferred quality target.

---

# **5%+ Precision Loss

Outside the V1 Release Target**

Some users are comfortable with much larger measured loss when the goal is simply to fit a larger model into limited hardware.

MagicQuant V1 took a stricter approach.

In the V1 philosophy:

* **Past 5% measured PPL loss** was outside the project’s preferred quality range
* **Double-digit measured loss** was considered too far from the original for the project’s intended use
* Very aggressive low-bit quantizations were treated primarily as experimental options rather than dependable defaults

MagicQuant V1 did *not* focus on optimizing or recommending:

* Q3 as a default target
* double-digit measured PPL loss
* quantizations that fit only by accepting substantial measured drift

That was not the purpose of the project.

---

# **Why MagicQuant Used a 5% Hard Limit**

MagicQuant V1 existed to find **small, fast models while keeping measured drift low**.

The project cared about:

* agentic consistency
* semantic fidelity
* repeatable automation
* reproducible behavior
* minimal degradation during long inference sessions

V1 used PPL delta as its primary proxy for those goals and drew a practical line at 5% measured loss.

So the project rule was:

> **If it’s above 5% measured loss, it doesn’t belong in MagicQuant V1’s recommended set.**

That does not mean the quantization is useless. It means it fell outside the quality target V1 was built around.

---

# **V1 Data Did Not Always Follow the Expected Quant Ordering**

People often talk about quants in terms of a simple ordering such as “Q8 = best, Q6 = slightly worse, Q4 = lower, Q3 = lowest.”

MagicQuant V1 repeatedly observed cases where the measured ordering was not that simple.

Its data included examples where:

* some models measured better with Q6 than Q8 under the tested metric
* some preferred hybrid patterns
* some quantization schemes affected FFNs differently from self-attention blocks
* some MoE models tolerated aggressive quantization differently from dense models
* some weight groups tolerated lower precision far better than others

This is why MagicQuant evaluated quantizations based on:

**measured precision loss, not quant scheme alone**
**benchmark drift, not bit-width assumptions**
**the actual measured model, not a fixed expected ordering**

This was central to the V1 project philosophy.

---

# **Final Philosophy Summary**

Under the V1 PPL-based framework:

* **0–0.1%** → Very low measured drift
* **0.1–1%** → Low measured drift; preferred V1 production range
* **1–3%** → Moderate measured drift; acceptable for many uses
* **3–5%** → High but still accepted by the V1 release policy
* **5%+** → Outside MagicQuant V1’s recommended target

MagicQuant V1 was fundamentally about **trustworthy downsizing as V1 knew how to measure it**.
If a downsized model drifted too far from the original under the project’s benchmark, the size savings were not considered worth the trade-off.

---

# **FAQ: Precision Loss, Model Size, and MagicQuant Philosophy**

## **If I can’t fit a low-precision-loss quant on my GPU, should I switch to a smaller model instead?**

**In V1’s philosophy, usually yes.**

If a larger model only fits by accepting substantial measured PPL drift, V1 generally preferred a smaller model that retained a closer benchmark result to its original weights.

Examples:

* If a **20B** only fits at *10% measured loss*, but a **14B** fits around **3–5%**, V1 would generally favor the 14B.
* If a **14B** fits at **Q6_K** with relatively low measured loss, V1 would consider that a strong option.
* If an **8B** fits at **Q8** with very low measured loss and the workload prioritizes fidelity, V1 might prefer that instead.

This also assumes models perform “linearly” with size, which they do not always do.
Some smaller models perform extremely well relative to their parameter count.

**Quality mattered more to V1 than parameter count alone.**

---

## **Why did V1 call 5%+ precision loss experimental?**

Because the project’s intended use emphasized reliability and fidelity to the original model.

V1 treated larger PPL drift as increasing evidence that the quantized model was moving away from the behavior of the base model. For **exploration, tinkering, or fun**, that could still be acceptable. For the project’s recommended releases, it was outside the preferred range.

MagicQuant existed to produce **dependable quants according to its evaluation framework**, so anything over 5% sat outside that mission.

---

## **If you personally preferred <3% loss, why did MagicQuant allow models up to 5% loss?**

Because not everyone has the same needs or the same tolerance.

My personal preference was stricter than the project-wide release threshold.

So the V1 limits were:

* **3%** → my personal maximum for serious use
* **5%** → the project’s maximum accepted measured loss for public release

Anything above that fell outside the reliability target MagicQuant V1 was trying to provide.

---

## **Why was MagicQuant so strict about precision loss anyway?**

Because the project was built around measurable justification for every release.

MagicQuant’s goal was to ensure:

* every release was benchmarked
* every quant had a reason to exist
* no model was included merely because it fit
* baselines were included when their measurements justified them
* hybrids were included when they improved the measured trade-off space
* every upload met the project’s fidelity threshold

Other repositories often upload Q8, Q6, Q5, Q4 simply because those are the variants users expect.

MagicQuant instead tried to publish a quant **only when the V1 measurements justified it**.

The intended promise was:

* the quant represented a measured trade-off
* measured loss stayed within the project’s accepted bounds
* drift was benchmarked rather than guessed
* the uploaded model had been evaluated against alternatives

This is why MagicQuant’s benchmark sheets and testing pipeline existed.

---

## **Why didn’t V1 cover the “1% to 2%” precision loss range in detail? Is that range bad?**

Not bad, just an awkward middle range in the way V1 categorized results.

Here’s why:

### 0.1% – 1%

This was the preferred low-drift range.

### 1% – 2%

This range was still considered very good, but many users tended to fall into two buckets:

1. **They wanted the lowest measured drift possible**
   → so they aimed for <1%

2. **They wanted the smallest practical model**
   → so they were willing to accept 2–5% measured loss for the size/TPS gains

That left the 1–2% tier in a somewhat awkward middle ground in V1’s release philosophy.

It was not considered bad.
It simply was not treated as a distinct target tier.
