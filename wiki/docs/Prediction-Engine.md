# MagicQuant Prediction Engine

## Rank-Safe Isolation Prediction, Practical Gravity, and Controlled Context Learning

MagicQuant is a **prediction-guided validation system** for GGUF hybrid quantization. It measures a small number of physically real tensor-group samples, uses those samples to build a rank-safe prediction space, searches that compressed space for candidates that could meaningfully improve the final frontier, then builds and benchmarks only the candidates that deserve a real test.

The main path is the DuckDB/rank-safe prediction engine. A newer secondary path, the **smart baseline-tuning fallback**, runs only after a normal dominance, premium, or interior discovery attempt fails to validate a winner. That fallback does not pretend to out-predict the full engine. It uses the same isolated measurements more conservatively, looking for small MDA-backed “free lunch” swaps and tightly budgeted protection trades that may have been intentionally pruned away from the main prediction space.

MagicQuant can also repeat a bounded number of useful comparisons inside controlled high-, mid-, and low-fidelity blankets. These probes do not replace normal isolation. They test whether an isolation ranking transfers when the surrounding model enters a different compression regime. The complete design is documented in [Regime-Aware Tensor Search](./Regime-Aware-Search.md).

The prediction engine is allowed to be approximate because it is not the final judge. Its job is to decide where compute should be spent. The real benchmark remains the final authority.

The system is built around a practical belief:

> Most quantization behavior follows gravity. When you lower fidelity, damage usually increases. When a candidate appears to violate that rule, the violation must be tested, scoped, and either rejected as normal gravity or learned as a contextual exception.

That is the philosophy behind the current engine.

---

## Table of Contents

1. [The One-Page Mental Model](#the-one-page-mental-model)
2. [Vocabulary](#vocabulary)
3. [Two Worlds: Prediction Space vs. Real Benchmark Truth](#two-worlds-prediction-space-vs-real-benchmark-truth)
4. [Tensor Groups and Effective Quantization](#tensor-groups-and-effective-quantization)
5. [MagicQuant Gravity: The Monotonic Degradation Assumption](#magicquant-gravity-the-monotonic-degradation-assumption)
6. [The Normal Isolation System](#the-normal-isolation-system)
7. [The Additive KLD Predictor](#the-additive-kld-predictor)
8. [Predicted Size](#predicted-size)
9. [Bit-Stress Interaction Fitting](#bit-stress-interaction-fitting)
10. [Rank-Safe Projection with PAVA](#rank-safe-projection-with-pava)
11. [Prediction Confidence](#prediction-confidence)
12. [DuckDB Materialization](#duckdb-materialization)
13. [Bad Trades and Search-Space Pruning](#bad-trades-and-search-space-pruning)
14. [Candidate Selection: How a Hybrid Earns a Real Benchmark](#candidate-selection-how-a-hybrid-earns-a-real-benchmark)
15. [Smart Baseline-Tuning Fallback: Conservative Free-Lunch Search](#smart-baseline-tuning-fallback-conservative-free-lunch-search)
16. [Anomaly Detection: Respecting Violations of Gravity](#anomaly-detection-respecting-violations-of-gravity)
17. [How Anomaly Rules Adjust Prediction Space](#how-anomaly-rules-adjust-prediction-space)
18. [The Qwen3.6-27B Q8 vs Q6 Example](#the-qwen36-27b-q8-vs-q6-example)
19. [Repeatable Algorithm](#repeatable-algorithm)
20. [What MagicQuant Claims, and What It Does Not Claim](#what-magicquant-claims-and-what-it-does-not-claim)
21. [Reimplementation Checklist](#reimplementation-checklist)

---

# The One-Page Mental Model

Imagine you have a model split into major tensor groups:

```text
embeddings
lm_head
attn_q
attn_kv
attn_output
ffn_up_gate
ffn_down
moe_experts
moe_router
```

A naive hybrid quantization engine might try every possible combination:

```text
Q choices per group, G groups => Q^G combinations
```

That explodes immediately.

MagicQuant does something more practical.

It asks:

```text
What happens if I quantize only this group?
What happens if I quantize only that group?
What does each group cost when isolated?
```

Then it builds a predictive map.

The map is not reality. The map is a controlled prediction space. It gives each candidate a predicted size, predicted KLD, confidence, and rank.

From there, MagicQuant asks sharper questions:

```text
Can this candidate strictly dominate an existing anchor?
Can it replace a nearby baseline with enough KLD improvement?
Can it land between two anchors and beat the boring linear tradeoff line?
Can a known synergistic peer group rescue a candidate that bad-trade pruning removed too aggressively?
Can a small guarded fallback recover an isolation-proven free lunch if the main prediction path finds nothing?
Can it expose a real anomaly where lower-bit choices beat their higher-bit twins?
Can isolated truth prove that a lower-declared-bit choice should inherit higher-bit stress behavior?
```

If a candidate passes prediction-space selection, it is physically built and benchmarked.

If the real benchmark fails, the candidate dies.

If it succeeds, it joins the real frontier and may eliminate other artifacts by strict dominance or meaningful spacing.

That is MagicQuant:

```text
Measure local physics.
Prune aggressively.
Second-chance known synergies when pruning may have clipped useful search space.
Predict the frontier.
Let isolated lower-bit winners inherit the stress truth they earned.
Try guarded free lunches only when prediction fails.
Probe suspicious violations.
Validate in reality.
Publish only survivors.
```

---

# Vocabulary

## Real benchmark truth

A physically built GGUF file that was benchmarked. This includes real KLD, PPL, size, and metadata. Real benchmark truth is authoritative.

## Prediction space

The internal coordinate system used to rank candidate combinations before building them. Prediction-space KLD is not promised to equal final real KLD. It is a **relative damage coordinate** used for candidate ordering.

## Tensor group

A logical family of tensors, such as `attn_q`, `attn_kv`, `ffn_down`, or `moe_experts`.

## Active group

A tensor group that exists in the current architecture and participates in search.

For example, a dense model may not use `moe_experts` or `moe_router`. A hybrid architecture may have unusual SSM or attention layouts. MagicQuant operates on the active tensor group profile for the current architecture.

## Base quant

The default quantization used by a candidate unless a tensor group overrides it.

## Group override

A tensor group-specific quantization choice.

For example:

```text
base        = Q8_0
ffn_down    = Q6_K
attn_output = Q8_0
```

Here, `ffn_down` explicitly uses `Q6_K`; any group without an explicit override inherits the base quant.

## Effective quant

The actual quantization state a group ends up using after base inheritance is resolved.

```text
effective_quant(group, candidate) =
    group override, if present
    otherwise candidate base quant
```

## Anchor

A known reference point on the size/KLD frontier. Anchors can be pure baselines, external baselines, or validated MagicQuant hybrids.

## Strict dominance

A candidate strictly dominates an anchor if it is no larger and has lower real KLD.

```text
candidate.size <= anchor.size
candidate.kld  < anchor.kld
```

## Interior better-than-linear candidate

A candidate between two anchors that beats the straight-line KLD tradeoff between those anchors.

This matters because being “between” two models is not enough. A candidate must be better than the boring interpolation to earn a slot.

## Gravity

MagicQuant’s practical name for the normal expected direction of quantization damage.

Higher-fidelity quantization usually has equal or lower KLD than lower-fidelity quantization when the rest of the context is held constant.

## Anomaly

A validated contextual violation of gravity.

For example, a lower-bit candidate may be smaller and have lower KLD than a higher-bit twin in the same quantized context. MagicQuant treats that as an exception to learn, not as proof that gravity no longer exists.

## Baseline blanket

A uniform baseline state used by the smart fallback.

For example, if the anchor is a pure `UD-Q4_K_XL`-style learned baseline, the smart fallback starts with every active group effectively using that same baseline state:

```text
embeddings   = UD-Q4_K_XL
lm_head      = UD-Q4_K_XL
attn_q       = UD-Q4_K_XL
attn_kv      = UD-Q4_K_XL
attn_output  = UD-Q4_K_XL
ffn_up_gate  = UD-Q4_K_XL
ffn_down     = UD-Q4_K_XL
```

Then it considers isolated group swaps against that blanket.

The smart fallback only starts from pure or uniform anchors. It does not begin from an already non-uniform hybrid, because then the fallback would no longer know whether a swap is a clean local improvement or a tangled interaction with an existing hybrid pattern.

## Smart baseline-tuning fallback

A conservative post-failure candidate generator.

It runs after a normal strict dominance, near-baseline, or interior prediction window fails to validate a winner. It does not query the main DuckDB prediction rows. It uses SQLite benchmark truth, base-only anchors, and single-group isolation samples to build a small number of guarded candidate plans.

Its guiding question is:

```text
If prediction-space cheese found nothing,
can isolated truth still show a same-size, smaller, or tightly budgeted protection trade worth trying?
```

## Free lunch

A group swap whose isolated measurement is both better in KLD and same-size-or-smaller relative to the blanket group state.

Shape:

```text
candidateGroup.size <= blanketGroup.size
candidateGroup.kld  <  blanketGroup.kld - epsilon
```

A free lunch is not assumed to be a guaranteed global win. It is only a very cheap reason to build one more candidate when the normal prediction path has failed.

## Brain protection

A premium/interior fallback pattern where MagicQuant spends a small allowed size budget protecting one or more disproportionately sensitive groups.

This is less certain than strict free lunch. It is a controlled gamble:

```text
This group costs bytes,
but isolated truth says it prevents a lot of damage,
and the requested size window still has room.
```

Brain protection is how the fallback can sometimes discover nonlinear premium or interior winners that the main prediction space did not select.

## Tensor group synergy

A static, explicit relationship between tensor groups whose downstream hybrid behavior can be coupled enough that pruning one group in isolation may accidentally remove useful combined search space.

The current hardcoded synergy is:

```text
ffn_up_gate + ffn_down
```

This does not mean the groups are merged. They still have separate measurements, separate effective quantization states, and separate additive KLD contributions.

It means MagicQuant knows they are connected enough to deserve one narrow post-pruning recovery pass.

## Synergy second-chance review

A post-bad-trade review pass for tensor group synergies.

If a quant survives in one member of a synergy group but is removed from the other member purely by bad-trade pruning, MagicQuant can ask the bad-trade question again with one deliberate rule change:

```text
ignore PPL as an elimination signal
keep KLD bad-trade logic active
```

If the candidate no longer fails the KLD-only review against the original bad-trade anchor, it is restored for that group.

This is not a broad rescue mechanism. It does not rescue dominance removals, equivalent-truth collapses, final cleanup removals, unsupported states, or candidates that still fail the KLD-only bad-trade check.

## Isolation bit-truth override

A bit-stress exception earned by isolated measurements.

Normally, interaction stress uses a quant's declared bit range:

```text
Q8_0      => about 8-bit stress truth
Q6_K      => about 6-bit stress truth
Q4_K      => about 4-bit stress truth
UD-IQ3_S  => about 3-bit stress truth
```

But if a lower-declared-bit candidate is same-size-or-smaller and lower-KLD than a higher-declared-bit candidate in the same isolated group, MagicQuant treats the lower-bit candidate as having inherited the highest stress bit truth it defeated for that group/baseline pair.

This does not rewrite the real quantization. It does not make the candidate globally Q8. It only prevents the interaction model from punishing an isolated-proven winner as if its declared bit range were the whole story.

---

# Two Worlds: Prediction Space vs. Real Benchmark Truth

This distinction is the spine of the whole system.

MagicQuant has two worlds:

```text
1. Prediction space
2. Real benchmark truth
```

They are related, but they are not the same thing.

## Real benchmark truth is final

A real benchmark answers:

```text
What actually happened after this GGUF was built and benchmarked?
```

Real benchmark truth includes:

```text
actual size
actual KLD
actual PPL
actual PPL delta
actual tensor mapping
actual benchmark category
actual imatrix context
```

If a candidate fails real validation, it does not survive.

## Prediction space is for ranking

Prediction space answers:

```text
Based on measured local behavior, which candidates are worth physically testing?
```

Prediction-space KLD is not supposed to be a perfect literal forecast of the final benchmark KLD.

It is better understood as:

```text
relative predicted damage coordinate
```

The prediction engine can be numerically imperfect and still be excellent if it ranks the right candidates high enough to test.

That is why MagicQuant can say:

```text
This predicted KLD may not be the final real KLD,
but this candidate is worth validating.
```

Then the real benchmark decides.

## Why this matters

If you misunderstand this, the whole system looks stranger than it is.

For example, MagicQuant may use an additive KLD model where several isolated KLD values are summed together. That sum is not claiming:

```text
The final model will exactly have this KLD.
```

It is claiming:

```text
Relative to other candidates in the same prediction bucket,
this candidate appears less or more damaged.
```

That is enough to guide search.

The final truth is always benchmarked.

---

# Tensor Groups and Effective Quantization

MagicQuant does not reason about every tensor independently during candidate search. It groups tensors into architecture-aware buckets.

Common groups include:

```text
embeddings
lm_head
attn_q
attn_kv
attn_output
ffn_up_gate
ffn_down
moe_experts
moe_router
```

Not every architecture has every group. Unused groups are forced out of the active search space.

A candidate can be represented as:

```text
candidate = base quant + optional group overrides
```

Example:

```text
base        = Q8_0
embeddings  = Q8_0
lm_head     = Q8_0
attn_q      = IQ4_XS
attn_kv     = Q5_K
attn_output = Q8_0
ffn_up_gate = Q6_K
ffn_down    = Q5_K
```

The prediction engine does not care merely what is written as an override. It cares about the **effective** quantization state:

```text
e_g(c) = override_g(c) if override exists
       = base(c)       otherwise
```

Where:

```text
g = tensor group
c = candidate
```

So if:

```text
base = Q6_K
attn_output = null
```

then:

```text
e_attn_output(c) = Q6_K
```

If:

```text
base = Q6_K
attn_output = Q8_0
```

then:

```text
e_attn_output(c) = Q8_0
```

This is critical because MagicQuant predicts behavior from effective group states, not from the cosmetic shape of the candidate string.

## External and custom baselines

External baselines, such as Unsloth Dynamic variants, may be mapped into a normalized baseline identity for prediction lookup when possible.

That does not erase the external artifact’s real benchmark truth.

It simply lets prediction space reuse an isolation library when the external file’s effective tensor behavior corresponds to a known quantization family.

In other words:

```text
Prediction may normalize for lookup.
Validation still uses real benchmark truth.
```

---

# MagicQuant Gravity: The Monotonic Degradation Assumption

MagicQuant’s default world model is gravity.

The formal version is the **Monotonic Degradation Assumption**, or MDA:

> If the surrounding context is held constant, moving a tensor group from a higher-fidelity quantization to a lower-fidelity quantization is expected to produce equal or worse KLD most of the time.

Example:

```text
same candidate context
same groups
same benchmark bucket
only attn_output changes
```

Gravity expects:

```text
KLD(attn_output = Q8_0) <= KLD(attn_output = Q6_K)
```

Most of the time, this is true enough to be useful.

But it is not a law of the universe.

It is an engineering prior.

## Gravity is not denial of weirdness

Quantization gets weird.

Sometimes a lower-bit format fits a tensor distribution better. Sometimes a group interacts with another group in a way that cancels damage. Sometimes two bad-looking local decisions produce a surprisingly good global model. Sometimes a lower-bit candidate beats a higher-bit twin by enough to matter.

MagicQuant does not deny this.

It separates the cases:

```text
normal gravity:
    lower fidelity is worse or not meaningfully better

noise or tiny inversion:
    lower fidelity wins by too little to matter

validated anomaly:
    lower fidelity wins by enough, in context, to change search behavior
```

The old failure mode would be to either worship every tiny inversion or ignore all inversions.

MagicQuant does neither.

It treats gravity as the default and anomalies as scoped, validated exceptions.

## The practical rule

A violation of gravity must earn attention.

A tiny Q6-over-Q8 win buried in noise does not deserve a combinatoric festival.

A candidate that beats a Q8 anchor by lower size and meaningfully lower KLD absolutely deserves attention.

This is the operating philosophy:

```text
Respect gravity until reality proves a violation is meaningful.
When reality proves it, learn the violation without burning down gravity.
```

---

# The Normal Isolation System

MagicQuant learns local tensor-group behavior through isolation samples.

The normal isolation system uses a carrier setup:

```text
base quant = Q8_0
all active tensor groups = native exact precision
one tested group = target quant
```

Native exact precision means the model’s native exact state for the run, such as BF16/F16/F32 depending on source and conversion.

A base-only isolation anchor looks like:

```text
base        = Q8_0
embeddings  = native exact
lm_head     = native exact
attn_q      = native exact
attn_kv     = native exact
attn_output = native exact
ffn_up_gate = native exact
ffn_down    = native exact
```

A single-group isolation sample looks like:

```text
base        = Q8_0
embeddings  = native exact
lm_head     = native exact
attn_q      = native exact
attn_kv     = native exact
attn_output = native exact
ffn_up_gate = native exact
ffn_down    = Q6_K
```

Everything is held exact except the one tested group.

This creates a measurement:

```text
D(ffn_down, Q6_K)
```

The same is repeated for each active group and each relevant quantization target.

## Q8 is not source of truth

This part is subtle but extremely important.

MagicQuant uses Q8 as the **carrier** for isolation logistics and prediction-space structure.

It does not treat Q8 as the source of truth.

The source of truth is always real benchmark truth.

Also, in the current prediction model:

```text
native exact aliases are zero-damage references
Q8_0 is not a zero-damage alias
```

That means:

```text
D(g, native exact) = 0
D(g, Q8_0)         = measured isolated Q8 KLD for that group
D(g, Q6_K)         = measured isolated Q6_K KLD for that group
```

This matters because Q8 is still quantized. It may be very good, but it is not BF16. MagicQuant therefore scores Q8 from measured Q8 isolation data rather than pretending it has no damage.

## What isolation teaches

Isolation teaches a local marginal behavior:

```text
When everything else is exact, what does this group/quant choice look like?
```

This is not enough to explain every nonlinear interaction.

But it is extremely useful.

It gives MagicQuant a stable local physics map:

```text
D(group, quant)
S(group, quant)
PPL(group, quant)
```

The prediction engine then uses that map to rank full hybrid combinations.

---

# The Additive KLD Predictor

The first prediction layer is additive.

For each active group, MagicQuant finds the group’s effective quantization and looks up the measured isolated KLD for that group/quant pair.

Let:

```text
G = active tensor groups
c = candidate configuration
e_g(c) = effective quantization of group g in candidate c
D(g, q) = measured isolated KLD for group g using quant q
```

Then the additive prediction is:

```text
A(c) = Σ D(g, e_g(c))
       for every active group g
```

Expanded:

```text
A(c) =
    D(embeddings,  e_embeddings(c))
  + D(lm_head,     e_lm_head(c))
  + D(attn_q,      e_attn_q(c))
  + D(attn_kv,     e_attn_kv(c))
  + D(attn_output, e_attn_output(c))
  + D(ffn_up_gate, e_ffn_up_gate(c))
  + D(ffn_down,    e_ffn_down(c))
  + D(moe_experts, e_moe_experts(c))
  + D(moe_router,  e_moe_router(c))
```

Unused groups are ignored.

Native exact aliases contribute zero:

```text
D(g, native exact) = 0
```

Q8 contributes measured isolation KLD:

```text
D(g, Q8_0) = measured isolated Q8_0 KLD for group g
```

If required isolation data is missing, MagicQuant marks the candidate as unsafe or incomplete for prediction. It does not silently invent missing truth.

## Why additive prediction works

Additive prediction works because most tensor-group quantization damage is approximately independent enough to be useful.

Not perfectly independent.

Not magically independent.

Useful.

That is the key.

If `attn_q` causes a little damage and `ffn_down` causes a little damage, then using both often lands somewhere near the sum.

The additive layer gives MagicQuant a conservative rank prior:

```text
Based on measured isolated group behavior,
this candidate appears less damaged than that candidate.
```

It is the backbone of gravity.

## Why additive prediction is not final KLD

The additive number is not the final real KLD.

It is a prediction-space coordinate.

Two reasons:

1. Isolation samples happen in a controlled exact-blanket environment.
2. Full hybrids contain many simultaneously quantized groups that can interact.

So the additive KLD should be read as:

```text
relative expected damage from isolated measurements
```

Not:

```text
what llama-perplexity will exactly report later
```

The latter is validated by building and benchmarking the actual GGUF.

---

# Predicted Size

MagicQuant also predicts candidate size.

The size predictor starts from a base-only anchor and adds group-level size deltas.

Let:

```text
S_base(b) = measured base-only size for base quant b
S_iso(g, q) = measured single-group isolation size for group g at quant q
S_q8_exact = measured Q8-carrier exact-blanket size
```

Then:

```text
Ŝ(c) = S_base(base(c)) + Σ [S_iso(g, e_g(c)) - S_q8_exact]
```

For every active group `g`.

Native exact aliases contribute no size delta because the base-only anchor already holds active groups in exact precision.

If size anchors are missing, MagicQuant marks the size prediction unsafe.

This predicted size is used for candidate selection windows, strict dominance checks, local linear line checks, and ranking.

But again:

```text
Predicted size selects candidates.
Actual file size validates candidates.
```

The final survivor metadata uses real materialized size.

---

# Bit-Stress Interaction Fitting

The additive predictor is intentionally simple.

But quantization damage is not always purely additive. Low-bit choices can interact. Two groups that look individually manageable can become worse together.

MagicQuant adds an interaction correction using a bit-stress cross-term.

## Bit stress

Each quantization has an approximate bit range:

```text
Q8_0  => about 8-bit
Q6_K  => about 6-bit
Q5_K  => about 5-bit
Q4_*  => about 4-bit
IQ4_* => about 4-bit-ish
IQ3_* => about 3-bit-ish
IQ2_* => about 2-bit-ish
```

MagicQuant tries candidate stress thresholds:

```text
B ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12}
```

For each group:

```text
stress_B(g, c) = max(0, B - bits(e_g(c)))
```

If `B = 8`, then approximately:

```text
Q8_0  => stress 0
Q6_K  => stress 2
Q5_K  => stress 3
Q4_K  => stress 4
IQ3_* => stress 5
IQ2_* => stress 6
```

The lower the bit range, the higher the stress.

## Isolation bit-truth overrides

Declared bit range is the default, not the final word.

If isolated sampling proves that a lower-declared-bit candidate dominates a higher-declared-bit candidate for the same tensor group, MagicQuant allows the lower-bit candidate to inherit the higher bit range for **bit-stress purposes only**.

The test is intentionally simple:

```text
same group
candidate declared bits < victim declared bits
candidate.size <= victim.size
candidate.kld  < victim.kld - epsilon
```

When that happens:

```text
stress bits(group, candidate) = max(candidate declared bits, highest dominated victim bits)
```

So if `UD-IQ3_S` in `embeddings` beats a higher-fidelity `Q8_0` isolation state, the interaction model can treat that `embeddings + UD-IQ3_S` state as roughly 8-bit stress truth instead of naïve 3-bit stress truth.

This matters because some Unsloth Dynamic-style artifacts are not simple uniform low-bit shrinks. A file labeled around a 3-bit family may protect important tensors heavily enough that its actual isolated behavior is closer to a higher-fidelity state. In that case, applying normal low-bit stress would punish the candidate for a label rather than for measured local physics.

A real example shape from Qwen3.6 35B A3B looked like this:

```text
embeddings = UD-IQ3_S
```

The candidate was not just small and reckless. It isolated extremely well, with lower KLD than higher-declared-bit alternatives in that group. Under the old bit-stress logic, that state could be thrown into the predictive swamp because the interaction correction saw “3-bit” and inflated the expected low-bit damage. Under bit-truth inheritance, it earns the stress behavior of the strongest higher-fidelity state it actually defeated.

The boundary is important:

```text
additive KLD still uses measured isolation KLD
predicted size still uses measured size deltas
real benchmark truth still decides survival
only bit-stress interaction pressure is adjusted
```

So the rule is not:

```text
lower bit is secretly always higher bit
```

It is:

```text
when isolated truth proves a lower-bit state beat a higher-bit state,
do not let declared bit stress erase that truth before the candidate gets considered
```

## Pairwise cross-term

For each pair of active groups, MagicQuant computes:

```text
cross_B(g, h, c) =
    D(g, e_g(c))
  × D(h, e_h(c))
  × stress_B(g, c)
  × stress_B(h, c)
```

Then sums across pairs:

```text
X_B(c) = Σ cross_B(g, h, c)
         for every g < h
```

This term grows when multiple damaged low-bit groups appear together.

## Fitting alpha and beta

MagicQuant fits the interaction model against existing real benchmark rows in the current prediction bucket.

The prediction bucket is scoped by things like:

```text
architecture family
tensor group profile
model hash
imatrix identity
benchmark category
```

For each benchmarked row `i`, MagicQuant has:

```text
actual KLD y_i
additive prediction A_i
cross-term X_i
```

For each candidate threshold `B`, it fits:

```text
Y_i = αA_i + βX_i
```

Using a two-feature least-squares fit.

The normal equations are:

```text
s11 = Σ A_i²
s12 = Σ A_iX_i
s22 = Σ X_i²
y1  = Σ A_i y_i
y2  = Σ X_i y_i

det = s11s22 - s12²
```

If the determinant is usable:

```text
α = (y1s22 - y2s12) / det
β = (s11y2 - s12y1) / det
```

If the fit is degenerate, MagicQuant falls back to an additive-only fit.

To keep noisy early fits from going feral, coefficients are clamped:

```text
α ∈ [0.05, 10.0]
β ∈ [-1,000,000, 1,000,000]
```

For each threshold, MagicQuant computes mean absolute error:

```text
MAE_B = mean(|max(0, αA_i + βX_i) - y_i|)
```

The threshold with the best MAE wins.

If too few benchmark rows exist, MagicQuant uses fallback behavior:

```text
α = 1
β = 0
B = default threshold
```

So the interaction layer becomes additive-only until there is enough local benchmark truth to justify fitting.

## Final interaction estimate

For a candidate `c`, the raw interaction estimate is:

```text
Y(c) = max(0, αA(c) + βX_B(c))
```

This is usually more numerically realistic than plain additive prediction, especially in lower-bit territory.

But it is still not allowed to become the final rank authority by itself.

That is where rank-safe projection enters.

---

# Rank-Safe Projection with PAVA

The bit-stress model improves numerical realism, but it can become locally too confident.

MagicQuant therefore applies a rank-safe projection.

This is the reconciliation layer between:

```text
gravity from additive isolation evidence
```

and:

```text
fitted interaction correction from benchmark rows
```

## Sort order

MagicQuant sorts predictable candidates by:

```text
1. AdditiveKld ascending
2. InteractionKld ascending
3. PredictedSizeBytes ascending
```

The additive prediction is the rank spine.

That means if isolated measurements say candidate A should be safer than candidate B, the final prediction cannot casually invert them just because the interaction fit twitched.

## Projection problem

Let:

```text
A_i = additive prediction of candidate i
Y_i = interaction prediction of candidate i
Z_i = rank-safe projected prediction of candidate i
```

After sorting by additive prediction:

```text
A_1 <= A_2 <= A_3 <= ... <= A_n
```

MagicQuant finds the closest monotonic sequence `Z` to the interaction predictions `Y`:

```text
minimize Σ (Z_i - Y_i)^2
subject to Z_1 <= Z_2 <= ... <= Z_n
```

This is isotonic regression.

MagicQuant computes it with PAVA: the **pool adjacent violators algorithm**.

## What PAVA means in plain language

If the interaction model says:

```text
candidate 1: 0.010
candidate 2: 0.008
```

but additive gravity says candidate 1 should not be worse than candidate 2, that is a violation.

PAVA pools them into a plateau:

```text
candidate 1: 0.009
candidate 2: 0.009
```

The plateau is not a bug.

It is the prediction engine saying:

```text
I do not have enough stable evidence to separate these two cleanly.
```

That is exactly what you want.

The model avoids fake precision.

## Final rank-safe KLD

After projection:

```text
BaseRankSafeKld(c) = Z(c)
```

This becomes the normal gravity-respecting prediction before anomaly adjustments.

Candidate ranks are then assigned by:

```text
PredictedKld ascending
PredictedSizeBytes ascending
PredictionConfidence descending
stable slot tie-breakers
```

This gives MagicQuant a stable prediction order without pretending every tiny difference is gospel.

---

# Prediction Confidence

MagicQuant assigns confidence to prediction rows.

Confidence is not final truth. It is a ranking and diagnostics tool.

Non-hybrid/pure rows can be treated as fully confident because they are anchor-like and not speculative in the same way.

For hybrids, the confidence is reduced when:

1. The fit had too little data.
2. PAVA had to adjust the interaction estimate heavily.
3. The candidate lands inside a large plateau.

The core shape is:

```text
confidence = clamp(
    baseConfidence
  × 1 / (1 + adjustmentRatio)
  × plateauPenalty,
  0,
  1
)
```

Where:

```text
adjustmentRatio = |ProjectedKld - InteractionKld| / max(ProjectedKld, 1e-9)
```

And:

```text
plateauPenalty = max(0.25, 1 / sqrt(blockCount))
```

A large PAVA block means many candidates collapsed together. That is useful uncertainty information, so confidence drops.

The base confidence depends on fit maturity:

```text
fallback fit => lower base confidence
more fit rows => higher base confidence, capped at 1
```

Again, confidence does not decide final publication.

It helps order candidates before real validation.

---

# DuckDB Materialization

MagicQuant materializes prediction metadata into DuckDB.

SQLite remains the long-term truth store for real benchmark data. DuckDB is the transient high-throughput search table used for candidate generation, prediction, ranking, and selection.

Each candidate row can receive:

```text
PredictedKld
PredictedSizeBytes
PredictionConfidence
PredictionRank
BaseRankSafeKld
AnomalyAdjustmentKld
FinalPredictedKld
```

The materialization process does roughly this:

1. Build the prediction model.
2. Build lookup tables for group effective states.
3. Join effective group state to isolation KLD and size deltas.
4. Compute additive KLD.
5. Compute predicted size.
6. Compute bit-stress cross-term.
7. Compute interaction KLD.
8. Sort by additive/interactions/size.
9. Apply PAVA blocks.
10. Compute prediction confidence.
11. Persist predicted KLD, size, confidence, rank, and rank-safe base KLD back into DuckDB.

This is important because final selection can query pre-ranked candidates directly from DuckDB instead of loading huge candidate spaces into C# memory.

That is also why the system scales.

MagicQuant is not trying to carry the entire combinatoric universe in memory. It stores a compressed predictive ranking and asks DuckDB for the top candidates in specific frontier windows.

---

# Bad Trades and Search-Space Pruning

Before final prediction-guided selection, MagicQuant trims candidate choices that are unlikely to be worth exploring.

This pruning is one reason the later smart fallback exists. The main prediction space is intentionally kept clean enough to find the obvious nonlinear wins. If every subtle local inversion or tiny trade were allowed to flood DuckDB, the ranking space could become muddy and the best cheese could be harder to see. The fallback is the pressure valve: after the main space fails, MagicQuant can still inspect isolated truth directly and try a few conservative free-lunch candidates without polluting the primary search.

One major pruning concept is the **bad trade**.

A bad trade is a candidate that buys only a tiny size reduction while paying disproportionate KLD or PPL damage.

Suppose there is an accepted anchor `a` and a smaller candidate `r`.

MagicQuant first checks:

```text
r.size < a.size
```

Then computes size gain:

```text
sizeDeltaPercent = (a.size - r.size) / a.size × 100
```

If the size gain is too large, MagicQuant does not classify it as a bad trade, because it may represent a real size tier.

If the size gain is small enough, MagicQuant compares damage ratios:

```text
kldRatio = r.kld / a.kld
pplRatio = |r.pplDeltaPercent| / |a.pplDeltaPercent|
```

With default-style thresholds such as:

```text
max size delta percent = 4.0
KLD multiplier         = 2.5
PPL multiplier         = 3.5
```

A candidate can be removed if:

```text
small size gain
and disproportionate KLD damage
```

or:

```text
small size gain
and disproportionate PPL damage
```

There is an important escape hatch: if the candidate is meaningfully better in one metric while worse in the other, MagicQuant treats it as a mixed tradeoff rather than deleting it blindly.

So the spirit is not:

```text
remove everything smaller that looks worse
```

It is:

```text
do not waste search space on tiny size wins that are obviously overpaying in damage
```

## Why bad trades exist

Bad trades are not about scientific purity.

They are about physics, time, and usefulness.

MagicQuant is not omniscient. It cannot afford to validate every sub-neighborhood of every tiny trade.

Bad-trade pruning lowers the probability of wasting validation budget on candidates that are very unlikely to earn a final slot.

This is part of the bigger philosophy:

```text
practical frontier discovery > exhaustive worship of every possible combination
```

## Synergy-aware second chance after bad trades

Bad-trade pruning is necessary, but it is still local.

That means it can sometimes over-prune a quant that looks uncomfortable in one tensor group by itself but becomes useful when paired with a related group. The first hardcoded example is the feed-forward relationship:

```text
ffn_up_gate + ffn_down
```

These groups are still measured separately. The synergy registry does not merge them and does not create a fuzzy runtime heuristic. It only says:

```text
if a quant survived in one feed-forward peer,
and that same quant was removed from the other peer only by bad-trade pruning,
then it may deserve one KLD-only second chance
```

The second-chance pass runs after normal group bad-trade pruning has retained the original bad-trade decisions.

For each synergy group, MagicQuant checks:

```text
1. Was candidate q removed from group g by bad-trade pruning?
2. Does q still survive in a peer group from the same synergy registry entry?
3. Is q still banned for g?
4. Does q pass the same bad-trade review if PPL is ignored?
```

The fourth question is the key alteration.

Normal bad-trade pruning can eliminate on either disproportionate KLD damage or disproportionate PPL damage. During synergy second chance, PPL is deliberately blinded:

```text
normal review:
    small size gain + disproportionate KLD damage => remove
    small size gain + disproportionate PPL damage => remove

synergy second chance:
    small size gain + disproportionate KLD damage => keep removed
    small size gain + disproportionate PPL damage => do not remove by itself
```

This is a blessing-and-curse choice. PPL volatility can be a useful smoke signal, but in this specific post-pruning recovery pass it is allowed to be too conservative. The goal is not to trust the candidate. The goal is to allow a narrow slice of synergy-backed opportunity space to survive long enough for prediction and real benchmarking to judge it.

The pass does not rescue candidates removed by dominance, equivalent-truth pruning, final KLD cleanup, unsupported combinations, or hard damage rules. It only reviews retained bad-trade removals.

## Example: Qwen3.6 35B A3B feed-forward restoration

This architecture is a strong example because the feed-forward/expert space dominates so much of the model that small local pruning decisions can erase almost the entire hybrid opportunity surface.

In one run, `ffn_down` retained higher-bit feed-forward candidates such as `Q6_K` and `UD-Q6_K`, while `ffn_up_gate` had pruned them under normal bad-trade rules. The synergy pass restored the matching candidates into `ffn_up_gate` because they survived in `ffn_down` and passed KLD-only review against their original anchors:

| Restored into |       Quant | Original anchor | Why restored                                             |
| ------------- | ----------: | --------------: | -------------------------------------------------------- |
| `ffn_up_gate` |      `Q6_K` |    `UD-Q6_K_XL` | Survived in `ffn_down`; KLD-only bad-trade review passed |
| `ffn_up_gate` |   `UD-Q6_K` |    `UD-Q6_K_XL` | Survived in `ffn_down`; KLD-only bad-trade review passed |
| `ffn_up_gate` | `UD-Q5_K_M` |    `UD-Q5_K_XL` | Survived in `ffn_down`; KLD-only bad-trade review passed |
| `ffn_up_gate` | `UD-Q5_K_S` |    `UD-Q5_K_XL` | Survived in `ffn_down`; KLD-only bad-trade review passed |

The same run restored lower-bit choices into `ffn_down` because those choices survived in `ffn_up_gate`:

| Restored into |        Quant | Original anchor | Why restored                                                |
| ------------- | -----------: | --------------: | ----------------------------------------------------------- |
| `ffn_down`    |     `IQ4_XS` |        `Q5_K_S` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |
| `ffn_down`    |    `IQ3_XXS` |    `UD-IQ3_XXS` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |
| `ffn_down`    |      `IQ2_M` |    `UD-IQ3_XXS` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |
| `ffn_down`    |  `UD-IQ4_XS` |        `Q5_K_S` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |
| `ffn_down`    |  `UD-Q3_K_M` |        `Q5_K_S` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |
| `ffn_down`    | `UD-Q3_K_XL` |        `Q5_K_S` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |
| `ffn_down`    |  `UD-Q4_K_S` |        `Q5_K_S` | Survived in `ffn_up_gate`; KLD-only bad-trade review passed |

This is not MagicQuant getting softer everywhere.

It is MagicQuant saying:

```text
pruning is critical,
but known coupled groups deserve a narrow pressure-release valve,
because the best final hybrid may need both sides of that coupling available
```

Without this recovery, some MOE-heavy architectures may produce few or no hybrids even when meaningful hybrids are actually achievable. With too much recovery, the prediction space becomes fuzzy and noisy. The synergy mechanism is the balance point: explicit, rare, traceable, and still subordinate to real benchmark validation.

## Equivalent-truth pruning

MagicQuant can also collapse candidates that are effectively indistinguishable in measured output space.

If two choices produce equivalent measured truth, the safer or cleaner representative can survive while redundant variants are removed.

Again, this is not hiding information.

It is preventing meaningless clutter from polluting the final frontier.

---

# Candidate Selection: How a Hybrid Earns a Real Benchmark

After prediction materialization, MagicQuant does not simply build the top N candidates globally.

It asks frontier-aware questions.

The final chooser starts from current real anchors, then uses predicted rows to find candidates worth validating.

The major phases are:

```text
1. Strict dominance replacement
2. Near-baseline replacement
3. Interior subspace discovery
4. Smart baseline-tuning fallback when a normal phase fails
5. Best confirmed anomaly reconciliation
6. Meaningful spacing
7. Final dominance
```

## Phase 1: Strict dominance replacement

A predicted candidate is considered for strict dominance when it appears able to beat an existing anchor at the same or smaller size.

Prediction-space query:

```text
PredictedSizeBytes <= anchor.PredictedSizeBytes
PredictedKld       <  anchor.PredictedKld
```

Then MagicQuant builds and benchmarks the candidate.

Real validation requires:

```text
actualSize <= anchor.actualSize
actualKld  < anchor.actualKld - epsilon
```

If it passes, it can replace the anchor.

This is the cleanest kind of win.

## Phase 2: Near-baseline replacement

Sometimes a hybrid is slightly larger than a smaller anchor but improves KLD enough to justify the tiny size premium.

This phase allows a controlled size growth window.

The candidate must still earn the extra bytes.

This prevents MagicQuant from saying:

```text
This is bigger and barely better, therefore publish it.
```

Instead, the question is:

```text
Did the candidate improve enough for the size it added?
```

## Phase 3: Interior subspace discovery

This is where the local linear KLD line matters.

Suppose there are two validated anchors:

```text
smaller anchor:
    size = S_small
    KLD  = K_small
    higher damage

larger anchor:
    size = S_large
    KLD  = K_large
    lower damage
```

For a candidate between them:

```text
S_small <= S_candidate <= S_large
```

MagicQuant computes the straight-line expected KLD:

```text
t = (S_candidate - S_small) / (S_large - S_small)

lineKld = K_small + t × (K_large - K_small)
```

The candidate must beat the line:

```text
candidateKld < lineKld
```

Why?

Because any random candidate can land between two anchors. That does not make it useful.

A hybrid earns an interior slot only when it is better than the boring interpolation between known choices.

That is how MagicQuant finds nonlinear trade spaces instead of merely generating extra files.

## Fallback attempts

MagicQuant has two different fallback ideas, and they should not be confused.

The first is the normal prediction-guided fallback. If the best DuckDB-selected candidate fails real validation, MagicQuant may try a limited number of additional predicted candidates from that same phase window. These are still normal prediction-space candidates. They were selected from the materialized DuckDB ranking.

The second is the smart baseline-tuning fallback. This runs only after the normal prediction-guided path fails to validate a candidate for the strict, near-baseline, or interior window. It is not a broader DuckDB query. It is a conservative SQLite/isolation-truth routine that starts from the anchor's uniform baseline blanket and asks whether isolated measurements reveal a cheap group swap worth trying.

This matters because the prediction engine is intentionally practical, not omniscient.

A false positive is acceptable if it is cheap and filtered by validation.

A fallback can still discover the real winner nearby, but the fallback must obey the same real benchmark contract as the phase that invoked it.

## Stop after success

For normal strict-dominance behavior, once a candidate validates successfully for an anchor, MagicQuant does not need to keep validating every other top candidate for that same anchor.

The system is trying to discover the frontier, not benchmark a trophy parade.

There can be a special anomaly-oriented mode that validates more candidates for analysis, but the default practical discovery behavior is:

```text
validate until success
accept success
move on
```

## Meaningful spacing

After candidate acceptance, MagicQuant applies meaningful spacing.

If two survivors are too close in size, the weaker one can be collapsed away.

The minimum neighbor gap is based on a fraction of the global survivor size span:

```text
minGap = globalSizeSpan × minimumNeighborGapFraction
```

If two candidates are closer than `minGap`, MagicQuant chooses the spacing winner by:

```text
strict dominance if possible
otherwise lower KLD
otherwise smaller size
```

This prevents the final release from showing a dozen nearly identical files that do not meaningfully help users choose.

## Final dominance

Finally, MagicQuant runs real dominance again.

If any survivor is no longer justified because another survivor is no larger and has lower real KLD, it is removed.

The final list is therefore not “all interesting things MagicQuant found.”

It is the cleaned, validated, useful frontier.


---

# Smart Baseline-Tuning Fallback: Conservative Free-Lunch Search

The smart baseline-tuning fallback exists because the main prediction space is deliberately disciplined.

DuckDB prediction is excellent at finding strong nonlinear candidates, but that strength comes from keeping the searchable space clean. Bad-trade pruning and deterministic filters prevent noisy, subtle, or locally suspicious choices from flooding the ranking table. That is usually the right tradeoff.

But it creates a blind spot.

Some real wins are not giant prediction-space discoveries. Some are tiny MDA-backed improvements:

```text
a group is isolated-tested against the blanket baseline
it is same-size-or-smaller
it has lower isolated KLD
there is no fancy combo theory needed
```

That kind of result can be too subtle for the main prediction space, especially if allowing every such tiny path into DuckDB would muddy the ranking universe.

So MagicQuant separates the jobs:

```text
DuckDB prediction path:
    find the high-confidence frontier cheese

Smart fallback path:
    after prediction fails, try a few conservative isolation-backed freebies
```

This is not a second brute-force engine.

It is a last-resort, low-count, benchmark-validated candidate generator.

## When the smart fallback runs

The fallback is gated by configuration:

```yaml
candidate_selection:
  smart_fallback_enabled: true
  smart_fallback_attempts_per_failure: 3
  smart_fallback_max_higher_fidelity_steps: 2
```

It runs only after the normal phase fails.

For strict dominance, it can run per anchor when:

```text
no predicted virtual anchor exists
or no physically eligible predicted candidates remain
or the normal predicted candidates fail real validation
```

For near-baseline replacement, it can run per adjacent anchor pair when:

```text
predicted anchors are missing
or no predicted candidate survives deterministic filters
or normal predicted candidates fail real validation
```

For interior subspace discovery, it can run after the normal interior phase finds no accepted candidate.

The fallback does not replace the main prediction path. It waits behind it.

## What truth the fallback uses

The smart fallback does not query materialized DuckDB prediction rows.

Its candidate notes are explicitly shaped as:

```text
smartFallback=sqlite-isolation-truth; not selected from DuckDB prediction rows
```

Instead, it loads the local benchmark truth needed to reason from isolation samples:

```text
active tensor groups
pure baseline snapshots
Q8_0 pure benchmark
Q8_0 native-exact base-only anchor
base-only anchors for known baselines when available
single-group isolation snapshots by group and baseline
```

The fallback therefore works from the same measured local physics as the normal predictor, but it uses that truth differently.

The normal predictor says:

```text
rank the full candidate universe
```

The smart fallback says:

```text
start from this one failed anchor blanket
look for a tiny number of local swaps that isolated truth says are better
```

## Baseline blanket resolution

The fallback first resolves the failed anchor into a baseline blanket.

It can use a pure baseline or a uniform learned/external baseline. For example:

```text
base        = UD-Q4_K_XL
embeddings  = UD-Q4_K_XL
lm_head     = UD-Q4_K_XL
attn_q      = UD-Q4_K_XL
attn_kv     = UD-Q4_K_XL
attn_output = UD-Q4_K_XL
ffn_up_gate = UD-Q4_K_XL
ffn_down    = UD-Q4_K_XL
```

It skips anchors that are native/exact precision.

It also skips anchors that are already non-uniform hybrids:

```text
anchor is already a mixed hybrid => no smart blanket fallback
```

That guard matters. The fallback is not trying to explain an already-complex hybrid. It is asking whether a uniform baseline has obvious isolated weak spots that can be tuned safely.

## Building isolated group options

For each active group, the fallback compares the blanket baseline's isolated sample against every allowed explicit group candidate.

For a group `g`, blanket baseline `b`, and candidate baseline `q`:

```text
baseIsolation      = D(g, b)
candidateIsolation = D(g, q)

kldGain = baseIsolation.kld - candidateIsolation.kld
```

A group option is allowed only when:

```text
kldGain > minimumKldImprovementEpsilon
```

So the fallback does not try swaps merely because they are smaller.

A lower-fidelity candidate can be used, but only if isolated truth says it is measurably better than the blanket group state.

A higher-fidelity candidate can be used, but only up to the configured climb limit:

```text
smart_fallback_max_higher_fidelity_steps
```

This prevents the fallback from “winning” by simply walking everything upward to expensive high fidelity.

For each accepted option, the fallback computes a group-level size delta:

```text
baseContributionBytes      = baseIsolation.size - q8ExactBlanket.size
candidateContributionBytes = candidateIsolation.size - q8ExactBlanket.size
sizeDeltaBytes             = candidateContributionBytes - baseContributionBytes
```

Then it scores the option using isolated KLD gain, damage avoided per MiB, and sensitivity.

Same-size-or-smaller improvements receive a strong free-lunch bias, because they represent exactly the kind of subtle MDA win that should not require a giant prediction-space theory.

## Plan strategies

The fallback turns group options into a small set of candidate plans.

The current strategy families are:

```text
free-lunch-same-or-smaller
single-sensitive-group
balanced-brain-protection
sensitivity-first-blend
```

### free-lunch-same-or-smaller

This plan chooses the best same-size-or-smaller KLD-improving option per group.

It is the cleanest fallback pattern:

```text
spend no extra size
reduce isolated KLD
try the resulting blanket tweak
```

This is how a candidate can emerge from pure MDA logic rather than prediction-space cheese.

### single-sensitive-group

This plan tries one high-value group swap by itself.

It exists because one tensor group may be disproportionately damaging in isolation. Sometimes the best practical candidate is not a broad hybrid; it is simply:

```text
protect this one weirdly sensitive thing
leave the rest alone
```

### balanced-brain-protection

This plan is used for premium and interior fallback windows.

It starts with free lunches, then tries to spend available size budget on the most valuable positive-size protection options. It keeps the plan only if the predicted size stays inside the requested real size window.

This is the “ehh... I wonder if...” mode.

It does not claim certainty. It says:

```text
this protected set costs bytes,
but isolated truth says those bytes protect disproportionate damage,
and the phase window gives us room to try
```

### sensitivity-first-blend

This plan blends the free-lunch set with one high-sensitivity positive-size option.

It is another controlled gamble for premium and interior windows, especially when one group looks like it may deserve special protection.

## Strict fallback behavior

Strict smart fallback is the most conservative mode.

For strict dominance, group options are filtered to same-size-or-smaller swaps:

```text
option.sizeDeltaBytes <= 0
```

The final plan must also predict no larger than the strict anchor size:

```text
predictedSize <= anchor.size
```

Real validation then uses the same strict dominance contract as the normal phase:

```text
actualSize <= anchor.actualSize
actualKld  < anchor.actualKld - epsilon
```

If it passes, the fallback candidate can eliminate the anchor with a reason like:

```text
smart baseline-tuning strict dominance fallback:
real benchmark validated lower KLD at same-or-smaller size
```

If it fails, it dies.

No philosophical trophy. No special pleading. Reality bonks it with the newspaper.

## Near-baseline and interior fallback behavior

Near-baseline and interior smart fallbacks are allowed to gamble more than strict fallback, but only inside the phase's real size window.

For these modes, positive-size group protection is allowed when it fits:

```text
realMinSize <= predictedSize <= realMaxSize
```

The fallback scores plans by:

```text
isolated group option strength
additive KLD gain versus the blanket
predicted gain over the local line
how effectively the plan uses the available size budget
```

But score is only for ordering attempts.

The real benchmark still decides.

Near-baseline validation requires:

```text
actual size inside the near-baseline premium window
actual KLD beats the real local linear KLD line
```

Interior validation requires:

```text
actual size inside the interior window
actual KLD beats the real interior linear KLD line
```

That is why a fallback can discover a premium winner without pretending it knew the final nonlinear outcome in advance.

It is allowed to say:

```text
This looks like a good isolated-damage trade.
It fits the allowed size window.
Let's spend one of the few fallback attempts and see if reality agrees.
```

## Attempt limits and confidence

The smart fallback is intentionally small.

By default, it gets:

```text
smart_fallback_attempts_per_failure = 3
```

Plans are converted into synthetic `HybridSelectionCandidate` rows with:

```text
PredictionConfidence = 0.50
CandidateTheoryFamilyKey = smart-baseline-tuning
DiversityMode = sqlite-isolation-fallback
```

That lower confidence is honest.

The fallback is not saying:

```text
I predicted this whole global hybrid interaction perfectly.
```

It is saying:

```text
I found a locally measured trade that is cheap enough and plausible enough to validate.
```

## Why this does not muddy prediction space

This design protects the main engine.

If subtle local wins were all allowed into the primary DuckDB materialization path, MagicQuant could end up overvaluing tiny isolated inversions, weak bad-trade-adjacent movements, or noisy local improvements. That can bury the cleaner nonlinear candidates the main engine is excellent at finding.

The smart fallback avoids that by running after failure, in a narrow anchor/window context, with very few attempts, and with the same real validation contracts.

So the system gets both behaviors:

```text
aggressive prediction-space cheese when the signal is strong
conservative MDA/free-lunch tuning when the strong signal fails
```

## Example: Qwen3.6-27B MQ-IQ4_NL

A representative fallback result is a `UD-Q4_K_XL` blanket with a few isolated group swaps:

```json
{
  "embeddings": "IQ4_NL",
  "lm_head": "UD-Q4_K_XL",
  "attn_q": "IQ4_XS",
  "attn_kv": "Q5_K_S",
  "attn_output": "UD-Q4_K_XL",
  "ffn_up_gate": "UD-Q4_K_XL",
  "ffn_down": "UD-Q4_K_XL"
}
```

The important part is not that the full predictor discovered some grand combination.

It did not.

The fallback saw local truths:

```text
IQ4_NL on embeddings beat the Q4/UD-Q4 blanket in isolation at same-or-smaller size.
IQ4_XS on attn_q was better and smaller than IQ4_NL in that group.
The saved size gave enough room to protect attn_kv with Q5_K_S.
```

That is free lunch plus one budgeted protection move.

No broad search-space pollution.

No claim of omniscience.

Just measured local physics turned into a small real benchmark attempt.

## Example: Qwen3.6-27B MQ-IQ3_M_1

Another fallback result used a `UD-Q3_K_XL` blanket:

```json
{
  "embeddings": "IQ4_NL",
  "lm_head": "UD-Q3_K_XL",
  "attn_q": "UD-Q3_K_XL",
  "attn_kv": "Q5_K_S",
  "attn_output": "UD-Q3_K_XL",
  "ffn_up_gate": "UD-Q3_K_XL",
  "ffn_down": "UD-Q3_K_XL"
}
```

This kind of candidate may not present as an obvious strict dominance candidate. It may instead be a premium or interior bet:

```text
can we protect the most damaging group enough,
stay inside the allowed size budget,
and beat the real nonlinear trade line?
```

The fallback does not know with high confidence that the answer is yes.

It knows the isolated trade is plausible enough to spend one of the few guarded attempts.

That is the whole point.

## The fallback contract

The smart fallback may suggest candidates.

It does not publish candidates.

A fallback candidate survives only if the real benchmark satisfies the same phase contract that any normal predicted candidate must satisfy.

The fallback is therefore best understood as:

```text
small empirical bet generator
```

not:

```text
second final judge
```

The final judge is still real benchmark truth.

---

# Anomaly Detection: Respecting Violations of Gravity

Normal prediction is gravity-respecting.

But some architectures produce meaningful violations.

An anomaly is not just:

```text
lower-bit quant looked slightly better somewhere
```

An anomaly is a validated contextual pattern where a candidate violates the expected monotonic direction enough to matter.

The current system is designed to detect, probe, classify, and learn these violations without destroying the normal gravity prior.

## Why anomaly detection uses Q8 context

Normal isolation uses:

```text
base Q8 carrier
all active groups native exact
one group changed
```

That is excellent for local marginal truth.

But some emergent behavior only appears when the surrounding context is also quantized.

If everything else is BF16/native exact, certain Q8-vs-Q6 effects can be hidden or dampened. In fully quantized space, the interaction can surface.

So anomaly detection uses a different kind of reference:

```text
explicit quantized context twin
```

For a Q8 anomaly probe, the reference twin can be:

```text
base        = Q8_0
embeddings  = Q8_0
lm_head     = Q8_0
attn_q      = Q8_0
attn_kv     = Q8_0
attn_output = Q8_0
ffn_up_gate = Q8_0
ffn_down    = Q8_0
```

Then a monotone-downgrade candidate might be:

```text
base        = Q8_0
embeddings  = Q8_0
lm_head     = Q8_0
attn_q      = Q8_0
attn_kv     = Q8_0
attn_output = Q8_0
ffn_up_gate = Q8_0
ffn_down    = Q6_K
```

This is not a BF16 isolation sample.

It is a contextual twin comparison inside quantized space.

That distinction is critical.

Normal isolation asks:

```text
What does this group look like against native-exact surroundings?
```

Anomaly probing asks:

```text
Does this lower-fidelity movement behave differently in a real quantized context?
```

## Controlled fidelity regimes: testing whether truth transfers downward

Anomaly detection is powerful, but it is not omniscient.

Normal isolation and anomaly probes intentionally sample a limited number of high-signal contexts. Testing every low-bit blanket, pair, triple, and surrounding quantization condition would explode the sample count.

A useful example is a group where `IQ4_NL` looks unusually strong in isolation or in a 4-bit-and-above context. It may beat `Q4_K_M` by a meaningful amount when the surrounding groups are Q8/native exact or otherwise high fidelity. That does not guarantee the relationship survives in 3-bit-or-lower territory.

In a much lower-fidelity context, the surrounding damage field changes. The local ranking can reverse.

MagicQuant now tests this risk with bounded controlled context strata:

```text
high fidelity: Q6/Q5 reference blankets
mid fidelity:  Q4 reference blanket
low fidelity:  IQ3 reference blanket, opt-in
```

For exploratory rank checks, it compares:

```text
isolation winner
versus
closest-size same-bit non-equivalent alternative
```

Both candidates are measured inside the same blanket. This provides evidence about transfer without replaying an old final mixture or brute-forcing every pair.

The resulting rule is matched against the effective non-selected surrounding groups, not merely the candidate's carrier label. Evidence can be beneficial, harmful, or suppression-only. Weak or noisy changes can therefore block unsafe promotion without creating an optimistic boost.

This closes part of the earlier blind spot, but it does not turn a handful of blankets into a complete map:

```text
controlled regime probes improve transfer knowledge
they do not eliminate combinatorics
```

Low-fidelity probing remains opt-in because the extra real builds are most valuable when Q3-or-lower artifacts are an important release target.

## Movement classification

MagicQuant compares a candidate to its reference twin and classifies the movement.

Common movement classes include:

```text
NoMovement
MonotoneDowngrade
MonotoneUpgrade
MixedTrade
LateralOrProviderEquivalent
Unknown
```

The anomaly smoke system cares heavily about monotone downgrades:

```text
candidate is same or lower fidelity than twin
candidate has no compensating upgrades
candidate may save size
candidate may unexpectedly improve KLD
```

A monotone downgrade that wins is the interesting case because it violates gravity.

## Smoke detection

Anomaly smoke can come from two places:

```text
1. Historical real benchmark rows
2. DuckDB prediction-space rows
```

Historical smoke asks:

```text
Have we already benchmarked a candidate and its contextual twin?
Did the lower-bit candidate actually beat the higher-bit twin?
```

Prediction-space smoke asks:

```text
Does the predicted search space contain a monotone downgrade that appears suspicious enough to probe?
```

A smoke candidate is not yet a rule.

It is a hypothesis.

## Smoke filters

MagicQuant avoids probing every possible anomaly-looking row.

A smoke candidate must satisfy conditions such as:

```text
valid contextual quantized config
monotone downgrade movement
changed group count under configured max
predicted size savings over twin
prediction-space gap not catastrophically bad
not duplicate of existing rule/suppression
```

The goal is to spend a tiny number of probes on high-signal suspects.

## Probe planning

For each smoke seed, MagicQuant builds contextual probes.

If the candidate changed several groups, MagicQuant can test subsets:

```text
single-group probes
pair probes
full probe
composition probes
```

Example:

```text
reference twin:
    all active groups Q8_0

candidate changes:
    attn_q      Q8_0 -> Q6_K
    ffn_down    Q8_0 -> Q6_K
    ffn_up_gate Q8_0 -> Q5_K
```

MagicQuant can probe:

```text
attn_q only
ffn_down only
ffn_up_gate only
attn_q + ffn_down
attn_q + ffn_up_gate
ffn_down + ffn_up_gate
all three
```

But it does this within strict budgets.

The point is not to validate every possible truth.

The point is to find enough evidence to guide frontier discovery.

## Probe validation

Each probe is physically built and benchmarked against its contextual twin.

Let:

```text
K_ref   = real KLD of higher-bit twin
K_probe = real KLD of lower-bit probe
```

The actual gain is:

```text
gain = K_ref - K_probe
```

A beneficial anomaly requires:

```text
probe.size <= reference.size
gain >= minimum actual gain threshold
```

If true, the lower-bit candidate is smaller and better.

That is a real violation.

If the probe is meaningfully worse than the twin, MagicQuant can store harmful evidence.

If neither side is strong enough, the result is treated as normal gravity or suppression-only evidence.

## Probe classifications

A validated beneficial probe can become:

```text
SingleGroupInversion
PairSynergy
HigherOrderSynergy
CounterfactualMdaViolation
```

A harmful probe can become:

```text
HarmfulInteraction
HarmfulInterference
ContaminatingPassenger
```

A failed anomaly becomes:

```text
NormalGravity
SuppressionOnly
```

That is important: a failed anomaly is still useful.

It tells MagicQuant:

```text
The smoke was not real enough. Trust gravity here.
```

---

# How Anomaly Rules Adjust Prediction Space

Once probes are validated, MagicQuant can persist scoped anomaly rules.

A rule is scoped to the current truth bucket:

```text
architecture family
tensor group profile
model hash
imatrix identity
benchmark category
```

It is not a universal law.

A rule says something closer to:

```text
In this model/profile/imatrix context,
this group movement showed a confirmed counterfactual effect.
```

## Actual KLD does not directly replace prediction KLD

This is subtle.

Even when a real probe confirms a beneficial anomaly, MagicQuant does not simply subtract the real KLD gain from every matching prediction row.

Prediction KLD is rank-relative.

So anomaly adjustment is sized around how much the candidate must move in prediction space to be ordered correctly relative to its twin.

Let:

```text
baseGap = candidatePredictedKld - twinPredictedKld
margin  = prediction-space violation margin
```

For a beneficial anomaly, the required ordering movement is:

```text
required = -(max(0, baseGap) + margin)
```

For a harmful anomaly:

```text
required = max(margin, abs(baseGap) + margin)
```

Then MagicQuant applies confidence, shrink factors, context multipliers, and caps.

The actual probe result affects:

```text
classification
confidence
rule acceptance
metadata
```

But prediction-space movement remains prediction-space movement.

This avoids mixing coordinate systems incorrectly.

## Beneficial rules are pairwise, not universal blessings

Beneficial anomaly adjustment is intentionally conservative.

A beneficial rule is applied as **pairwise twin ordering**.

That means:

1. Find rows matching the candidate side of the rule.
2. Find their higher-fidelity twins in the same context.
3. Move the candidate below the twin by a small margin if needed.

Formula shape:

```text
newFinalKld = max(0, min(candidateCurrentFinalKld, twinEffectiveKld - margin))
```

Then:

```text
AnomalyAdjustmentKld = newFinalKld - BaseRankSafeKld
```

This is not saying:

```text
Q6_K is now universally better than Q8_0.
```

It is saying:

```text
Where this contextual twin relationship exists,
order the confirmed lower-bit anomaly beneath its higher-bit twin.
```

If the twin row is missing, MagicQuant does not apply a broad fallback boost.

That restraint matters.

## Harmful rules can demote more broadly

Harmful rules are safer to apply broadly because they are cautionary.

If a pattern was shown to be harmful, MagicQuant can increase predicted KLD for matching rows, subject to caps.

Shape:

```text
FinalPredictedKld = BaseRankSafeKld + bounded harmful adjustment
```

This helps avoid spending validation budget on patterns already shown to be contaminating.

## Suppression-only rules

Suppression-only evidence does not mutate prediction scores.

It records:

```text
we tested this smoke
it did not confirm a useful anomaly
normal gravity applies
```

That prevents repeated waste without pretending a new numeric law was learned.

## No global PAVA after anomaly exceptions

This is also important.

Normal gravity prediction is PAVA-projected into a rank-safe monotonic sequence.

Anomaly adjustment is then applied as scoped exception evidence.

MagicQuant does not rerun global PAVA after anomaly adjustments, because doing so would smear local exceptions back into the global gravity field.

The final prediction columns therefore mean:

```text
BaseRankSafeKld:
    normal gravity-respecting PAVA prediction

AnomalyAdjustmentKld:
    scoped exception adjustment

FinalPredictedKld:
    prediction used after anomaly adjustment
```

That separation keeps the system honest.

---

# The Qwen3.6-27B Q8 vs Q6 Example

The Qwen3.6-27B run is the cleanest example of why the new system matters.

The normal isolation evidence said gravity was still real.

For `ffn_down`, the isolated measurements showed Q8 better than Q6:

```text
ffn_down = Q8_0:
    isolated KLD ≈ 0.000680

ffn_down = Q6_K:
    isolated KLD ≈ 0.001247
```

In other words, normal isolation said:

```text
Q8_0 is safer than Q6_K for ffn_down.
```

That is exactly what gravity predicts.

But the full quantized context exposed a stronger pattern.

A candidate emerged:

```text
base        = Q8_0
embeddings  = Q8_0
lm_head     = Q8_0
attn_q      = Q8_0
attn_kv     = Q8_0
attn_output = Q8_0
ffn_up_gate = Q8_0
ffn_down    = Q6_K
```

This candidate became:

```text
MQ-Q6_K_1
```

It validated at approximately:

```text
size = 27.25 GB
KLD  = 0.002845
```

The pure llama.cpp Q8_0 anchor was approximately:

```text
size = 28.60 GB
KLD  = 0.003768
```

So the hybrid was:

```text
smaller than Q8_0
lower KLD than Q8_0
```

That is strict dominance.

This is not a normal tiny inversion.

This is a meaningful violation of the usual gravity expectation.

## What the example does not mean

It does not mean:

```text
Q6_K is universally better than Q8_0.
```

It does not mean:

```text
MDA is false.
```

It does not mean:

```text
Q8 is source of truth.
```

It means:

```text
In this architecture/context, lowering ffn_down from Q8_0 to Q6_K produced a validated beneficial anomaly.
```

The correct lesson is not to abandon gravity.

The correct lesson is:

```text
Gravity is the default.
Real validated anomaly patterns are exceptions.
Those exceptions should be learned, scoped, and exploited.
```

That is what the modern engine does.

---

# Repeatable Algorithm

This section describes the process as if rebuilding the system from scratch.

## Step 1: Define the prediction bucket

Scope all truth to:

```text
model hash
architecture family
tensor group profile
imatrix identity
benchmark category
```

Do not mix unrelated truth unless intentionally designing cross-model research.

## Step 2: Define active tensor groups

Start from known tensor group definitions.

Remove groups unused by the current architecture.

Example active group set:

```text
embeddings
attn_q
attn_kv
attn_output
ffn_up_gate
ffn_down
```

## Step 3: Define explicit tensor group synergies

Load the static synergy registry.

The current hardcoded synergy is:

```text
ffn_up_gate + ffn_down
```

Keep this registry small and explicit. Synergy is an architectural exception, not a general fuzzy rescue rule.

## Step 4: Build pure baselines

Benchmark pure baselines and external baselines:

```text
Q8_0
Q6_K
Q5_K
Q4_K_M
IQ4_XS
Unsloth Dynamic variants
other configured custom baselines
```

These become real anchors.

## Step 5: Build the Q8 exact-blanket base-only anchor

Create:

```text
base = Q8_0
all active groups = native exact
```

Benchmark it.

This is the exact-blanket reference for normal isolation.

## Step 6: Build single-group isolation samples

For every active group `g` and target quant `q`:

```text
base = Q8_0
all active groups = native exact
only group g = q
```

Benchmark and store:

```text
D(g, q)
S_iso(g, q)
PPL(g, q)
```

Do not treat Q8 as zero damage. Q8 has measured isolated truth.

## Step 7: Generate candidate space

Generate candidate tensor configurations from allowed base quants and group choices.

Apply runtime bans, unsupported combinations, unused groups, and search-space policy.

## Step 8: Prune bad trades

Within each group, remove candidates that offer tiny size savings but disproportionate damage.

Keep mixed tradeoffs where a candidate meaningfully improves one metric while worsening another.

Retain structured bad-trade removals, including:

```text
removed group
removed candidate
accepted anchor
original bad-trade reason
KLD/PPL/size values
```

That retained provenance is what makes synergy second chance possible.

## Step 9: Apply synergy second-chance review

For each static synergy group, compare survivor sets across peer groups.

If candidate `q` survived in one peer but was removed from another peer only by bad-trade pruning, review it again against the original anchor with PPL ignored:

```text
small size gain + disproportionate KLD damage => keep removed
small size gain + disproportionate PPL damage => allow restoration
```

If the KLD-only bad-trade check passes, unban the candidate for that group and record a synergy second-chance reinstatement.

Do not restore dominance removals, equivalent-truth collapses, final cleanup removals, unsupported choices, or candidates that still fail KLD-only review.

## Step 10: Compute effective quantization

For every candidate and active group:

```text
e_g(c) = override_g(c) if override exists
       = base(c) otherwise
```

Normalize external baseline IDs for isolation lookup when safe.

## Step 11: Compute additive KLD

```text
A(c) = Σ D(g, e_g(c))
```

If required isolation data is missing, mark candidate not safely predictable.

## Step 12: Compute predicted size

```text
Ŝ(c) = S_base(base(c)) + Σ [S_iso(g, e_g(c)) - S_q8_exact]
```

If size anchors are missing, mark candidate unsafe for size selection.

## Step 13: Build isolation bit-truth overrides

For each active group, compare isolated candidate states against higher-declared-bit states.

If a lower-declared-bit candidate is same-size-or-smaller and lower-KLD than a higher-declared-bit victim, record the highest defeated bit range:

```text
stressBits(g, q) = max(declaredBits(q), highestDominatedVictimBits)
```

This override is used only by the bit-stress interaction correction.

Additive KLD, size prediction, and final validation remain based on measured truth.

## Step 14: Fit interaction correction

Use existing real benchmark rows in the current bucket.

For thresholds `B`, compute:

```text
X_B(c) = Σ D_iD_jstress_i stress_j
```

Where:

```text
stress_i = max(0, B - stressBits(group_i, effectiveQuant_i))
```

Fit:

```text
Y(c) = αA(c) + βX_B(c)
```

Choose the threshold with lowest MAE.

Fallback to additive-only if too few rows exist.

## Step 15: Apply rank-safe projection

Sort predictable candidates by:

```text
AdditiveKld
InteractionKld
PredictedSizeBytes
```

Apply PAVA to interaction KLD values.

Store:

```text
BaseRankSafeKld = projected KLD
```

## Step 16: Materialize predictions into DuckDB

Persist:

```text
PredictedKld
PredictedSizeBytes
PredictionConfidence
PredictionRank
BaseRankSafeKld
FinalPredictedKld
```

At this point, `FinalPredictedKld` equals `BaseRankSafeKld` unless anomaly rules are applied.

## Step 17: Detect anomaly smoke

Look for monotone downgrades versus higher-fidelity contextual twins.

Use both:

```text
historical benchmark smoke
DuckDB prediction-space smoke
```

Reject invalid sparse/native-exact anomaly configs.

Require quantized contextual twins.

Also identify justified context rank pairs: the isolation winner and the closest-size same-bit non-equivalent alternative for the selected group.

Remember the scope limit: anomaly smoke and controlled strata are not an exhaustive low-bit context map.

## Step 18: Probe anomaly candidates

For smoke candidates, build scoped probes:

```text
single-group
pair
composition
full changed set
```

Benchmark probe and reference twin.

For transfer and exploratory rank probes, rebuild both sides inside the same configured high-, mid-, or opt-in low-fidelity blanket.

Classify as beneficial, harmful, normal gravity, or suppression-only.

## Step 19: Persist anomaly rules

Create scoped rules from validated probe evidence.

Beneficial rules reorder candidates beneath matching twins.

Harmful rules demote matching rows.

Suppression-only rules prevent repeated wasted probing.

Scope transfer rules to the measured effective surrounding-group context, excluding the groups changed by the rule itself. Do not use the carrier label as a substitute for the actual context.

## Step 20: Query frontier candidates

Use DuckDB to query candidates for:

```text
strict dominance
near-baseline replacement
interior better-than-linear discovery
```

Use prediction-space windows to select candidates.

## Step 21: Build and benchmark normal prediction-guided candidates

Prediction gets the candidate into the arena.

Reality decides.

## Step 22: Run smart baseline-tuning fallback after phase failure

If the normal prediction-guided path fails to validate a winner for a strict, near-baseline, or interior window, optionally run the smart fallback.

Start from the failed anchor's pure/uniform baseline blanket.

Load SQLite/isolation truth:

```text
base-only anchors
single-group isolation samples
active group profile
```

Build guarded plans from isolated group improvements:

```text
free-lunch-same-or-smaller
single-sensitive-group
balanced-brain-protection
sensitivity-first-blend
```

Limit attempts by:

```text
smart_fallback_attempts_per_failure
```

Reject plans that do not fit the phase's real size window.

Then build and benchmark the selected fallback candidates.

## Step 23: Accept or reject

A candidate survives only if the real benchmark satisfies the phase contract.

Strict dominance:

```text
actualSize <= anchorSize
actualKld  < anchorKld
```

Interior:

```text
inside real size window
actualKld < real local linear line
```

Near-baseline:

```text
inside allowed size premium
actual KLD improvement justified by local tradeoff
```

## Step 24: Clean final frontier

Apply:

```text
real dominance
meaningful spacing
final dominance
```

Export survivors and manifests.

---

# What MagicQuant Claims and What It Does Not Claim

## MagicQuant claims

MagicQuant claims:

```text
The prediction engine is good enough to identify candidates worth validating.
```

MagicQuant claims:

```text
Real benchmark truth owns final publication.
```

MagicQuant claims:

```text
Most quantization behavior follows gravity, but meaningful contextual violations exist.
```

MagicQuant claims:

```text
Validated anomalies should be exploited without pretending they are universal laws.
```

MagicQuant claims:

```text
If the main prediction path finds no validated winner,
a small isolation-truth fallback can still test plausible free lunches
without polluting the main prediction space.
```

MagicQuant claims:

```text
The final frontier should be useful, not cluttered with redundant micro-variants.
```

## MagicQuant does not claim

MagicQuant does not claim:

```text
Predicted KLD is exact final KLD.
```

It does not claim:

```text
MDA is a perfect law.
```

It does not claim:

```text
Every local inversion deserves exhaustive search.
```

It does not claim:

```text
High-bit isolation truth always transfers perfectly into low-bit contexts.
```

It does not claim:

```text
Smart fallback candidates are high-confidence global predictions.
```

It does not claim:

```text
Hybrids always beat baselines.
```

It does not claim:

```text
Q8 is the source of truth.
```

It does not claim:

```text
A Q6-over-Q8 anomaly means Q6 is universally better than Q8.
```

The actual claim is more useful:

```text
MagicQuant builds a practical, empirically grounded prediction space,
uses it to spend benchmark compute intelligently,
learns real contextual exceptions,
and publishes only candidates that survive real validation.
```

---

# Reimplementation Checklist

If someone wanted to rebuild the core idea without copying code, they would need the following pieces.

## Data model

Track:

```text
architecture family
model hash
tensor group profile
imatrix identity
benchmark category
baseline definitions
tensor configs
real benchmark results
learned isolation samples
anomaly probe sessions
anomaly rules
bad-trade pruning diagnostics
synergy second-chance reinstatement diagnostics
```

## Tensor config system

Represent:

```text
BaseQuant
Embeddings
LmHead
AttnQ
AttnKV
AttnOutput
FfnUpGate
FfnDown
MoeExperts
MoeRouter
```

Each group slot should support:

```text
null/inherit base
explicit baseline ID
native exact alias
unused group
```

## Effective quant resolver

Implement:

```text
e_g(c) = override if present else base
```

Ignore unused groups.

Normalize external baselines for prediction lookup when possible.

## Isolation planner

Build:

```text
Q8 exact-blanket base-only anchor
single-group isolation samples for each active group/quant
archival coverage for groups pruned out of current search
```

## Synergy registry and bad-trade provenance

Implement a static tensor-group synergy registry.

At minimum, the current known synergy is:

```text
ffn_up_gate + ffn_down
```

During bad-trade pruning, retain structured records of bad-trade removals so a later synergy pass can review only those removals.

The second-chance pass should:

```text
look for the same quant surviving in a peer synergy group
reuse the original accepted anchor
ignore PPL as an elimination signal
keep KLD bad-trade rejection active
restore only candidates removed by bad-trade pruning
```

## Additive predictor

Compute:

```text
A(c) = Σ D(g, e_g(c))
```

Use measured Q8 KLD for Q8.

Use zero only for native exact aliases.

## Size predictor

Compute:

```text
Ŝ(c) = S_base(base(c)) + Σ [S_iso(g, e_g(c)) - S_q8_exact]
```

## Interaction model

Before computing stress, derive group/baseline stress bit ranges.

Declared bit range is the default. If isolated sampling shows a lower-declared-bit candidate is same-size-or-smaller and lower-KLD than a higher-declared-bit candidate in the same group, let it inherit the highest defeated bit range for interaction stress only.

For benchmarked rows:

```text
X_B(c) = Σ D_iD_jstress_i stress_j
Y(c) = αA(c) + βX_B(c)
```

Fit alpha/beta per threshold and choose lowest MAE.

Fallback to additive-only when too little truth exists.

## Rank-safe projection

Sort by:

```text
A(c), Y(c), Ŝ(c)
```

Apply PAVA:

```text
minimize Σ(Z_i - Y_i)^2
subject to monotonic Z in additive order
```

Store:

```text
BaseRankSafeKld = Z(c)
```

## DuckDB ranking

Materialize:

```text
PredictedKld
PredictedSizeBytes
PredictionConfidence
PredictionRank
BaseRankSafeKld
FinalPredictedKld
```

Query directly from DuckDB for frontier windows.

## Bad-trade pruning

Remove small-size-gain candidates with disproportionate KLD/PPL damage.

Keep mixed tradeoffs.

## Candidate selection

Implement:

```text
strict dominance
near-baseline replacement
interior better-than-linear discovery
normal DuckDB fallback attempts
real validation
```

## Smart baseline-tuning fallback

Implement a separate post-failure fallback path:

```text
enabled only after normal phase failure
starts from pure/uniform baseline blankets
loads SQLite/base-only/isolation truth
builds group options only from isolated KLD improvements
limits higher-fidelity climbs
creates free-lunch and protected-budget plans
uses a small attempt limit
marks candidates as sqlite-isolation-fallback
validates with the same real phase contract
```

Required strategies:

```text
free-lunch-same-or-smaller
single-sensitive-group
balanced-brain-protection
sensitivity-first-blend
```

Do not let this fallback contaminate the primary DuckDB ranking space.

## Anomaly detection

Implement contextual quantized twin comparisons:

```text
reference = all active groups at Q8/Q6/etc. context
candidate = monotone downgrade from reference
```

Reject BF16/native-exact anomaly configs.

Probe subsets of changed groups.

Add bounded controlled context strata when transfer behavior matters:

```text
high-fidelity Q6/Q5 blankets
mid-fidelity Q4 blanket
opt-in low-fidelity IQ3 blanket
```

For exploratory rank pairs, compare the isolation winner with the closest-size same-bit non-equivalent alternative inside the same blanket.

Classify:

```text
beneficial
harmful
normal gravity
suppression-only
```

Match learned rules against effective non-selected surrounding groups. Do not replay historical final winners and do not treat a search-row carrier label as sufficient context identity.

## Anomaly prediction adjustment

Keep normal prediction and anomaly exceptions separate:

```text
BaseRankSafeKld      = gravity prediction
AnomalyAdjustmentKld = scoped exception adjustment
FinalPredictedKld    = adjusted prediction
```

Beneficial rules should be pairwise twin ordering, not universal boosts.

Harmful rules can demote matching patterns.

## Final release cleanup

Apply:

```text
real dominance
meaningful spacing
final dominance
export manifests
clone configs
hybrid maps
replacement logs
bad-trade logs
```

---

# Final Summary

MagicQuant works because it refuses two bad extremes.

It does not brute-force the universe.

It also does not blindly trust a simplistic predictor.

Instead, it builds a practical prediction space from isolated tensor-group measurements, stabilizes that space with monotonic rank projection, uses fitted interaction correction where enough truth exists, keeps subtle bad-trade-adjacent ideas out of the main ranking space, recovers a few of them through guarded smart fallback when prediction fails, detects contextual violations of gravity through quantized twins and controlled fidelity strata, and validates every serious candidate with real benchmarks.

The system’s deepest idea is not merely:

```text
hybrid quantization can beat baselines
```

The deeper idea is:

```text
hybrid quantization search can be made practical by separating prediction from truth.
```

Prediction space finds the big doors.

Smart fallback checks whether a tiny side door was hiding in the isolated measurements.

Benchmarks decide which doors actually open.

That is the engine.
