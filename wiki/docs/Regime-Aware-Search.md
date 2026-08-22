# Regime-Aware Tensor Search

Isolated tensor-group testing is one of MagicQuant's most useful tools.

It answers questions such as:

```text
If every other group is held constant,
what happens when attn_q changes from Q4_K_M to IQ4_NL?
```

But there is a subtle problem:

> **A group choice that wins inside a high-fidelity surrounding model may not remain the winner after the surrounding model becomes much more compressed.**

MagicQuant therefore distinguishes isolation truth from regime transfer truth.

---

## The Rank-Flip Problem

Imagine an embeddings experiment inside a Q4-or-better blanket:

```text
surrounding groups: Q4 and above

embeddings IQ4_NL  => excellent
embeddings Q4_K_M  => worse
```

It is tempting to promote `IQ4_NL` everywhere.

Now repeat the same head-to-head comparison while the surrounding groups are Q3 or lower:

```text
surrounding groups: Q3 and below

embeddings IQ4_NL  => worse
embeddings Q4_K_M  => better
```

The tensor group did not change. Its context did.

That does not mean either measurement is wrong. It means the local ranking is conditional:

```text
best choice for group g
    may depend on
effective surrounding fidelity
```

This is especially important near aggressive compression regimes, where error from other groups is no longer small relative to the isolated change being measured.

---

## Why Normal Isolation Cannot See It

A standard isolation probe changes one group while keeping a single reference blanket fixed.

Conceptually:

```text
reference blanket = Q8

Q8 model + candidate embeddings
Q8 model + candidate attn_q
Q8 model + candidate ffn_down
```

This is excellent for learning a clean marginal signal. It reduces interference from unrelated compressed groups.

But it cannot answer:

```text
does the same marginal ranking survive in a Q4 blanket?
does it survive in an IQ3 blanket?
```

The controlled answer is not to abandon isolation. It is to repeat a small number of justified head-to-head probes inside deliberately chosen surrounding-fidelity strata.

---

## Controlled Context Strata

MagicQuant can define reference blankets at several fidelity levels:

```yaml
synergy_detection:
  transfer_probe_context_strata:
    high_fidelity_reference_quants: [Q6_K, Q5_K]
    mid_fidelity_reference_quants: [Q4_K_M]
    low_fidelity_reference_quants: [IQ3_S]
    low_fidelity_enabled: false
```

The default categories are conceptual rather than universal laws:

| Stratum | Purpose |
| --- | --- |
| High fidelity | Check behavior near relatively clean surrounding groups |
| Mid fidelity | Check the common Q4 neighborhood where many release candidates live |
| Low fidelity | Check whether behavior transfers or flips under aggressive compression |

Low-fidelity probing is opt-in because it is not free.

It adds real builds and benchmarks. It can also generate misleading excitement if a noise-scale difference is treated as a universal winner.

The purpose of a low-fidelity probe is not:

```text
make every Q3-or-lower candidate more important
```

It is:

```text
detect when evidence learned in a cleaner regime stops transferring
```

---

## Context Rank Pairs

MagicQuant does not need to retest every quant type against every other quant type in every blanket.

That would recreate the combinatorial explosion the prediction system exists to avoid.

Instead, exploratory context probing chooses a disciplined pair:

```text
candidate A = isolation winner
candidate B = closest-size, same-bit, non-equivalent alternative
```

Then both are remeasured inside the same controlled blanket.

Why the closest-size alternative?

Because a head-to-head comparison is most informative when it is not secretly comparing a large size jump.

Why same-bit?

Because the question is whether tensor behavior or quant-family behavior changes with context, not whether four bits usually beat three bits.

Why non-equivalent?

Because spending two real benchmarks on recipes that produce effectively identical tensor truth teaches nothing.

The relevant controls are:

```yaml
synergy_detection:
  exploratory_context_pair_enabled: true
  max_exploratory_context_pairs_per_run: 14
  exploratory_pair_bit_ranges: [4]
  exploratory_pair_context_strata: [mid-fidelity, low-fidelity]
```

The budget keeps this as targeted scientific measurement rather than an accidental exhaustive sweep.

---

## Historical Sources: Vocabulary, Not Winner Replay

Historical quant repositories can be valuable because they expose tensor configurations that do not exist in the newest source.

Suppose an earlier external revision used:

```text
quant family A for embeddings
quant family B for ffn_down
an unusual protection pattern for attn_output
```

A later revision may no longer publish all of those choices.

MagicQuant can digest both revisions to preserve the available tensor vocabulary.

It must not conclude:

```text
this old mixed model was a winner before
therefore replay the old mixture now
```

That would import historical frontier bias into a new model and a new measurement campaign.

The safer abstraction is:

```text
historical source
    => additional independently testable tensor recipes
    != historical final winner replay
```

Current isolation, context probing, prediction, and real validation still decide whether any recipe belongs in a current hybrid.

This distinction is the heart of historical learning in MagicQuant:

> **Learn what configurations exist. Relearn what they do.**

---

## Pin the Source Revision

Labels such as “Dynamic v2” or “Dynamic v3” are not precise enough for reproducible tensor digestion.

A repository can change while keeping the same human-facing name. Files can be replaced. Manifests can change. A rerun months later can silently learn a different vocabulary.

For every external tensor source, record the exact revision:

```text
repository identity
revision or commit hash
artifact filename
artifact checksum where available
```

During the Qwen3.8 27B campaign, the historical sources were pinned to exact revisions rather than read from a moving branch:

| Source generation | Pinned revision |
| --- | --- |
| Dynamic v2 source | `313447f257f7ebde0b968e4778feef774546ed81` |
| Dynamic v3 source | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |

Those hashes document what was digested. They do not endorse one generation as globally better than the other.

---

## Rules Must Match Effective Context

A context result is only useful if it is applied to compatible contexts.

The wrong shortcut is:

```text
the search row began from a Q4 carrier
therefore call its context Q4
```

A hybrid may replace several groups. The carrier label can stop describing the effective surrounding model.

MagicQuant therefore evaluates context using the groups that remain around the rule-selected groups:

```text
effective context = non-selected surrounding groups
```

Rule-selected groups are excluded from the match because they are the intervention, not the environment.

The application controls include:

```yaml
synergy_detection:
  context_scoped_rule_application_enabled: true
  max_non_rule_group_context_mismatches: 1
```

A small mismatch allowance can tolerate a nearly equivalent surrounding recipe. A broad carrier-based match would transfer evidence far beyond what was actually measured.

---

## Beneficial, Harmful, and Suppression Evidence

Context evidence is not only a source of bonuses.

MagicQuant records three important outcomes:

### Beneficial evidence

The candidate improved more than expected in the measured context.

This can justify a conservative, context-scoped prediction adjustment or a transfer probe.

### Harmful evidence

The candidate became worse in the measured context.

This should block optimistic transfer into that regime.

### Suppression-only evidence

The apparent difference exists, but it is too small, noisy, or inconsistent to support promotion.

This is still useful. It prevents an attractive high-fidelity result from being overgeneralized.

The asymmetry is intentional:

```text
weak positive evidence should not create a broad boost
harmful or uncertain evidence can still prevent unsafe transfer
```

This is how MagicQuant uses historical and contextual knowledge without allowing it to bias the whole search.

---

## Synergy Is Not Assumed

Two individually interesting context rules do not automatically compose into a stronger multi-group rule.

MagicQuant can run bounded composition probes, but the combined result must earn support from measurement.

During the Qwen3.8 27B campaign, composition probes did not justify a general synergy boost.

That is a valid result.

The system learned:

```text
some effects are context-specific
some are suppressive
the measured evidence does not support broad composition
```

Not finding synergy is better than inventing it.

Across the active campaign history, the persisted rule evidence was:

| Rule evidence | Count |
| --- | ---: |
| Confirmed beneficial | 12 |
| Confirmed harmful | 95 |
| Exact-context suppression | 117 |

The imbalance is informative. Most contextual knowledge was useful as a boundary on transfer, not as permission to improve predicted KLD broadly.

---

## Qwen3.8 27B: What the Probes Actually Found

The controlled campaign demonstrated why both promotion and restraint matter.

### Embeddings

`IQ4_XS` and `UD-IQ4_XS` changed rank in the IQ3 context, but the KLD difference was only `0.000031` while the size difference was about `0.157 GiB`.

That was treated as noise-scale suppression evidence, not proof of a universal low-fidelity winner.

### Attention query

For `attn_q`, `IQ4_NL` remained ahead of `Q4_K_M` in both the Q4 and IQ3 controlled blankets on this model.

The general rank-flip concern was valid, but the exact example did not reproduce for this group and model.

### Attention output

`Q4_K_M` and `UD-Q4_K_XL` improved the Q4 blanket but worsened the IQ3 blanket.

This is direct evidence that a useful mid-fidelity effect should not be transferred downward blindly.

### Feed-forward down projection

`UD-Q4_K_XL` was smaller but worse in the Q4 context, then became a strict win in the IQ3 context.

This is the kind of reversal that a single Q8 isolation blanket cannot reveal.

### Frontier outcome

One context-derived candidate reached:

```text
size: 16.241945 GiB
KLD: 0.011087
```

It was nondominated, but it did not dominate the already published point at:

```text
size: 16.414560 GiB
KLD: 0.007412
```

That distinction matters. The contextual system found valid evidence and a real frontier point. It did not justify claiming that every new probe improves the best release choice.

---

## Deciding Whether Low-Fidelity Probing Is Worth It

Enable low-fidelity context work when:

- Q3-or-lower artifacts are an important release target
- high-fidelity isolation winners are being transferred into much lower regimes
- a model architecture shows unusually strong cross-group interactions
- external sources expose materially different same-bit tensor recipes
- the run budget can afford real controlled comparisons

Leave it disabled when:

- the release is focused on Q4 and above
- there is no credible alternative pair to compare
- the expected effect is below benchmark noise
- the extra builds would displace more valuable frontier validation

The goal is not maximum probing.

The goal is maximum useful information per real build.

---

## What Regime-Aware Search Does Not Claim

It does not claim:

- that Q3 and lower always reverse Q4 behavior
- that historical revisions should vote on current winners
- that one controlled blanket represents every hybrid at that bit range
- that every measured rank flip deserves a prediction boost
- that context rules remove the need for final real benchmarking

The controlled probes improve the prediction system's understanding of where evidence transfers.

The final contract remains unchanged:

> **Prediction spends the benchmark budget. Real size and real KLD decide the survivor.**
