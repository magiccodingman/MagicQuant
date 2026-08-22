# Pareto Archives, Release Curation, and Reproducibility

MagicQuant produces two kinds of truth that should not be confused:

```text
measured evidence archive
and
curated release menu
```

The archive answers:

> What measured candidates are nondominated across everything compared?

The release menu answers:

> Which of those candidates are useful enough and sufficiently separated to present to users?

Both are valuable. They serve different jobs.

---

## Strict Pareto Dominance

Candidate A strictly dominates candidate B when:

```text
A.size <= B.size
A.KLD  <  B.KLD
```

with the expected measurement epsilon applied by the implementation.

If A is smaller or equal in size and has lower KLD, B no longer represents a rational size/fidelity choice in that comparison space.

The full nondominated union is the scientific frontier.

It should be preserved even when several points are very close together.

---

## Why the Release Can Be Smaller Than the Archive

A release containing dozens of models separated by tiny size differences can be technically complete and practically unhelpful.

MagicQuant therefore applies meaningful spacing to produce a cleaner survivor menu.

That curation is useful, but it is not the same as dominance.

```text
dominance:
    this point is objectively unnecessary under measured size/KLD

spacing:
    this point may be valid, but it is too close to another point for the release menu
```

A spacing-removed candidate should not disappear from the evidence record.

The recommended model is:

```text
all measured candidates
        |
        v
strict nondominated archive
        |
        v
curated, meaningfully spaced release view
```

---

## The Limitation of Global-Span Spacing

The current spacing rule derives a minimum neighbor gap from a fraction of the global size span:

```text
minGap = (largest survivor size - smallest survivor size)
         * minimumNeighborGapFraction
```

This is simple and predictable.

It also has a limitation.

When the full frontier spans from very small low-bit models to very large high-fidelity models, a global percentage can become large enough to hide useful distinctions in a dense high-quality region.

For example:

```text
full frontier span: very large
high-quality neighborhood: several meaningful choices within a narrow size band
global 3% spacing: collapses most of that neighborhood
```

That does not mean the removed points were dominated. It means the release policy chose a sparse menu.

Future curation can improve this with quality-aware or local-density spacing. Until then, the full nondominated archive is essential for honest comparison and later recuration.

---

## Cross-Run Comparison Requires Source Identity

SQLite numeric IDs are local database identifiers.

They are not stable artifact identities across runs.

This is dangerous when comparing an existing release to a new campaign. Two different source generations can reuse the same local tensor-combination or baseline ID.

The wrong merge rule is:

```text
same numeric ID => same measured artifact
```

The safer identity includes:

```text
source repository and revision
artifact filename or recipe
tensor-group assignment
model and imatrix scope
measured size and KLD
```

Rows should be collapsed as duplicates only when their source identity or reconstructed recipe and measurements agree.

---

## Qwen3.8 27B Frontier Audit

The Qwen3.8 27B campaign compared the existing published frontier with a new regime-aware run.

The corrected full-Pareto run reused 43 of 43 valid startup artifacts, retained 33 measured survivors, and contained 20 local GGUFs plus 13 upstream references. Its search evidence included 212 isolation samples and 95 recorded bad trades. Reuse shortened the campaign, but every reused object still had to match its recorded identity and scope.

The source-aware union contained:

| Evidence | Count |
| --- | ---: |
| Published candidates | 23 |
| Challenger candidates | 33 |
| Numeric-key overlaps | 12 |
| Exact measured overlaps | 11 |
| Distinct measured artifacts | 45 |
| Nondominated measured artifacts | 41 |

The difference between 12 numeric-key overlaps and 11 exact overlaps exposed a real identity collision: local key `100` referred to different `IQ2_XXS` measurements across source generations.

Treating numeric IDs as global identities would have silently erased evidence.

Of the 41 nondominated artifacts:

- 20 came from the published frontier
- 21 came from the challenger run
- 3 published points were strictly dominated by challenger points
- 1 challenger point was strictly dominated by a published point

The new run produced three strict measured replacements:

| Previous size / KLD | New size / KLD |
| --- | --- |
| 22.564615 GiB / 0.001439 | 22.537283 GiB / 0.001364 |
| 15.365976 GiB / 0.011351 | 15.362543 GiB / 0.011154 |
| 14.335722 GiB / 0.014502 | 14.332670 GiB / 0.014381 |

This proves improvement without pretending the entire old frontier became obsolete.

When the existing 3% global-span spacing policy was applied, only 17 of the 41 nondominated measurements remained in the curated view.

That is why both layers should be reported:

```text
41 = complete nondominated evidence
17 = one useful release curation under the current spacing policy
```

---

## What a Reproducible Campaign Must Record

A final README table is not enough to reproduce a search campaign.

The campaign record should preserve:

### Model scope

- exact base model repository
- exact model revision
- source file checksums where practical
- architecture and tensor-group mapping used

### Tensor configuration sources

- repository identity for every external source
- exact revision for every source generation
- artifact filenames and checksums where practical
- learned tensor recipes, not only provider labels

### Measurement scope

- imatrix file identity and checksum
- KLD/PPL corpus identity
- bucket or benchmark configuration
- `llama.cpp` binary fingerprint
- hardware topology and usable memory limits

### Search scope

- complete YAML configuration
- prediction and pruning thresholds
- context strata and probe budgets
- random seeds where applicable
- code commit or PR revision
- explicit runtime root used by the process

### Evidence

- SQLite campaign database or export
- manifests
- logs
- isolation measurements
- harmful, beneficial, and suppression context evidence
- every final artifact's real size and real KLD
- the unspaced nondominated union
- the curated release decision

This allows a future reader to separate:

```text
what was measured
what was inferred
what was removed by dominance
what was removed only by presentation policy
```

An isolated campaign root is especially useful when several runs share the same machine:

```text
--magic-quant-root /path/to/isolated/runtime
```

That root determines where campaign state such as SQLite data is resolved. Record it and verify the live process is using it before a long run; opening a different historical database can make cache reuse look valid when it belongs to another campaign.

---

## Cached Work and Reproducibility

Reusing previously downloaded baselines, learned tensor digestion, imatrix artifacts, or hardware plans can save enormous time.

Reuse is valid when its scope is verified.

The cache key must prove that the reused object belongs to the same relevant inputs. A convenient directory name is not enough.

For example:

```text
hardware plan reuse requires matching hardware and llama binary
tensor vocabulary reuse requires matching source revision
benchmark truth reuse requires matching model, artifact, corpus, and measurement setup
```

If the scope cannot be proven, the correct action is to rebuild or remeasure.

---

## A Practical Comparison Checklist

Before claiming that a new run beat an old run:

```text
[ ] Compare real size and real KLD, not prediction rows
[ ] Build the source-aware union of both runs
[ ] Do not deduplicate by local numeric ID alone
[ ] Collapse only exact artifact/recipe and measurement matches
[ ] Recompute strict dominance across the full union
[ ] Report which source contributed each nondominated point
[ ] List every strict replacement explicitly
[ ] Preserve the full nondominated archive
[ ] Apply release spacing only after the archive is known
[ ] Record the configuration, revisions, manifests, and logs
```

This is more work than comparing two final README tables.

It is also the difference between a persuasive result and a reproducible one.

---

## Core Principle

MagicQuant should be aggressive in search and conservative in claims.

That means:

```text
preserve measured evidence
identify artifacts by provenance, not coincidence
separate Pareto truth from release presentation
claim only the improvements the source-aware union proves
```

The final survivor menu is for users.

The complete evidence archive is what keeps that menu trustworthy.
