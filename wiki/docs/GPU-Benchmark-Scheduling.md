# GPU Benchmark Scheduling and Measured Topology Planning

MagicQuant performs many real `llama.cpp` benchmarks.

On a machine with multiple GPUs, the obvious question sounds simple:

> Should every benchmark use every GPU, or should MagicQuant run one benchmark per GPU?

The honest answer is:

> **It depends on the model size, the GPUs, the stable offload depth, and whether there is actually a batch of independent jobs waiting.**

MagicQuant therefore treats GPU use as a measured scheduling problem rather than a fixed command-line preference.

---

## The Important Correction

`llama.cpp` benchmarking is not inherently limited to one GPU.

A single benchmark process can split a model across multiple visible GPUs. That can be the only practical way to keep a larger GGUF fully or mostly offloaded.

But using every GPU for every job is not automatically the fastest way to finish a batch.

Consider two strategies on a two-GPU system:

```text
shared topology:
    benchmark A uses GPU 0 + GPU 1
    benchmark B waits

independent topology:
    benchmark A uses GPU 0
    benchmark B uses GPU 1
```

The shared process may finish one large job faster. The independent topology may finish two smaller jobs faster in aggregate.

Those are different optimization targets:

```text
single-job latency != total batch throughput
```

MagicQuant measures both when the hardware and workload allow it.

---

## The Two Topologies

### Shared multi-GPU topology

A shared slot exposes multiple GPUs to one benchmark process.

Conceptually:

```text
candidate.gguf
      |
      v
llama.cpp benchmark process
      |
      +---- GPU 0
      |
      +---- GPU 1
```

This topology is useful when:

- the candidate is too large to fit well on one GPU
- one process needs the combined VRAM pool
- only one benchmark is ready
- the caller is performing an adaptive or sequential benchmark
- measured shared performance is better for that part of the size range

### Independent per-GPU topology

An independent topology creates one benchmark slot per stable GPU.

Conceptually:

```text
candidate-A.gguf ---- benchmark process A ---- GPU 0
candidate-B.gguf ---- benchmark process B ---- GPU 1
```

This topology is useful when:

- multiple independent candidates are ready at the same time
- each candidate fits within an individual GPU's measured capacity
- concurrent workers produce higher aggregate throughput
- neither process needs to borrow the other GPU's VRAM

Independent mode is not selected merely because two GPUs exist. There must be real batch concurrency to exploit.

---

## Why Static Rules Are Not Enough

A rule such as:

```text
two GPUs => always run two workers
```

fails when the model is too large for either GPU.

A rule such as:

```text
two GPUs => always split every benchmark
```

wastes throughput when two smaller jobs could run concurrently.

A file-size-only estimate is also incomplete. Two GPUs with the same advertised VRAM can have different usable headroom because of display usage, other processes, driver behavior, or configured safety limits.

MagicQuant instead discovers:

- the highest stable GPU-layer offload for the shared slot
- the highest stable offload for each independent GPU slot
- the throughput of the shared topology
- the aggregate throughput of the independent topology
- the candidate-size crossover below which independent workers are expected to win

The result is a hardware execution plan measured on the machine that will perform the work.

---

## NGL Is Candidate-Aware

`NGL` is the number of model layers requested for GPU offload.

The correct value is not assumed to be identical for every candidate or every GPU.

MagicQuant starts from measured slot capability, then plans the offload depth for the actual candidate size. If a candidate cannot run stably at the planned depth, the benchmark path can reduce offload rather than treating the whole campaign as impossible.

This matters on asymmetric systems.

For example:

```text
GPU 0 stable Q8 offload: 44 / 66 layers
GPU 1 stable Q8 offload: 57 / 66 layers
shared stable offload:   66 / 66 layers
```

The scheduler should not pretend those independent slots are interchangeable.

When several candidates are ready, candidate-aware best-fit placement prefers:

- a near-full-fit candidate on the weakest adequate GPU
- a candidate needing more offload help on the stronger GPU

That preserves scarce capacity instead of assigning jobs by GPU index alone.

---

## The Measured Crossover

The crossover is a candidate-size boundary, not a universal constant.

Below it, measured independent-worker throughput is expected to be better:

```text
candidate size <= measured crossover
and concurrent batch intent exists
    => independent GPU slots may be used
```

Above it, MagicQuant uses the shared multi-GPU slot:

```text
candidate size > measured crossover
    => shared multi-GPU slot
```

The shared topology also remains the safe choice for singleton and adaptive calls, because there is no second ready job from which to obtain concurrency.

This distinction is important. A crossover derived from two concurrent workers should not be used to force a single benchmark onto one GPU.

---

## Measured Example: Two RTX 3090 GPUs

During the Qwen3.8 27B development campaign, a two-RTX-3090 machine produced this plan:

| Measurement | Result |
| --- | ---: |
| Candidate maximum NGL | 66 |
| Shared stable NGL | 66 / 66 |
| GPU 0 independent Q8 NGL | 44 / 66 |
| GPU 1 independent Q8 NGL | 57 / 66 |
| Measured independent crossover | roughly 23.35–23.72 GiB |
| Configured shared tensor split | `19,23` |

For sub-crossover Q5-sized work, the measured aggregate rates were approximately:

| Topology | Aggregate throughput |
| --- | ---: |
| Shared multi-GPU workers | 0.529 jobs/second |
| Independent per-GPU workers | 0.762 jobs/second |

That is roughly a 44% aggregate throughput improvement for that workload.

This is a case study, not a promised multiplier and not a portable 23 GiB rule. Another model, CUDA build, driver, GPU pair, background load, or benchmark corpus can produce a different crossover.

The transferable lesson is the method:

```text
measure the machine
measure both viable topologies
schedule according to candidate size and batch shape
```

---

## Configuring Usable GPU Memory

MagicQuant can accept per-GPU usable memory limits:

```yaml
hardware:
  gpu_memory_limits_gb:
    0: 19
    1: 23
```

These values are intentionally usable limits, not a claim about the physical capacity printed on the GPU box.

They can reserve headroom for:

- the display server
- another process that must remain active
- driver overhead
- a known stability margin

When the shared slot uses multiple GPUs, MagicQuant passes the configured split in visible-GPU order.

One small but important implementation detail is that `llama.cpp` tools do not all accept the same separator:

```text
common llama.cpp CLI tools:  --tensor-split 19,23
llama-bench:                 --tensor-split 19/23
```

MagicQuant builds the correct argument for the target executable.

---

## Hardware Plan Caching

Probing a Q8 or native model deeply enough to discover stable GPU behavior is useful, but it should not be paid on every run.

MagicQuant caches the execution plan in SQLite. The cache is scoped to the things that can materially change the answer, including:

- model identity
- imatrix identity where relevant to the generated probe artifact
- detected GPU topology
- configured memory limits
- `llama.cpp` executable fingerprint
- execution-plan schema version

When that scope still matches, later runs can reuse the measured plan.

When the hardware, model, binary, or configuration changes, a stale plan should not silently remain authoritative.

To force a fresh hardware probe:

```text
--recheck-hardware-probe
```

The alias `--force-refresh-hardware-probe` is also accepted.

---

## Failure Behavior

GPU planning is an optimization layer. It should not turn an otherwise valid benchmark into a brittle all-or-nothing operation.

The intended fallback order is:

```text
use cached measured plan when valid
    |
    v
probe stable shared and independent slots when needed
    |
    v
choose topology for candidate size and caller batch intent
    |
    v
reduce candidate offload if the planned NGL is unstable
    |
    v
fall back safely when a GPU topology cannot be validated
```

An unstable independent slot disables that optimization path. It does not justify fabricating a throughput estimate.

---

## What This Planner Does Not Claim

The planner does not claim:

- that multi-GPU is always faster
- that one process per GPU is always faster
- that advertised VRAM equals usable VRAM
- that a crossover measured for one model is valid for another
- that KLD or PPL changes because of the scheduling topology

The benchmark result remains the evidence. The planner changes how quickly MagicQuant can collect that evidence.

Its job is simple:

> **Spend the available GPU capacity in the highest-throughput stable shape that the current candidate batch can actually use.**
