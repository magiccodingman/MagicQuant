# MagicQuant Wiki
> Evolution process to find the best quant tensor weights to build the most optimal GGUF options for an AI model.


### Index

* [Naming Scheme](docs/naming-scheme.md) - Learn about the MagicQuant naming scheme which helps shorten lengthy names.
* [Precision Loss Guide](docs/precision-loss-guide.md) - A guide to understanding precision loss and the philosophy behind this project.
* [Model Selection & Philosophy](docs/model-selection.md) - Why and what models are uploaded.
* [HuggingFace: Magic Quant Collection](https://huggingface.co/collections/magiccodingman/magic-quant) - The best measured candidates from this version of the project.

---

## Evolutionary Tensor Search: A Hybrid Quantization Framework for LLM Compression

### Abstract

Current quantization methodologies typically apply global precision schemes (e.g., Q4, Q8) uniformly across model weights. This can overlook differences in how various parts of Transformer architectures respond to quantization noise. This paper introduces **"Magic Quant,"** an evolutionary search framework. By dynamically grouping tensors, probing group sensitivity, and employing a residual-learning predictor, the system searches for hybrid quantization combinations that improve the measured Size/Perplexity/Speed trade-off relative to standard baselines.

### 1\. Introduction

A common quantization workflow is to choose a preset format (like GGUF’s Q4\_K\_M) and apply it across the model. MagicQuant V1 explored whether model-specific mixed-precision configurations could occupy useful trade-off points that standard presets do not directly represent. A "Q4" model might, for example, contain some parameter groups that tolerate heavier compression while other groups are more sensitive.

This framework proposes a shift from **Global Quantization** to **Functional Group Mixed-Precision Quantization**. By automating the discovery of these mixtures, V1 searched for models that could be smaller than one standard baseline while retaining perplexity closer to a higher-precision baseline.

### 2\. Methodology

#### 2.1. Tensor Grouping Strategy

Rather than optimizing per-layer (which creates a search space of $N_{schemes}^{N_{layers}}$), the system groups tensors by architectural role. This reduces dimensionality while retaining functional granularity.

  * **Groups:** `embeddings`, `lm_head`, `attn_q`, `attn_kv`, `attn_output`, `ffn_up_gate`, `ffn_down`.
  * **MoE Specifics:** If a Mixture-of-Experts architecture is detected, the system isolates `moe_router` from `moe_experts` so their behavior can be measured separately.

#### 2.2. The Sensitivity Probe Phase ("Feeling Out the Tensors")

Before the search begins, the system executes a "Probe Phase" to estimate how each functional group responds to compression.

1.  **Baseline:** Measures the uncompressed (F16/BF16) perplexity.
2.  **Aggressive Probing:** For each group, the system creates a temporary hybrid where *only* that group is aggressively quantized (e.g., MXFP4) while the rest remain at baseline.
3.  **Sensitivity Weighting:** The degradation in perplexity ($\Delta PPL$) is measured. Groups that cause high degradation are assigned high sensitivity weights; groups showing less degradation receive lower weights.

#### 2.3. The Predictive Scoring Model

To avoid measuring millions of combinations, a predictive model estimates the performance of a theoretical combination $C$.

**The Loss Prediction:**

$$Loss_{pred} = \sum_{g \in Groups} (S_g \times W_g)$$

Where $S_g$ is the measured sensitivity of the group and $W_g$ is the quantization noise factor of the chosen scheme (e.g., Q4 vs Q8).

**The Non-Linear Collapse Penalty:**
The system includes a heuristic penalty for combinations where multiple high-sensitivity groups (such as Embeddings + Output Head + Attention Output) are simultaneously assigned aggressive quantization schemes.

$$if \ Count(Sensitive_{aggressive}) \ge 2: Loss_{final} = max(Loss_{pred} \times \alpha, Loss_{pred} + \beta)$$

**Active Learning (Residual Corrections):**
As real measurements come in, the system compares $Loss_{measured}$ vs $Loss_{predicted}$. It calculates residual pattern keys (e.g., `Base=MXFP4 | Embed=Q8`). These residuals are fed back into the predictor, allowing it to learn that specific combinations perform better or worse than the initial estimate suggests.

#### 2.4. Evolutionary Survival Rounds

The search process is iterative (Survival Rounds), borrowing the idea of selection and mutation from evolutionary search:

1.  **Generation:** A large pool of candidates is generated based on valid schemes (BF16, Q8, Q6, Q5, IQ4, MXFP4).
2.  **Classification:** Candidates are sorted into "Tiers" based on standard baselines. (e.g., "Tier Q4" contains hybrids smaller than a Q5 but larger than a Q3).
3.  **Selection (The Tournament):** Within each tier, candidates are chosen for:
      * *Lowest Loss* (Precision Winner)
      * *Smallest Size* (Efficiency Winner)
      * *Highest TPS* (Speed Winner)
      * *Balanced Score* (Weighted average of all three)
4.  **Mutation:** Top candidates undergo mutation:
      * *Protector Strategy:* The most sensitive group not already at high precision is upgraded (e.g., Q6 $\to$ Q8).
      * *Crusher Strategy:* A lower-sensitivity group is downgraded to test whether additional size savings are worthwhile (e.g., Q4 $\to$ MXFP4).

#### 2.5. Epsilon-Greedy Exploration

To reduce the chance of getting stuck in local optima, the system employs an $\epsilon$-greedy strategy. In every round, a random sample of lower-ranked predicted candidates is sent to the measurement phase. If one performs unexpectedly well, the Residual Correction engine updates and the search can shift toward that pattern.

### 3\. Experimental Setup

  * **Context:** Benchmarks use a **32k context window**, stressing the KV cache and attention layers more than shorter-context tests.
  * **Perplexity:** Measured across three distinct domains:
    1.  *General* (Wikitext)
    2.  *Code* (CodeParrot)
    3.  *Math* (GSM8k)
  * **Inference Speed:** Tokens Per Second (TPS) is measured to track the speed behavior of different quantization mixes.

### 4\. Conclusion

Within the V1 evaluation framework, the "Evolutionary Tensor Search" showed that mixed-precision configurations could produce interesting trade-off points relative to standard quantization presets. The system therefore treated quantization as a model-specific search problem rather than assuming one static preset would always be the most useful option.

-----

# Technical Architecture & Methodology

## System Overview: The "Magic Quant" Pipeline

The Magic Quant framework operates as an automated research system. Instead of treating model quantization as a static format conversion, it treats it as a high-dimensional search problem. The system employs an **Active Learning Evolutionary Strategy** to navigate the trade-off space between Model Size, Inference Speed (TPS), and Perplexity (PPL).

The pipeline consists of four distinct phases that cycle until convergence: **Initialization**, **Sensitivity Probing**, **Evolutionary Search**, and **Validation**.

### 1. Phase I: Structural Analysis & Initialization
Before optimization begins, the system constructs a model of the target LLM’s tensor layout and benchmark baselines.

* **Tensor Topology Scanning:** The system inspects the raw GGUF structure to map specific tensors to functional groups. It identifies architectural components such as *Embeddings*, *Attention Heads (Q/K/V)*, *Output Projections*, and *Feed-Forward Networks (FFN)*. It automatically detects specialized architectures like Mixture-of-Experts (MoE) or Tied Embeddings.
* **Baseline Calibration:** The system builds and benchmarks standard reference points (e.g., FP16, Q8_0, Q6_K, Q4_K_M). These serve as boundaries for the **Tier System**, ensuring that discovered hybrids can be classified relative to known standards (e.g., "Tier Q4" = larger than Q3, smaller than Q5).
* **Seed Injection (Warm Start):** The search is seeded with heuristic starting points. For example, in MoE models, it can inject "High-Contrast" seeds (High-Precision Router / Low-Precision Experts) so those strategies are evaluated early.

### 1.5. Dynamic Weighting and Modeling Inputs

Immediately following the Tensor Topology Scanning, the system calculates two sets of weights that govern the predictive process:

* **1. Physics Weights (Size/TPS Factor):** The system scans the unquantized GGUF to count the total number of parameters belonging to each functional group.

    * $$W_{physics, g} = \frac{\text{Parameter Count}_g}{\text{Total Model Parameters}}$$
    
    * These weights sum to 1.0 and are used to estimate the model's final **size** and **TPS**. They are useful because the FFN group might account for a large majority of model parameters, meaning its compression factor heavily influences final size.
    
    * The prediction provides a consistency check for the expected physical size of the hybrid before performance is estimated.

* **2. Sensitivity Weights (Loss Factor):** These weights, determined in Phase II, measure the observed PPL impact of quantizing each group.
    * $$W_{sensitivity, g} = \frac{\Delta PPL_{probe, g}}{\sum \Delta PPL_{probes}}$$
    
    * These weights govern the **Loss Prediction** and identify groups where V1 observed larger changes under aggressive quantization.

### 2. Phase II: The Sensitivity Probe
Before generating hybrids, the system measures how each functional group responds to aggressive compression in isolation.

* **The Probe Strategy:** The system generates temporary test models where a single functional group (e.g., `attn_output`) is aggressively compressed (e.g., to `MXFP4`) while the rest of the model remains at high precision.
* **Sensitivity Weighting:** By measuring the perplexity degradation ($\Delta PPL$) of these probes, the system assigns a **Sensitivity Score** to each group.
    * *High Sensitivity:* Groups that produce a large measured PPL change when aggressively quantized.
    * *Low Sensitivity:* Groups that show relatively little measured PPL change.
    * These weights inform the predictive engine, allowing it to prioritize higher precision for groups that appeared more sensitive in the probe.

### 3. Phase III: The Evolutionary Core (The Engine)
This is the iterative loop where the system searches for useful configurations. It follows a **Predict $\rightarrow$ Measure $\rightarrow$ Learn** cycle.

#### A. The Predictive Model
Rather than measuring millions of combinations physically, the system uses a simulator to estimate the performance of a theoretical hybrid.
* **Loss Prediction:** Calculated using the **Learned Sensitivity Weights** derived from Phase II, combined with the quantization noise factors of the chosen formats.
* **Non-Linear Collapse Penalty:** The system applies a heuristic penalty when multiple high-sensitivity groups are aggressively quantized at the same time. This raises their predicted loss and reduces how often those combinations are selected for measurement.
* **Balanced Scoring:** Candidates are ranked by a composite objective function:
    $$Score = (0.4 \times Precision) + (0.3 \times Size) + (0.3 \times Speed)$$

#### B. The Selection Tournament (Measurement)
Because benchmarking is computationally expensive, the system limits the number of candidates that receive real measurements. It selects candidates based on:
1.  **Tier Winners:** The best predicted combo for every Tier (Q6, Q5, Q4, etc.).
2.  **Ambiguity Resolution:** If two combos have near-identical predicted scores, both are measured to resolve the tie.
3.  **Epsilon-Greedy Exploration:** To avoid local optima, the system randomly samples lower-ranked predicted candidates. If one performs unexpectedly well, it triggers an update to the learning model.
4.  **Strict Dominance Filtering:** Any hybrid that is physically as large as a `Q8_0` file is discarded unless it improves on the baseline in both measured precision and speed.

#### C. Adaptive Mutation Strategies
When a successful candidate is identified (a "Survivor"), the system spawns variants intended to push the measured trade-off further:
* **The Protector Strategy:** Identifies the group with the highest sensitivity score that is *not* yet max precision and upgrades it (e.g., `Attn_Output: Q6` $\rightarrow$ `Q8`).
* **The Crusher Strategy:** Identifies a lower-sensitivity group and downgrades it to test additional size savings (e.g., `FFN: Q4` $\rightarrow$ `MXFP4`).

### 4. Phase IV: Active Learning & Calibration
This is the feedback mechanism that adapts the predictor to the model being tested.
* **Residual Analysis:** After every real-world measurement, the system compares the *Predicted Loss* vs. the *Actual Loss*.
* **Pattern Correction:** It identifies specific patterns causing prediction errors (e.g., *"This architecture reacts poorly to MXFP4 Embeddings regardless of what the initial estimate says"*).
* **Global Calibration:** The system applies a correction factor to future predictions containing that pattern. Over several rounds, the predictor adapts to quirks observed in the model architecture.

### 5. Benchmark Validation
The system runs its validation suite on measured candidates:
* **Inference Speed:** Measured in Tokens Per Second (TPS) to identify whether a quantization mix changes decoding performance.
* **Tri-Domain Perplexity:** Precision is measured across three datasets:
    1.  **General Language** (Wikitext)
    2.  **Coding Capabilities** (CodeParrot)
    3.  **Mathematical Reasoning** (GSM8k)

### 6. Special Handling: Mixture of Experts (MoE)
The system includes specialized logic for MoE architectures (e.g., Mixtral, Qwen-MoE).
* **Router Prioritization:** The system treats the `moe_router` (Gating) tensors separately and can keep them at higher precision when the probes show high sensitivity.
* **Expert Compression:** It independently evaluates more aggressive `MXFP4` or `IQ4` quantization for `moe_experts` while maintaining higher precision elsewhere when the measurements support that pattern.

---

# Empirical Findings & The "MXFP4 Anomaly"

### 3.1. The "MXFP4" Phenomenon
One of the recurring patterns observed by the **Evolutionary Tensor Search** was the use of `MXFP4_MOE` (Microscaling Float 4) as a *base layer*, including on some dense (non-MoE) models.

In V1 testing, the evolutionary algorithm frequently converged on a pattern where:
1.  **Base:** A large portion of parameters, often including Feed-Forward Network weights, used `MXFP4_MOE`.
2.  **Protected Groups:** A smaller set of groups that measured as more sensitive—such as Embeddings, Attention Output, or Routers—used `Q8_0`, `Q6_K`, or another higher-precision option.

Under V1’s PPL/size/TPS measurements, this pattern sometimes produced trade-off points that compared favorably with standard Q4/Q5/Q6 baselines.

### 3.2. Case Study: Heavy MXFP4 Damage in a Small Model (Granite 4.0 350M)
Small models (<1B parameters) can be sensitive to quantization because they have less redundancy. In this test, a standard `MXFP4` conversion produced severe degradation.

**The V1 Result:**
MagicQuant found that keeping selected groups at higher precision while using MXFP4 elsewhere produced a dramatically lower measured PPL delta than the pure MXFP4 baseline.

| Metric | Standard MXFP4_MOE | Magic Quant Hybrid (MXFP4 Base) | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg PPL Loss** | 1172.27% (Incoherent) | **0.0816%** (Very Low Measured Drift) | **~14,000x lower measured PPL loss** |
| **File Size** | 0.34 GB | 0.54 GB | Slightly Larger |
| **TPS** | ~1700 | ~1705 | Equivalent |

*Significance in V1:* The experiment showed that a pure MXFP4 baseline could fail badly while a mixed-precision configuration using MXFP4 for only part of the model could produce a much lower PPL delta.

### 3.3. Comparison With Standard Tiers (Qwen3 4B)
For mid-sized dense models, V1 tested whether hybrids could occupy useful positions between standard K-quant baselines.

**Comparison vs. Standard Q5_K:**

| Configuration | Size (GB) | Speed (TPS) | PPL Loss (Lower is Better) |
| :--- | :--- | :--- | :--- |
| **Standard Q5_K** | 2.69 | 361.03 | 0.5973% |
| **Hybrid (MXFP4 akv_Q6_K-ao_Q5_K-aq_Q5_K-emb_Q6_K-fd_Q5_K-fug_Q5_K)** | **2.65** | **356.70** | **1.15%** |
| **Hybrid (MXFP4 Base - akv_BF16-ao_Q8_0-aq_Q6_K-emb_Q6_K-fd_Q8_0-fug_Q8_0)** | **3.98** | **403.80** | **0.0826%** |

*Note: The 2.65GB hybrid shows higher measured loss than Q5_K, while the 3.98GB variant shows substantially lower measured PPL loss and higher TPS than the Q5_K baseline on the tested hardware. These are different trade-off points rather than a single configuration dominating every dimension.*

In V1, results like these supported the idea that mixed-precision configurations could produce combinations of size, speed, and PPL delta not represented by a standard preset.

### 3.4. Architectural Adaptability (Apriel 1.5 15B)
The evolutionary algorithm does not simply force MXFP4 everywhere. It evaluates different quantization choices for the model being tested. In the case of Apriel 15B, the search selected `Q5_K` for specific groups where that option performed well under the V1 measurements.

**Selected Survivors (Apriel 15B):**
* `mxfp4_moe-te_Q5_K-out_Q5_K-rt_Q5_K-gt_Q5_K` (0.148% Loss)
* `mxfp4_moe-te_IQ4_NL-out_IQ4_NL-rt_IQ4_NL-gt_Q5_K` (0.277% Loss)

This was evidence, within V1’s framework, that the search was responding to model-specific measurements rather than applying one fixed rule such as "always quantize FFN to Q4."

### 3.5. Resources & Models
The models referenced in this research, along with other discovered hybrids, are available for public analysis.

* **Selected Magic Quant V1 Candidates:**
    [HuggingFace: Magic Quant Collection](https://huggingface.co/collections/magiccodingman/magic-quant)

* **Experimental MXFP4 Hybrids (research data, not intended as production recommendations):**
    [HuggingFace: MXFP4 Hybrid Collection](https://huggingface.co/collections/magiccodingman/mxfp4-hybrid-gguf)
