using MagicQuant.Models;

namespace MagicQuant;

public static class Config
{
    public static readonly int MaxDataCollectedPerCategory = 5;
    public static readonly int MaxSurvivalRounds = 4;
    public static readonly double CollapseMultiplier = 1.5;
    
    
    public static readonly List<string> SensitivityProbeGroups = new()
    {
        "embeddings",
        "lm_head",
        "attn_q",
        "attn_kv",
        "attn_output",
        "ffn_up_gate",
        "ffn_down",
    };

    // MoE-specific probe groups (added if MoE detected)
    public static readonly List<string> SensitivityProbeGroupsMoe = new()
    {
        "moe_router",
        "moe_experts",
    };

    // Critical "brain" layers that cause non-linear collapse when crushed together
    public static readonly List<string> BrainLayers = new()
    {
        "embeddings",
        "lm_head",
        "attn_output",
    };

    // Schemes that trigger collapse penalty when applied to brain layers
    public static readonly List<string> CollapsePenaltySchemes = new()
    {
        "MXFP4",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
    };
    

    
    public static readonly List<string> MoeIndicatorTensors = new()
    {
        "blk.*.ffn_up_expert_0.weight",
        "blk.*.ffn_gate_expert_0.weight",
        "blk.*.ffn_down_expert_0.weight",

        // Qwen3-MOE / Unsloth / modern MOE
        "blk.*.ffn_up_exps.weight",
        "blk.*.ffn_gate_exps.weight",
        "blk.*.ffn_down_exps.weight",
        "blk.*.ffn_gate_inp.weight",

        // router variants
        "router.weight",
        "gate.weight",
        "blk.*.router.*",
        "blk.*.gate_proj.*",
        "blk.*.gate_inp.*",
    };


}