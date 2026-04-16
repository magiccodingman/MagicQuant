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
        // Older / generic expert patterns you already had
        "blk.*.ffn_up_expert_0.weight",
        "blk.*.ffn_gate_expert_0.weight",
        "blk.*.ffn_down_expert_0.weight",

        // Older modern-MoE / GGUF-ish patterns
        "blk.*.ffn_up_exps.weight",
        "blk.*.ffn_gate_exps.weight",
        "blk.*.ffn_down_exps.weight",
        "blk.*.ffn_gate_inp.weight",

        // Generic router variants
        "router.weight",
        "gate.weight",
        "blk.*.router.*",
        "blk.*.gate_proj.*",
        "blk.*.gate_inp.*",

        // Qwen3.5 native HF MoE
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
        "model.language_model.layers.*.mlp.gate.weight",
        "model.language_model.layers.*.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.*.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.*.mlp.shared_expert.down_proj.weight",

        // Gemma 4 MoE
        "model.language_model.layers.*.experts.gate_up_proj",
        "model.language_model.layers.*.experts.down_proj",
        "model.language_model.layers.*.router.proj.weight",
        "model.language_model.layers.*.router.per_expert_scale",
        "model.language_model.layers.*.router.scale",
    };


}