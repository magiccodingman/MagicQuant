using MagicQuant.Models;

namespace MagicQuant;

public static class Config
{
    public static readonly int MaxDataCollectedPerCategory = 5;
    public static readonly int MaxSurvivalRounds = 4;
    public static readonly double CollapseMultiplier = 1.5;
    
    public static readonly List<string> BaselineQuants = new()
    {
        "Q8_0",
        "Q6_K",
        "Q5_K",
        "Q4_K_M",
        "IQ4_NL",
        "MXFP4_MOE"
    };
    
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
    
    public static readonly List<string> BaseConversionModes = new()
    {
        "mxfp4_moe",
        "iq4_nl"
    };
    
    public static readonly List<string> TensorWeightSchemes = new()
    {
        "BF16",
        "F16",
        "MXFP4",
        "Q8_0",
        "Q6_K",
        "Q5_K",
        "IQ4_NL",
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

    
    public static readonly List<TensorGroup> TensorGroups = new()
{
    new TensorGroup
    {
        Name = "embeddings",
        Tensors = new()
        {
            "token_embd.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "word_embeddings.weight",
            "transformer.wte.weight",
            "gpt_neox.embed_in.weight",
            "shared.weight",
            "wte.weight",
        }
    },

    new TensorGroup
    {
        Name = "lm_head",
        Tensors = new()
        {
            "output.weight",
            "lm_head.weight",
            "final_logits_proj.weight",
            "model.embed_out.weight",
            "lm_head.decoder.weight",
            "cls.predictions.decoder.weight",
            "gpt_neox.embed_out.weight",
            "decoder.output_projection.weight",
            "decoder.output_dense.weight",
            "transformer.wte.weight",
            "shared.weight",
        }
    },

    new TensorGroup
    {
        Name = "attn_q",
        Tensors = new()
        {
            "blk.*.attn_q.weight",
            ".*q_proj.*weight",
            ".*query.weight",
            ".*q_proj.weight",
            ".*self_attn.q_proj.weight",
            ".*attention.self.query.weight",
            ".*SelfAttention.q.weight",
            ".*c_attn.weight",
            ".*query_key_value.weight",
        }
    },

    new TensorGroup
    {
        Name = "attn_kv",
        Tensors = new()
        {
            "blk.*.attn_k.weight",
            "blk.*.attn_v.weight",
            ".*k_proj.*weight",
            ".*v_proj.*weight",
            ".*key.weight",
            ".*value.weight",
            ".*self_attn.k_proj.weight",
            ".*self_attn.v_proj.weight",
            ".*attention.self.key.weight",
            ".*attention.self.value.weight",
            ".*SelfAttention.k.weight",
            ".*SelfAttention.v.weight",
            ".*EncDecAttention.k.weight",
            ".*EncDecAttention.v.weight",
            ".*c_attn.weight",
            ".*query_key_value.weight",
        }
    },

    new TensorGroup
    {
        Name = "attn_output",
        Tensors = new()
        {
            "blk.*.attn_output.weight",
            ".*out_proj.*weight",
            ".*o_proj.*weight",
            ".*c_proj.weight",
            ".*attention.output.dense.weight",
            ".*self_attn.out_proj.weight",
            ".*SelfAttention.o.weight",
            ".*self_attention.dense.weight",
            ".*attention.proj.weight",
        }
    },

    new TensorGroup
    {
        Name = "ffn_up_gate",
        Tensors = new()
        {
            "blk.*.ffn_up.weight",
            "blk.*.ffn_gate.weight",
            ".*intermediate.dense.weight",
            ".*c_fc.weight",
            ".*fc1.weight",
            ".*fc_in.weight",
            ".*dense_h_to_4h.weight",
            ".*wi.weight",
            ".*wi_0.weight",
            ".*wi_1.weight",
            ".*mlp.up_proj.weight",
            ".*mlp.gate_proj.weight",
            ".*DenseReluDense.wi_0.weight",
            ".*DenseReluDense.wi_1.weight",
            ".*experts.*wi_0.weight",
            ".*experts.*wi_1.weight",
            "blk.*.ffn_up_exps.weight",
            "blk.*.ffn_gate_exps.weight",
        }
    },

    new TensorGroup
    {
        Name = "ffn_down",
        Tensors = new()
        {
            "blk.*.ffn_down.weight",
            ".*output.dense.weight",
            ".*c_proj.weight",
            ".*fc2.weight",
            ".*fc_out.weight",
            ".*wo.weight",
            ".*dense_4h_to_h.weight",
            ".*mlp.down_proj.weight",
            ".*DenseReluDense.wo.weight",
            ".*experts.*wo.weight",
            "blk.*.ffn_down_exps.weight",
        }
    },

    new TensorGroup
    {
        Name = "moe_experts",
        Tensors = new()
        {
            "blk.*.ffn_.*_expert.*",
            "blk.*.ffn_.*_exps.*",

            ".*experts?\\..*wi_0.*",
            ".*experts?\\..*wi_1.*",
            ".*experts?\\..*wo.*",
            ".*experts?\\..*fc1.*",
            ".*experts?\\..*fc2.*",
            ".*experts?\\..*dense_h_to_4h.*",
            ".*experts?\\..*dense_4h_to_h.*",
        }
    },

    new TensorGroup
    {
        Name = "moe_router",
        Tensors = new()
        {
            "router.*",
            "gate.*",
            "gating.*",
            "routing.*",

            "blk.*.ffn_gate_inp.weight",

            "blk.*.router.*",
            "blk.*.gate_inp.*",
            "blk.*.gate_proj.*",
            "blk.*.gate.weight",
            "blk.*.router_fc.*",

            ".*router.weight",
            ".*gate.weight",
            ".*router_fc.*",
            ".*gating_network.*weight",
            ".*moe_gate.*weight",
        }
    },
};

}