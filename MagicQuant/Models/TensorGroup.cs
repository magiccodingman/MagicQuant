using System.Collections.Immutable;
using System.Linq;


namespace MagicQuant.Models;

public class TensorGroupInfo
{   
    public TensorGroup Group { get; set; }
}

/// <summary>
/// Represents a categorized group of tensors with a unique name and matching patterns.
/// </summary>
public record TensorGroup(byte UniqueId, string Name, ImmutableArray<string> Tensors)
{
    /// <summary>
    /// Helper to map the group name to a single-character identifier for CLI or UI display.
    /// </summary>
    public char ShortCode => Name switch
    {
        "embeddings"  => 'E',
        "lm_head"     => 'H',
        "attn_q"      => 'Q',
        "attn_kv"     => 'K',
        "attn_output" => 'O',
        "ffn_up_gate" => 'U',
        "ffn_down"    => 'D',
        "moe_experts" => 'X',
        "moe_router"  => 'R',
        _             => '?'
    };
}

/// <summary>
/// Tensor Registry
/// </summary>
public static class TReg
{
    public static readonly TensorGroup Embeddings = new(0, "embeddings", [
        "token_embd\\.weight", 
        "model\\.embed_tokens\\.weight", 
        "embed_tokens\\.weight",
        "tok_embeddings\\.weight", 
        "word_embeddings\\.weight", 
        "transformer\\.wte\\.weight",
        "wte\\.weight"
    ]);

    public static readonly TensorGroup LmHead = new(1, "lm_head", [
        "output\\.weight", 
        "lm_head\\.weight", 
        "final_logits_proj\\.weight",
        "model\\.embed_out\\.weight", 
        "lm_head\\.decoder\\.weight"
    ]);

    public static readonly TensorGroup AttnQ = new(2, "attn_q", [
        // Matches blk.0.attn_q.weight
        "blk\\..*\\.attn_q\\.weight",
        ".*q_proj.*weight", 
        ".*query\\.weight", 
        ".*self_attn\\.q_proj\\.weight", 
        ".*attention\\.self\\.query\\.weight",
        ".*SelfAttention\\.q\\.weight", 
        ".*c_attn\\.weight", 
        ".*query_key_value\\.weight"
    ]);

    public static readonly TensorGroup AttnKV = new(3, "attn_kv", [
        "blk\\..*\\.attn_k\\.weight", 
        "blk\\..*\\.attn_v\\.weight", 
        ".*k_proj.*weight", 
        ".*v_proj.*weight",
        ".*key\\.weight", 
        ".*value\\.weight", 
        ".*self_attn\\.k_proj\\.weight", 
        ".*self_attn\\.v_proj\\.weight",
        ".*attention\\.self\\.key\\.weight", 
        ".*attention\\.self\\.value\\.weight", 
        ".*SelfAttention\\.k\\.weight",
        ".*SelfAttention\\.v\\.weight", 
        ".*EncDecAttention\\.k\\.weight", 
        ".*EncDecAttention\\.v\\.weight"
    ]);

    public static readonly TensorGroup AttnOutput = new(4, "attn_output", [
        "blk\\..*\\.attn_output\\.weight", 
        ".*out_proj.*weight", 
        ".*o_proj.*weight", 
        ".*c_proj\\.weight",
        ".*attention\\.output\\.dense\\.weight", 
        ".*self_attn\\.out_proj\\.weight",
        ".*SelfAttention\\.o\\.weight", 
        ".*self_attention\\.dense\\.weight", 
        ".*attention\\.proj\\.weight"
    ]);

    public static readonly TensorGroup FfnUpGate = new(5, "ffn_up_gate", [
        // This is where ffn_gate belongs!
        "blk\\..*\\.ffn_up\\.weight", 
        "blk\\..*\\.ffn_gate\\.weight", 
        
        ".*intermediate\\.dense\\.weight",
        ".*c_fc\\.weight", 
        ".*fc1\\.weight", 
        ".*fc_in\\.weight", 
        ".*dense_h_to_4h\\.weight",
        ".*wi\\.weight", 
        ".*wi_0\\.weight", 
        ".*wi_1\\.weight", 
        ".*mlp\\.up_proj\\.weight",
        ".*mlp\\.gate_proj\\.weight", 
        ".*DenseReluDense\\.wi_0\\.weight", 
        ".*DenseReluDense\\.wi_1\\.weight",
        ".*experts.*wi_0\\.weight", 
        ".*experts.*wi_1\\.weight", 
        "blk\\..*\\.ffn_up_exps\\.weight",
        "blk\\..*\\.ffn_gate_exps\\.weight"
    ]);

    public static readonly TensorGroup FfnDown = new(6, "ffn_down", [
        "blk\\..*\\.ffn_down\\.weight", 
        ".*output\\.dense\\.weight", 
        ".*c_proj\\.weight", 
        ".*fc2\\.weight",
        ".*fc_out\\.weight", 
        ".*wo\\.weight", 
        ".*dense_4h_to_h\\.weight", 
        ".*mlp\\.down_proj\\.weight",
        ".*DenseReluDense\\.wo\\.weight", 
        ".*experts.*wo\\.weight", 
        "blk\\..*\\.ffn_down_exps\\.weight"
    ]);

    public static readonly TensorGroup MoeExperts = new(7, "moe_experts", [
        "blk\\..*\\.ffn_.*expert.*", 
        "blk\\..*\\.ffn_.*exps.*", 
        ".*experts?\\..*wi_0.*",
        ".*experts?\\..*wi_1.*", 
        ".*experts?\\..*wo.*", 
        ".*experts?\\..*fc1.*",
        ".*experts?\\..*fc2.*", 
        ".*experts?\\..*dense_h_to_4h.*", 
        ".*experts?\\..*dense_4h_to_h.*"
    ]);

    public static readonly TensorGroup MoeRouter = new(8, "moe_router", [
        // Strict Router definitions
        "router.*", 
        "gating.*", 
        "routing.*", 
        
        // This was the culprit. 
        // We use Negative Lookbehind (?<!ffn_) to ensure we don't match "ffn_gate"
        ".*(?<!ffn_)gate\\.weight", 
        
        ".*gating_network\\.weight", 
        ".*moe_gate\\.weight",
        "blk\\..*\\.ffn_gate_inp\\.weight", // Explicit Qwen MoE router
        "blk\\..*\\.gate_inp\\.weight",
        "blk\\..*\\.gate_proj\\.weight", 
        
        // Only match "gate.weight" if it is strictly that (e.g. Mixtral)
        // or ensure it doesn't have ffn_ before it.
        "blk\\..*\\.router.*",
        "blk\\..*\\.router_fc.*"
    ]);

    /// <summary>
    /// Provides a complete list of all registered tensor groups.
    /// </summary>
    public static readonly ImmutableArray<TensorGroup> All = 
    [
        Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, 
        FfnUpGate, FfnDown, MoeExperts, MoeRouter
    ];

    /// <summary>
    /// Look up a group by its string name (useful when parsing external configs).
    /// </summary>
    public static TensorGroup? GetByName(string name) => 
        All.FirstOrDefault(g => g.Name.Equals(name, System.StringComparison.OrdinalIgnoreCase));
}