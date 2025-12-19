namespace MagicQuant.Models;

public class TensorGroup
{
    public string Name { get; set; } = "";
    public List<string> Tensors { get; set; } = new();
    
    // Helper to map "embeddings" -> "E", etc.
    public char ShortCode => Name switch
    {
        "embeddings" => 'E',
        "lm_head" => 'H',
        "attn_q" => 'Q',
        "attn_kv" => 'K',
        "attn_output" => 'O',
        "ffn_up_gate" => 'U',
        "ffn_down" => 'D',
        "moe_experts" => 'X',
        "moe_router" => 'R',
        _ => '?'
    };
}