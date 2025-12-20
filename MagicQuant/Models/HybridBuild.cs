namespace MagicQuant.Models;

public class HybridTensor
{
    public TensorGroup TensorGroup { get; set; } = null!;
    public string TensorType { get; set; }
}

public class HybridBuild
{
    public string Base { get; set; }
    public List<HybridTensor>? Tensors { get; set; } 
}