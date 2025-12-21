namespace MagicQuant.Models;

public class HybridQuant
{
    public BaselineQuants BaseQuant { get; set; } = default!;
    public List<HybridTensor> Tensors { get; set; } = new List<HybridTensor>();

    // Converting constructor: TensorConfig -> HybridQuant
    public HybridQuant(TensorConfig c)
    {
        // If you don’t like LINQ here, swap to dictionary/array maps.
        BaseQuant  = BaselineQuants.All.First(b => b.UniqueId == c.BaseQuant);
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.Embeddings,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.Embeddings)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.LmHead,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.LmHead)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.AttnQ,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.AttnQ)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.AttnKV,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.AttnKV)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.AttnOutput,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.AttnOutput)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.FfnUpGate,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.FfnUpGate)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.FfnDown,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.FfnDown)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.MoeExperts,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.MoeExperts)
        });
        
        Tensors.Add(new HybridTensor()
        {
            TGroup = TReg.MoeRouter,
            TensorType = TensorWeightScheme.All.First(g => g.UniqueId == c.MoeRouter)
        });
    }

    // Conversion operator: TensorConfig -> HybridQuant
    public static explicit operator HybridQuant(TensorConfig c) => new HybridQuant(c);
}

public class HybridTensor
{
    public TensorGroup TGroup { get; set; } = null!;
    public TensorWeightScheme TensorType { get; set; }
}