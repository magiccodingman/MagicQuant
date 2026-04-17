namespace MQ.DB.Models;

public class HybridQuant
{
    public BaselineQuants BaseQuant { get; set; } = default!;
    public List<HybridTensor> Tensors { get; set; } = new();

    public HybridQuant() { }

    public HybridQuant(TensorConfig c)
    {
        BaseQuant = BaselineQuants.FromId(c.BaseQuant);

        AddIfNotNull(TReg.Embeddings, c.Embeddings);
        AddIfNotNull(TReg.LmHead, c.LmHead);
        AddIfNotNull(TReg.AttnQ, c.AttnQ);
        AddIfNotNull(TReg.AttnKV, c.AttnKV);
        AddIfNotNull(TReg.AttnOutput, c.AttnOutput);
        AddIfNotNull(TReg.FfnUpGate, c.FfnUpGate);
        AddIfNotNull(TReg.FfnDown, c.FfnDown);
        AddIfNotNull(TReg.MoeExperts, c.MoeExperts);
        AddIfNotNull(TReg.MoeRouter, c.MoeRouter);
    }

    private void AddIfNotNull(TensorGroup group, byte schemeId)
    {
        if (schemeId == TensorWeightScheme.NULL.UniqueId)
            return;

        var scheme = TensorWeightScheme.All_Allowed_Hybrid_Quants.First(g => g.UniqueId == schemeId);

        Tensors.Add(new HybridTensor
        {
            TGroup = group,
            TensorType = scheme
        });
    }

    public HybridQuant Clone()
    {
        return new HybridQuant
        {
            BaseQuant = BaseQuant,
            Tensors = Tensors
                .Select(t => new HybridTensor
                {
                    TGroup = t.TGroup,
                    TensorType = t.TensorType
                })
                .ToList()
        };
    }

    public static HybridQuant CreatePureBaseline(BaselineQuants baseQuant)
    {
        return new HybridQuant
        {
            BaseQuant = baseQuant,
            Tensors = new List<HybridTensor>()
        };
    }

    public static HybridQuant CreateBlanket(
        BaselineQuants baseQuant,
        IEnumerable<TensorGroup> groups,
        TensorWeightScheme blanketScheme)
    {
        return new HybridQuant
        {
            BaseQuant = baseQuant,
            Tensors = groups
                .Select(g => new HybridTensor
                {
                    TGroup = g,
                    TensorType = blanketScheme
                })
                .ToList()
        };
    }

    public static explicit operator HybridQuant(TensorConfig c) => new HybridQuant(c);
}

public class HybridTensor
{
    public TensorGroup TGroup { get; set; } = null!;
    public TensorWeightScheme TensorType { get; set; } = default!;
}