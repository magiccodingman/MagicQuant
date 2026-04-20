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

    private void AddIfNotNull(TensorGroup group, byte candidateId)
    {
        if (candidateId == TensorWeightScheme.NULL.UniqueId)
            return;

        var candidate = BaselineQuants.FromId(candidateId);

        Tensors.Add(new HybridTensor
        {
            TGroup = group,
            CandidateBaseline = candidate,
            TensorType = candidate.DefaultTensorScheme ?? TensorWeightScheme.GetCurrentNativePrecisionScheme()
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
                    CandidateBaseline = t.CandidateBaseline,
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
        BaselineQuants blanketCandidate)
    {
        return new HybridQuant
        {
            BaseQuant = baseQuant,
            Tensors = groups
                .Select(g => new HybridTensor
                {
                    TGroup = g,
                    CandidateBaseline = blanketCandidate,
                    TensorType = blanketCandidate.DefaultTensorScheme ?? TensorWeightScheme.GetCurrentNativePrecisionScheme()
                })
                .ToList()
        };
    }

    public static explicit operator HybridQuant(TensorConfig c) => new HybridQuant(c);
}

public class HybridTensor
{
    public TensorGroup TGroup { get; set; } = null!;
    public BaselineQuants CandidateBaseline { get; set; } = default!;
    public TensorWeightScheme TensorType { get; set; } = default!;
}
