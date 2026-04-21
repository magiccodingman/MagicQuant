namespace MQ.DB.Models;

public enum HybridTensorOverrideMode
{
    LearnedBaselineCandidate = 1,
    ExactTensorScheme = 2
}

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

    private void AddIfNotNull(TensorGroup group, byte storedId)
    {
        if (BaselineQuants.IsNullTensorConfigGroupSlot(storedId))
            return;

        var decodedBaselineId = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedId);

        if (BaselineQuants.IsNativeExactAlias(decodedBaselineId))
        {
            var exactScheme = BaselineQuants.ResolveExactOverrideScheme(decodedBaselineId);
            Tensors.Add(HybridTensor.CreateExact(group, exactScheme));
            return;
        }

        var candidate = BaselineQuants.FromId(decodedBaselineId);
        Tensors.Add(HybridTensor.CreateLearned(group, candidate));
    }

    public HybridQuant Clone()
    {
        return new HybridQuant
        {
            BaseQuant = BaseQuant,
            Tensors = Tensors
                .Select(t => t.Clone())
                .ToList()
        };
    }

    public HybridTensor? TryGetTensor(TensorGroup group) =>
        Tensors.FirstOrDefault(x => x.TGroup.UniqueId == group.UniqueId);

    public HybridTensor GetRequiredTensor(TensorGroup group) =>
        TryGetTensor(group) ?? throw new InvalidOperationException($"HybridQuant does not contain group '{group.Name}'.");

    public void SetExactOverride(TensorGroup group, TensorWeightScheme exactScheme)
    {
        RemoveGroupIfPresent(group);
        Tensors.Add(HybridTensor.CreateExact(group, exactScheme));
    }

    public void SetLearnedCandidateOverride(TensorGroup group, BaselineQuants candidateBaseline)
    {
        RemoveGroupIfPresent(group);
        Tensors.Add(HybridTensor.CreateLearned(group, candidateBaseline));
    }

    public void RemoveGroupIfPresent(TensorGroup group)
    {
        Tensors.RemoveAll(x => x.TGroup.UniqueId == group.UniqueId);
    }

    public static HybridQuant CreatePureBaseline(BaselineQuants baseQuant)
    {
        return new HybridQuant
        {
            BaseQuant = baseQuant,
            Tensors = new List<HybridTensor>()
        };
    }

    public static HybridQuant CreateExactBlanket(
        BaselineQuants baseQuant,
        IEnumerable<TensorGroup> groups,
        TensorWeightScheme exactScheme)
    {
        return new HybridQuant
        {
            BaseQuant = baseQuant,
            Tensors = groups
                .Select(g => HybridTensor.CreateExact(g, exactScheme))
                .ToList()
        };
    }

    public static HybridQuant CreateLearnedCandidateBlanket(
        BaselineQuants baseQuant,
        IEnumerable<TensorGroup> groups,
        BaselineQuants candidateBaseline)
    {
        return new HybridQuant
        {
            BaseQuant = baseQuant,
            Tensors = groups
                .Select(g => HybridTensor.CreateLearned(g, candidateBaseline))
                .ToList()
        };
    }

    public static explicit operator HybridQuant(TensorConfig c) => new HybridQuant(c);
}

public class HybridTensor
{
    public TensorGroup TGroup { get; set; } = null!;
    public HybridTensorOverrideMode OverrideMode { get; set; }
    public BaselineQuants? CandidateBaseline { get; set; }
    public TensorWeightScheme? ExactTensorScheme { get; set; }
    public TensorWeightScheme MaterializedTensorScheme { get; set; } = default!;

    // Compatibility alias retained for older call sites.
    public TensorWeightScheme TensorType
    {
        get => MaterializedTensorScheme;
        set => MaterializedTensorScheme = value;
    }

    public static HybridTensor CreateLearned(TensorGroup group, BaselineQuants candidateBaseline)
    {
        if (BaselineQuants.IsNativeExactAlias(candidateBaseline))
        {
            throw new InvalidOperationException(
                $"Baseline '{candidateBaseline.Names[0]}' is an exact/native alias and cannot be used as a learned baseline candidate.");
        }

        return new HybridTensor
        {
            TGroup = group,
            OverrideMode = HybridTensorOverrideMode.LearnedBaselineCandidate,
            CandidateBaseline = candidateBaseline,
            ExactTensorScheme = null,
            MaterializedTensorScheme = candidateBaseline.DefaultTensorScheme ?? TensorWeightScheme.GetCurrentNativePrecisionScheme()
        };
    }

    public static HybridTensor CreateExact(TensorGroup group, TensorWeightScheme exactScheme)
    {
        return new HybridTensor
        {
            TGroup = group,
            OverrideMode = HybridTensorOverrideMode.ExactTensorScheme,
            CandidateBaseline = null,
            ExactTensorScheme = exactScheme,
            MaterializedTensorScheme = exactScheme
        };
    }

    public HybridTensor Clone()
    {
        return new HybridTensor
        {
            TGroup = TGroup,
            OverrideMode = OverrideMode,
            CandidateBaseline = CandidateBaseline,
            ExactTensorScheme = ExactTensorScheme,
            MaterializedTensorScheme = MaterializedTensorScheme
        };
    }

    public void ValidateOrThrow()
    {
        if (TGroup == null)
            throw new InvalidOperationException("HybridTensor.TGroup is required.");

        switch (OverrideMode)
        {
            case HybridTensorOverrideMode.LearnedBaselineCandidate:
                if (CandidateBaseline == null)
                    throw new InvalidOperationException($"HybridTensor for group '{TGroup.Name}' is missing CandidateBaseline.");

                if (ExactTensorScheme != null)
                    throw new InvalidOperationException($"HybridTensor for group '{TGroup.Name}' cannot specify ExactTensorScheme when OverrideMode is LearnedBaselineCandidate.");

                if (BaselineQuants.IsNativeExactAlias(CandidateBaseline))
                    throw new InvalidOperationException($"HybridTensor for group '{TGroup.Name}' cannot use exact/native alias '{CandidateBaseline.Names[0]}' as a learned baseline candidate.");
                break;

            case HybridTensorOverrideMode.ExactTensorScheme:
                if (ExactTensorScheme == null)
                    throw new InvalidOperationException($"HybridTensor for group '{TGroup.Name}' is missing ExactTensorScheme.");

                if (CandidateBaseline != null)
                    throw new InvalidOperationException($"HybridTensor for group '{TGroup.Name}' cannot specify CandidateBaseline when OverrideMode is ExactTensorScheme.");
                break;

            default:
                throw new InvalidOperationException($"HybridTensor for group '{TGroup.Name}' has unknown OverrideMode '{OverrideMode}'.");
        }
    }
}
