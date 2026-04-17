using System.Collections.Immutable;
using MQ.DB;

namespace MQ.DB.Models;

public sealed class TensorWeightScheme
{
    private readonly HashSet<byte> _defaultBannedGroupIds;

    public byte UniqueId { get; }
    public bool RequiresImatrix { get; }
    public ImmutableArray<string> Names { get; }
    public List<TensorGroup> BannedGroups { get; }
    public ushort? BlockNeo { get; }
    public bool IsEligibleForBaseline { get; }

    private TensorWeightScheme(
        byte uniqueId,
        bool requiresImatrix,
        ImmutableArray<string> names,
        IEnumerable<TensorGroup> bannedGroups,
        ushort? blockNeo,
        bool isEligibleForBaseline = true)
    {
        UniqueId = uniqueId;
        RequiresImatrix = requiresImatrix;
        Names = names;
        BlockNeo = blockNeo;
        IsEligibleForBaseline = isEligibleForBaseline;

        var distinctGroups = bannedGroups
            .GroupBy(x => x.UniqueId)
            .Select(x => x.First())
            .ToList();

        BannedGroups = distinctGroups;
        _defaultBannedGroupIds = distinctGroups.Select(x => x.UniqueId).ToHashSet();
    }

    public void ResetRuntimeBans()
    {
        BannedGroups.Clear();
        foreach (var group in TReg.All.Where(x => _defaultBannedGroupIds.Contains(x.UniqueId)))
            BannedGroups.Add(group);
    }

    public bool IsBannedFor(TensorGroup group) => BannedGroups.Any(x => x.UniqueId == group.UniqueId);

    public static void ResetAllRuntimeBans()
    {
        foreach (var scheme in All)
            scheme.ResetRuntimeBans();
    }

    public static void ValidateSmallestConfiguration()
    {
        var ordered = GetSmallestInOrder();

        if (ordered.Length == 0)
            throw new InvalidOperationException("TensorWeightScheme.GetSmallestInOrder() must return at least one item.");

        var duplicateIds = ordered
            .GroupBy(x => x.UniqueId)
            .Where(g => g.Count() > 1)
            .Select(g => g.First().Names[0])
            .ToList();

        if (duplicateIds.Count > 0)
        {
            throw new InvalidOperationException(
                $"TensorWeightScheme.GetSmallestInOrder() contains duplicates: {string.Join(", ", duplicateIds)}");
        }

        var knownIds = All.Select(x => x.UniqueId).ToHashSet();
        var unknown = ordered
            .Where(x => !knownIds.Contains(x.UniqueId))
            .Select(x => x.Names[0])
            .Distinct()
            .ToList();

        if (unknown.Count > 0)
        {
            throw new InvalidOperationException(
                $"TensorWeightScheme.GetSmallestInOrder() includes unknown schemes: {string.Join(", ", unknown)}");
        }
    }


    /// <summary>
    /// This labels the models that're supposed to quantize the smallest
    /// in order. Top of array being the smallest, the further down,
    /// it becomes larger in order of expected quantization size.
    /// </summary>
    /// <returns></returns>
    public static TensorWeightScheme[] GetSmallestInOrder()
    {
        return [IQ4_XS, IQ4_NL, Q4_K, Q6_K, Q8_0, BF16, F16, F32];
    }

    public static TensorWeightScheme GetCurrentNativePrecisionScheme()
    {
        return (Cache.TorchType ?? Cache.MainTorchType.BF16) switch
        {
            Cache.MainTorchType.BF16 => BF16,
            Cache.MainTorchType.F16 => F16,
            Cache.MainTorchType.F32 => F32,
            _ => BF16
        };
    }

    public static bool IsNativePrecisionScheme(TensorWeightScheme scheme)
    {
        return scheme.UniqueId == BF16.UniqueId ||
               scheme.UniqueId == F16.UniqueId ||
               scheme.UniqueId == F32.UniqueId;
    }

    // Compatibility shim for any older code still referencing BF16_F16.
    public static TensorWeightScheme BF16_F16 => GetCurrentNativePrecisionScheme();

    public static readonly TensorWeightScheme NULL =
        new(0, false, ["NULL"], Array.Empty<TensorGroup>(), null, isEligibleForBaseline: false);

    public static readonly TensorWeightScheme BF16 =
        new(1, false, ["BF16", "BFLOAT16"], Array.Empty<TensorGroup>(), null, isEligibleForBaseline: false);

    /*public static readonly TensorWeightScheme MXFP4 =
        new(
            2,
            false,
            ["MXFP4"],
            new[]
            {
                TReg.AttnQ,
                TReg.MoeRouter,
                TReg.MoeExperts
            },
            32,
            isEligibleForBaseline: false);*/

    public static readonly TensorWeightScheme Q8_0 =
        new(3, false, ["Q8_0"], Array.Empty<TensorGroup>(), null);

    public static readonly TensorWeightScheme Q6_K =
        new(4, false, ["Q6_K"], Array.Empty<TensorGroup>(), 256);

    public static readonly TensorWeightScheme Q5_K =
        new(5, false, ["Q5_K"], new[] { TReg.MoeRouter }, 256);

    public static readonly TensorWeightScheme IQ4_XS =
        new(6, false, ["IQ4_XS"], new[] { TReg.MoeRouter }, 32);

    public static readonly TensorWeightScheme IQ4_NL =
        new(7, false, ["IQ4_NL"], new[] { TReg.MoeRouter }, 32);

    public static readonly TensorWeightScheme IQ3_S =
        new(
            8,
            true,
            ["IQ3_S"],
            new[]
            {
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter
            },
            32
        );

    public static readonly TensorWeightScheme IQ3_XS =
        new(
            9,
            true,
            ["IQ3_XS"],
            new[]
            {
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter
            },
            32
        );

    public static readonly TensorWeightScheme IQ3_XXS =
        new(
            10,
            true,
            ["IQ3_XXS"],
            new[]
            {
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter
            },
            32
        );

    public static readonly TensorWeightScheme IQ2_S =
        new(
            11,
            true,
            ["IQ2_S"],
            new[]
            {
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter,
                TReg.MoeExperts
            },
            32
        );

    public static readonly TensorWeightScheme IQ2_XS =
        new(
            12,
            true,
            ["IQ2_XS"],
            new[]
            {
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter,
                TReg.MoeExperts
            },
            32
        );

    public static readonly TensorWeightScheme IQ2_XXS =
        new(
            13,
            true,
            ["IQ2_XXS"],
            new[]
            {
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter,
                TReg.MoeExperts,
                TReg.AttnKV
            },
            32
        );

    public static readonly TensorWeightScheme Q4_K =
        new(
            14,
            false,
            ["Q4_K"],
            new[] { TReg.MoeRouter },
            32
        );

    public static readonly TensorWeightScheme F16 =
        new(15, false, ["F16", "FLOAT16", "FP16", "HALF"], Array.Empty<TensorGroup>(), null, isEligibleForBaseline: false);

    public static readonly TensorWeightScheme F32 =
        new(16, false, ["F32", "FLOAT32", "FP32", "FLOAT"], Array.Empty<TensorGroup>(), null, isEligibleForBaseline: false);

    // This is the set used by hybrid search / combination generation.
    public static readonly ImmutableArray<TensorWeightScheme> All_Allowed_Hybrid_Quants =
    [
        NULL,
        BF16,
        //F16,
        //MXFP4,
        Q8_0,
        Q6_K,
        Q5_K,
        IQ4_XS,
        IQ4_NL,
        Q4_K
    ];

    // This is the true registry of everything known.
    public static readonly ImmutableArray<TensorWeightScheme> All =
    [
        NULL,
        BF16,
        F16,
        F32,
        //MXFP4,
        Q8_0,
        Q6_K,
        Q5_K,
        IQ4_XS,
        IQ4_NL,
        IQ3_S,
        IQ3_XS,
        IQ3_XXS,
        IQ2_S,
        IQ2_XS,
        IQ2_XXS,
        Q4_K
    ];
}
