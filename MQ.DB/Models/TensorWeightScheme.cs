using System.Collections.Immutable;

namespace MQ.DB.Models;

public sealed class TensorWeightScheme
{
    private readonly HashSet<byte> _defaultBannedGroupIds;

    public byte UniqueId { get; }
    public bool RequiresImatrix { get; }
    public ImmutableArray<string> Names { get; }
    public List<TensorGroup> BannedGroups { get; }
    public ushort? BlockNeo { get; }
    public bool IsSmallest { get; }

    private TensorWeightScheme(
        byte uniqueId,
        bool requiresImatrix,
        ImmutableArray<string> names,
        IEnumerable<TensorGroup> bannedGroups,
        ushort? blockNeo,
        bool isSmallest = false)
    {
        UniqueId = uniqueId;
        RequiresImatrix = requiresImatrix;
        Names = names;
        BlockNeo = blockNeo;
        IsSmallest = isSmallest;
        
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

    public bool IsBannedFor(TensorGroup group)
    {
        return BannedGroups.Any(x => x.UniqueId == group.UniqueId);
    }

    public static void ResetAllRuntimeBans()
    {
        foreach (var scheme in All)
            scheme.ResetRuntimeBans();
    }

    public static readonly TensorWeightScheme NULL =
        new(
            0,
            false,
            ["NULL"],
            Array.Empty<TensorGroup>(),
            null);

    public static readonly TensorWeightScheme BF16_F16 =
        new(
            1,
            false,
            ["BF16", "F16", "F32"],
            Array.Empty<TensorGroup>(),
            null);

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
            32);*/

    public static readonly TensorWeightScheme Q8_0 =
        new(3, false, ["Q8_0"], Array.Empty<TensorGroup>(), null);

    public static readonly TensorWeightScheme Q6_K =
        new(4, false, ["Q6_K"], Array.Empty<TensorGroup>(), 256);

    public static readonly TensorWeightScheme Q5_K =
        new(
            5,
            false,
            ["Q5_K"],
            new[] { TReg.MoeRouter },
            256);

    public static readonly TensorWeightScheme IQ4_XS =
        new(
            6,
            false,
            ["IQ4_XS"],
            new[] { TReg.MoeRouter },
            32,
            true);
    
    /*
    public static TensorWeightScheme IQ4_NL =
        new(
            7,
            false,
            ["IQ4_NL"],
            new[] { TReg.MoeRouter },
       32
        );

    public static TensorWeightScheme IQ3_S =
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

    public static TensorWeightScheme IQ3_XS =
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

    public static TensorWeightScheme IQ3_XXS =
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

    public static TensorWeightScheme IQ2_S =
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

    public static TensorWeightScheme IQ2_XS =
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

    public static TensorWeightScheme IQ2_XXS =
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
    */

    public static readonly ImmutableArray<TensorWeightScheme> All =
    [
        NULL,
        BF16_F16,
       // MXFP4,
        Q8_0,
        Q6_K,
        Q5_K,
        IQ4_XS,
    ];
}