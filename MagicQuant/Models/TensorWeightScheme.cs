using System.Collections.Immutable;

namespace MagicQuant.Models;

public sealed class TensorWeightScheme
{
    public byte UniqueId { get; }
    public bool RequiresImatrix { get; }
    public ImmutableArray<string> Names { get; }
    public List<TensorGroup> BannedGroups { get; }
    public ushort? BlockNeo { get; }

    private TensorWeightScheme(
        byte uniqueId,
        bool requiresImatrix,
        ImmutableArray<string> names,
        IEnumerable<TensorGroup> bannedGroups,
        ushort? blockNeo)
    {
        UniqueId = uniqueId;
        RequiresImatrix = requiresImatrix;
        Names = names;
        BannedGroups = new List<TensorGroup>(bannedGroups);
        BlockNeo = blockNeo;
    }

    // NULL: always-present groups, never nullable
    public static TensorWeightScheme NULL =
        new(
            0,
            false,
            ["NULL"],
            
            Array.Empty<TensorGroup>(), 
            null
        );

    // BF16 and F16 intentionally share UniqueId
    public static TensorWeightScheme BF16_F16 =
        new(
            1,
            false,
            ["BF16", "F16"],
            Array.Empty<TensorGroup>(),
            null
        );

    public static TensorWeightScheme MXFP4 =
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
            32
        );

    public static TensorWeightScheme Q8_0 =
        new(3, false, ["Q8_0"], Array.Empty<TensorGroup>(), null);

    public static TensorWeightScheme Q6_K =
        new(4, false, ["Q6_K"], Array.Empty<TensorGroup>(), 256);

    public static TensorWeightScheme Q5_K =
        new(
            5,
            false,
            ["Q5_K"],
            new[] { TReg.MoeRouter }, 
            256
        );

    public static TensorWeightScheme IQ4_XS =
        new(
            6,
            false,
            ["IQ4_XS"],
            new[] { TReg.MoeRouter },
            32
        );

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
        MXFP4,
        Q8_0,
        Q6_K,
        Q5_K,
        IQ4_XS,
        //IQ4_NL,
        /*IQ3_S,
        IQ3_XS,
        IQ3_XXS,
        IQ2_S,
        IQ2_XS,
        IQ2_XXS*/
    ];
}
