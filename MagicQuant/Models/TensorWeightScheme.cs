using System.Collections.Immutable;

namespace MagicQuant.Models;

public record TensorWeightScheme(
    sbyte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    ImmutableArray<TensorGroup> BannedGroups,
    bool AlwaysBuild = true)
{
    // NULL: always-present groups, never nullable
    public static readonly TensorWeightScheme NULL =
        new(
            0,
            false,
            ["NULL"],
            [
                TReg.Embeddings,
                TReg.AttnQ,
                TReg.AttnKV,
                TReg.AttnOutput,
                TReg.FfnDown,
                TReg.FfnUpGate
            ]
        );

    // BF16 and F16 intentionally share UniqueId
    public static readonly TensorWeightScheme BF16_F16 =
        new(
            1,
            false,
            ["BF16", "F16"],
            ImmutableArray<TensorGroup>.Empty
        );

    public static readonly TensorWeightScheme MXFP4 =
        new(
            2,
            false,
            ["MXFP4"],
            [
                TReg.AttnQ,
                TReg.MoeRouter,
                TReg.MoeExperts
            ]
        );

    public static readonly TensorWeightScheme Q8_0 =
        new(3, false, ["Q8_0"], ImmutableArray<TensorGroup>.Empty);

    public static readonly TensorWeightScheme Q6_K =
        new(4, false, ["Q6_K"], ImmutableArray<TensorGroup>.Empty);

    public static readonly TensorWeightScheme Q5_K =
        new(
            5,
            false,
            ["Q5_K"],
            [TReg.MoeRouter]
        );

    public static readonly TensorWeightScheme IQ4_XS =
        new(
            6,
            false,
            ["IQ4_XS"],
            [TReg.MoeRouter]
        );
    
    /*
    public static readonly TensorWeightScheme IQ4_NL =
        new(
            7,
            false,
            ["IQ4_NL"],
            [TReg.MoeRouter]
        );

  

    // IQ3 levels
    public static readonly TensorWeightScheme IQ3_S =
        new(
            8,
            true,
            ["IQ3_S"],
            [
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter
            ]
        );

    public static readonly TensorWeightScheme IQ3_XS =
        new(
            9,
            true,
            ["IQ3_XS"],
            [
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter
            ]
        );

    public static readonly TensorWeightScheme IQ3_XXS =
        new(
            10,
            true,
            ["IQ3_XXS"],
            [
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter
            ]
        );

    // IQ2 levels: extremely restrictive
    public static readonly TensorWeightScheme IQ2_S =
        new(
            11,
            true,
            ["IQ2_S"],
            [
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter,
                TReg.MoeExperts
            ]
        );

    public static readonly TensorWeightScheme IQ2_XS =
        new(
            12,
            true,
            ["IQ2_XS"],
            [
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter,
                TReg.MoeExperts
            ]
        );

    public static readonly TensorWeightScheme IQ2_XXS =
        new(
            13,
            true,
            ["IQ2_XXS"],
            [
                TReg.Embeddings,
                TReg.LmHead,
                TReg.MoeRouter,
                TReg.MoeExperts,
                TReg.AttnKV
            ]
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

