using System.Collections.Immutable;

namespace MQ.DB.Models;

public record BaselineQuants(
    byte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    HybridQuant? BaseConversionBase = null)
{
    /// <summary>
    /// Reserved internal ID for the original/native source model (BF16/F16/F32).
    /// This MUST NOT collide with any real llama.cpp export base quant.
    /// </summary>
    public const byte NativeSourceUniqueId = 250;

    public static readonly BaselineQuants Q8_0 = new(0, false, ["Q8_0"]);
    public static readonly BaselineQuants Q6_K = new(1, false, ["Q6_K"]);
    public static readonly BaselineQuants Q5_K = new(2, false, ["Q5_K"]);
    public static readonly BaselineQuants Q4_K_M = new(3, false, ["Q4_K_M"]);

    public static readonly BaselineQuants MXFP4_MOE = new(4, false, ["MXFP4_MOE"],
        new HybridQuant
        {
            BaseQuant = MXFP4_MOE,
            Tensors = TReg.All
                .Select(g => new HybridTensor
                {
                    TGroup = g,
                    TensorType = TensorWeightScheme.MXFP4
                })
                .ToList()
        });

    public static readonly BaselineQuants IQ4_XS = new(6, false, ["IQ4_XS"],
        new HybridQuant
        {
            BaseQuant = IQ4_XS,
            Tensors = TReg.All
                .Select(g => new HybridTensor
                {
                    TGroup = g,
                    TensorType = TensorWeightScheme.IQ4_XS
                })
                .ToList()
        });

    public static readonly BaselineQuants IQ4_NL = new(5, false, ["IQ4_NL"]);

    public static BaselineQuants GetBF16Quant()
    {
        return new(
            NativeSourceUniqueId,
            false,
            [Cache.TorchType?.ToString() ?? "BF16"]
        );
    }

    // IQ3 and lower require imatrix
    //public static readonly BaselineQuants IQ3_M = new(7, true,  ["IQ3_M"], true);
    //public static readonly BaselineQuants IQ2_M = new(8, true,  ["IQ2_M"], true);

    public static readonly ImmutableArray<BaselineQuants> All =
    [
        Q8_0,
        Q6_K,
        Q5_K,
        Q4_K_M,
        MXFP4_MOE,
        IQ4_NL,
        IQ4_XS,
        //IQ3_M,
        //IQ2_M
    ];
}