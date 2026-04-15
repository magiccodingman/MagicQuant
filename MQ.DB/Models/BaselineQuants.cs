using System.Collections.Immutable;

namespace MQ.DB.Models;

public record BaselineQuants(
    byte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    TensorWeightScheme? DefaultTensorScheme,
    HybridQuant? BaseConversionBase = null)
{
    public const byte NativeSourceUniqueId = 250;

    public static readonly BaselineQuants Q8_0 = new(0, false, ["Q8_0"], TensorWeightScheme.Q8_0);
    public static readonly BaselineQuants Q6_K = new(1, false, ["Q6_K"], TensorWeightScheme.Q6_K);
    public static readonly BaselineQuants Q5_K = new(2, false, ["Q5_K"], TensorWeightScheme.Q5_K);
    public static readonly BaselineQuants Q4_K_M = new(3, false, ["Q4_K_M"], TensorWeightScheme.Q4_K);

    public static readonly BaselineQuants IQ4_NL = new(5, false, ["IQ4_NL"], TensorWeightScheme.IQ4_NL);

    public static readonly BaselineQuants IQ4_XS = new(
        6,
        false,
        ["IQ4_XS"],
        TensorWeightScheme.IQ4_XS,
        new HybridQuant
        {
            BaseQuant = null!,
            Tensors = TReg.All
                .Select(g => new HybridTensor
                {
                    TGroup = g,
                    TensorType = TensorWeightScheme.IQ4_XS
                })
                .ToList()
        });

    // IQ3 and lower require imatrix
    //public static readonly BaselineQuants IQ3_M = new(7, true,  ["IQ3_M"], true);
    //public static readonly BaselineQuants IQ2_M = new(8, true,  ["IQ2_M"], true);

    public static readonly ImmutableArray<BaselineQuants> All =
    [
        Q8_0,
        Q6_K,
        Q5_K,
        Q4_K_M,
        IQ4_NL,
        IQ4_XS,
        //IQ3_M,
        //IQ2_M
    ];

    static BaselineQuants()
    {
        IQ4_XS.BaseConversionBase!.BaseQuant = IQ4_XS;
        ValidateIntegrityOrThrow();
    }

    public static void ValidateIntegrityOrThrow()
    {
        var invalidBaselines = All
            .Where(x => x.DefaultTensorScheme == null)
            .Select(x => x.Names.IsDefaultOrEmpty ? $"id:{x.UniqueId}" : x.Names[0])
            .ToList();

        if (invalidBaselines.Count > 0)
        {
            throw new InvalidOperationException(
                "Every BaselineQuants entry must define DefaultTensorScheme. Missing for: " +
                string.Join(", ", invalidBaselines));
        }

        var duplicateDefaultSchemeIds = All
            .GroupBy(x => x.DefaultTensorScheme!.UniqueId)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicateDefaultSchemeIds.Count > 0)
        {
            var duplicateNames = duplicateDefaultSchemeIds
                .Select(id => TensorWeightScheme.All.First(s => s.UniqueId == id).Names[0]);

            throw new InvalidOperationException(
                "DefaultTensorScheme must be unique across BaselineQuants entries. Duplicates: " +
                string.Join(", ", duplicateNames));
        }

        var schemesMissingBaseline = TensorWeightScheme.All
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            // Some schemes can be experimental and intentionally not promoted to baseline.
            // Hard-enforce only for the established shipped baseline set.
            .Where(x => x.UniqueId != TensorWeightScheme.MXFP4.UniqueId)
            .Where(x => !All.Any(b => b.DefaultTensorScheme!.UniqueId == x.UniqueId))
            .Select(x => x.Names[0])
            .ToList();

        if (schemesMissingBaseline.Count > 0)
        {
            throw new InvalidOperationException(
                "Every TensorWeightScheme must be linked by exactly one BaselineQuants.DefaultTensorScheme. Missing for: " +
                string.Join(", ", schemesMissingBaseline));
        }
    }

    public static BaselineQuants GetBF16Quant()
    {
        return new(
            NativeSourceUniqueId,
            false,
            [(Cache.TorchType ?? Cache.MainTorchType.BF16).ToString()],
            TensorWeightScheme.BF16_F16);
    }

    public static BaselineQuants FromId(byte id)
    {
        if (id == NativeSourceUniqueId)
            return GetBF16Quant();

        var found = All.FirstOrDefault(x => x.UniqueId == id);
        if (found == null)
            throw new InvalidOperationException($"Unknown baseline quant id '{id}'.");

        return found;
    }
}
