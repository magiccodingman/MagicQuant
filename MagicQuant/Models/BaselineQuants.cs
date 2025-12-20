using System.Collections.Immutable;

namespace MagicQuant.Models;

public record BaselineQuants(
    sbyte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    bool AllowedAsBaseConversion = false)
{
    public static readonly BaselineQuants Q8_0 = new(0, false, ["Q8_0"]);
    public static readonly BaselineQuants Q6_K = new(1, false, ["Q6_K"]);
    public static readonly BaselineQuants Q5_K = new(2, false, ["Q5_K"]);
    public static readonly BaselineQuants Q4_K_M = new(3, false, ["Q4_K_M"]);

    public static readonly BaselineQuants MXFP4_MOE = new(4, false, ["MXFP4_MOE"], true);
    public static readonly BaselineQuants IQ4_NL = new(5, false, ["IQ4_NL"], true);
    
    public static readonly BaselineQuants IQ4_XS = new(6, false, ["IQ4_NL"]);

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
