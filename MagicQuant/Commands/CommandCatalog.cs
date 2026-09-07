namespace MagicQuant.Commands;

/// <summary>One registry for dispatch and top-level help, including script compatibility aliases.</summary>
public static class CommandCatalog
{
    public static bool IsHelp(string argument) =>
        argument.Equals("help", StringComparison.OrdinalIgnoreCase) || argument is "--help" or "-h";

    public static Dictionary<string, (string Description, Func<ICommand> Factory)> Create() =>
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["pipeline"] = ("Learn baselines, discover hybrids, validate and export survivors", () => new QuantizationPipeline()),
            ["evolution"] = ("Compatibility alias for pipeline", () => new QuantizationPipeline()),
            ["validate-predictions"] = ("Compare KLD predictions with existing SQLite benchmarks", () => new ValidatePredictions()),
            ["build-hybrids"] = ("Compatibility entry point for the full pipeline and export", () => new BuildHybrids()),
            ["clone-repository-quants"] = ("Rebuild final tensor configurations from a compatible repository/manifest", () => new CloneRepositoryQuants()),
            ["initialize-llama-cpp"] = ("Initialize or update llama.cpp and Python dependencies", () => new InitializeLlamaCpp())
        };
}
