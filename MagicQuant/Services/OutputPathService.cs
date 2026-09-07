namespace MagicQuant.Services;

/// <summary>
/// Resolves output locations without creating directories. Relative-path differences
/// are historical command contracts; keep them explicit to avoid relocating artifacts.
/// </summary>
public static class OutputPathService
{
    public static string Pipeline(string modelWorkDirectory, string? configuredOutput) =>
        string.IsNullOrWhiteSpace(configuredOutput)
            ? Path.Combine(modelWorkDirectory, "Final_Outputs")
            : Path.GetFullPath(Path.Combine(modelWorkDirectory, configuredOutput));

    public static string Clone(string modelWorkDirectory, string? explicitOutput, string? configuredOutput) =>
        Path.GetFullPath(!string.IsNullOrWhiteSpace(explicitOutput)
            ? explicitOutput
            : !string.IsNullOrWhiteSpace(configuredOutput)
                ? configuredOutput
                : Path.Combine(modelWorkDirectory, "FinalOutput"));

    public static string PredictionValidation(string modelWorkDirectory, string? explicitOutput, string? configuredOutput) =>
        !string.IsNullOrWhiteSpace(explicitOutput)
            ? Path.GetFullPath(explicitOutput)
            : !string.IsNullOrWhiteSpace(configuredOutput)
                ? Path.Combine(Path.GetFullPath(configuredOutput), "PredictionValidation")
                : Path.Combine(modelWorkDirectory, "PredictionValidation");
}
