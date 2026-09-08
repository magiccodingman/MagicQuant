using Spectre.Console;

namespace MagicQuant.Services;

public static class MagicQuantManifestPathService
{
    public const string ManifestDirectoryName = "magicquant-manifest";

    public const string CloneConfigsFileName = "magicquant.clone-configs.json";
    public const string FinalSurvivorsFileName = "magicquant.final-survivors.json";
    public const string ReplacementsFileName = "magicquant.replacements.json";
    public const string HybridMapFileName = "magicquant.hybrid-map.json";
    public const string CloneBenchmarksFileName = "magicquant.clone-benchmarks.json";
    public const string IsolationSamplesFileName = "magicquant.isolation-samples.json";
    public const string BadTradesFileName = "magicquant.bad-trades.json";

    public static readonly IReadOnlyList<string> KnownManifestFileNames =
    [
        CloneConfigsFileName,
        FinalSurvivorsFileName,
        ReplacementsFileName,
        HybridMapFileName,
        CloneBenchmarksFileName,
        IsolationSamplesFileName,
        BadTradesFileName
    ];

    public static string EnsureManifestDirectory(string outputDirectory)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("Output directory is required.", nameof(outputDirectory));

        string normalizedOutput = NormalizeDirectoryPath(outputDirectory);

        // Idempotency guard: callers sometimes already pass the manifest directory itself.
        // Do not create magicquant-manifest/magicquant-manifest.
        if (string.Equals(Path.GetFileName(normalizedOutput), ManifestDirectoryName, StringComparison.OrdinalIgnoreCase))
        {
            Directory.CreateDirectory(normalizedOutput);
            return normalizedOutput;
        }

        string path = Path.Combine(normalizedOutput, ManifestDirectoryName);
        Directory.CreateDirectory(path);
        return path;
    }

    public static string GetManifestFilePath(string outputDirectory, string fileName)
        => Path.Combine(EnsureManifestDirectory(outputDirectory), NormalizeManifestFileName(fileName));

    public static string RelativeManifestPath(string fileName)
    {
        string normalized = NormalizeManifestFileName(fileName).Replace('\\', '/').TrimStart('/');

        if (normalized.StartsWith(ManifestDirectoryName + "/", StringComparison.OrdinalIgnoreCase))
            return normalized;

        return $"{ManifestDirectoryName}/{normalized}";
    }

    public static string HuggingFaceResolvePath(string fileName, bool download = false)
    {
        string path = $"./../../resolve/main/{RelativeManifestPath(fileName)}";
        return download ? path + "?download=true" : path;
    }

    public static string HuggingFaceGgufResolvePath(string fileName)
        => $"./../../resolve/main/{Uri.EscapeDataString(fileName)}?download=true";

    public static void WriteManifestLog(string message, bool isWarning = false, bool isError = false)
    {
        string color = isError ? "red" : isWarning ? "yellow" : "grey";
        string line = $"[{DateTime.Now:HH:mm:ss}] manifest: {message}";
        AnsiConsole.MarkupLine($"[{color}]{Markup.Escape(line)}[/]");
    }

    private static string NormalizeDirectoryPath(string path)
    {
        string full = Path.GetFullPath(path);
        string root = Path.GetPathRoot(full) ?? string.Empty;
        string trimmed = full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return string.IsNullOrWhiteSpace(trimmed) ? root : trimmed;
    }

    private static string NormalizeManifestFileName(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            throw new ArgumentException("Manifest file name is required.", nameof(fileName));

        string normalized = fileName.Trim().Replace('\\', '/').TrimStart('/');

        if (normalized.StartsWith(ManifestDirectoryName + "/", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[(ManifestDirectoryName.Length + 1)..];

        return normalized;
    }
}
