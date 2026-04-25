using System.Text.RegularExpressions;
using MagicQuant.Models;
using MQ.DB.Models;

namespace MagicQuant.Services;

/// <summary>
/// Centralizes the public naming rules used by exported GGUF files, README rows,
/// links, and diagnostic logs. Internal tensor-combo display names stay internal.
/// </summary>
public sealed class FinalArtifactNamingService
{
    private static readonly Regex UnsafeFileChars = new(@"[^A-Za-z0-9._-]+", RegexOptions.Compiled);

    public FinalArtifactNamingContext CreateContext(
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots)
    {
        var ranges = BuildRanges(pureBaselineSnapshots);
        return new FinalArtifactNamingContext(ranges);
    }

    public FinalArtifactName BuildName(
        BenchmarkSnapshotRecord snapshot,
        FinalArtifactNamingContext context,
        ISet<string>? reservedFileNames = null)
    {
        string prefix = SanitizeToken(Config.OutputNamePrefix);
        if (string.IsNullOrWhiteSpace(prefix))
            prefix = "Model";

        string tag;
        string providerToken;
        string quantFamily;

        if (snapshot.IsHybrid)
        {
            providerToken = "MQ";
            quantFamily = ResolveHybridRangeFamily(snapshot, context);
            int ordinal = context.NextHybridOrdinal(quantFamily);
            tag = $"{providerToken}-{SanitizeToken(quantFamily)}_{ordinal}";
        }
        else if (snapshot.Quant.BaseQuant.IsExternalRepositoryBaseline)
        {
            providerToken = ResolveExternalProviderToken(snapshot.Quant.BaseQuant);
            quantFamily = NormalizeExternalDisplayName(snapshot.Quant.BaseQuant.Names[0], providerToken);
            tag = SanitizeToken(quantFamily);
        }
        else
        {
            providerToken = "LM";
            quantFamily = snapshot.Quant.BaseQuant.Names[0];
            tag = $"{providerToken}-{SanitizeToken(quantFamily)}";
        }

        string stem = $"{prefix}-{tag}";
        string fileName = MakeUniqueFileName($"{stem}.gguf", reservedFileNames);

        return new FinalArtifactName
        {
            FileName = fileName,
            DisplayName = Path.GetFileNameWithoutExtension(fileName),
            ProviderToken = providerToken,
            QuantFamilyOrBaseline = quantFamily
        };
    }

    public string BuildDisplayLabel(
        BenchmarkSnapshotRecord snapshot,
        FinalArtifactNamingContext context)
    {
        string prefix = SanitizeToken(Config.OutputNamePrefix);
        if (string.IsNullOrWhiteSpace(prefix))
            prefix = "Model";

        if (snapshot.IsHybrid)
            return $"{prefix}-MQ-{SanitizeToken(ResolveHybridRangeFamily(snapshot, context))}";

        if (snapshot.Quant.BaseQuant.IsExternalRepositoryBaseline)
        {
            string providerToken = ResolveExternalProviderToken(snapshot.Quant.BaseQuant);
            return $"{prefix}-{SanitizeToken(NormalizeExternalDisplayName(snapshot.Quant.BaseQuant.Names[0], providerToken))}";
        }

        return $"{prefix}-LM-{SanitizeToken(snapshot.Quant.BaseQuant.Names[0])}";
    }

    public IReadOnlyList<ProviderCredit> BuildProviderCredits(
        IReadOnlyCollection<ExportedArtifactRecord> artifacts)
    {
        var result = new Dictionary<string, ProviderCredit>(StringComparer.OrdinalIgnoreCase);

        result["llama.cpp"] = new ProviderCredit
        {
            Name = "llama.cpp",
            Url = "https://github.com/ggml-org/llama.cpp",
            Note = "Baseline quantization formats and llama.cpp tooling."
        };

        foreach (var artifact in artifacts)
        {
            foreach (var baseline in EnumerateBaselinesUsedBy(artifact.Snapshot.Quant))
            {
                if (!baseline.IsExternalRepositoryBaseline)
                    continue;

                string providerName = string.IsNullOrWhiteSpace(baseline.ShortSourceName)
                    ? "External provider"
                    : baseline.ShortSourceName!;

                string providerToken = ResolveExternalProviderToken(baseline);
                if (string.Equals(providerName, "Unsloth", StringComparison.OrdinalIgnoreCase))
                    providerName = "Unsloth";

                string? url = HybridBenchmarkRepository.BuildExternalRepositoryUrl(baseline);
                if (string.IsNullOrWhiteSpace(url) && !string.IsNullOrWhiteSpace(baseline.SourceRepository))
                    url = $"https://huggingface.co/{baseline.SourceRepository}";

                string key = !string.IsNullOrWhiteSpace(url) ? url : providerName;
                result[key] = new ProviderCredit
                {
                    Name = providerName,
                    Url = url ?? string.Empty,
                    Note = $"External learned baseline source ({providerToken})."
                };
            }
        }

        return result.Values
            .OrderBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
            .ThenBy(x => x.Url, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static IEnumerable<BaselineQuants> EnumerateBaselinesUsedBy(HybridQuant quant)
    {
        yield return quant.BaseQuant;

        foreach (var tensor in quant.Tensors)
        {
            if (tensor.OverrideMode == HybridTensorOverrideMode.LearnedBaselineCandidate && tensor.CandidateBaseline != null)
                yield return tensor.CandidateBaseline;
        }
    }

    private static IReadOnlyList<FinalArtifactRange> BuildRanges(IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots)
    {
        var anchors = pureBaselineSnapshots
            .Where(x => !x.IsHybrid)
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ToList();

        var ranges = new List<FinalArtifactRange>();

        for (int i = 0; i < anchors.Count - 1; i++)
        {
            var lowerDamageLarger = anchors[i];
            var higherDamageSmaller = anchors[i + 1];

            if (higherDamageSmaller.SizeBytes >= lowerDamageLarger.SizeBytes)
                continue;

            ranges.Add(new FinalArtifactRange
            {
                MinSizeBytes = higherDamageSmaller.SizeBytes,
                MaxSizeBytes = lowerDamageLarger.SizeBytes,
                RangeFamily = NormalizeRangeFamily(higherDamageSmaller.Quant.BaseQuant),
                SmallerHigherDamageAnchor = higherDamageSmaller,
                LargerLowerDamageAnchor = lowerDamageLarger
            });
        }

        return ranges
            .OrderBy(x => x.MinSizeBytes)
            .ThenBy(x => x.MaxSizeBytes)
            .ToList();
    }

    private static string ResolveHybridRangeFamily(
        BenchmarkSnapshotRecord snapshot,
        FinalArtifactNamingContext context)
    {
        var match = context.Ranges
            .Where(x => snapshot.SizeBytes >= x.MinSizeBytes && snapshot.SizeBytes <= x.MaxSizeBytes)
            .OrderBy(x => x.MaxSizeBytes - x.MinSizeBytes)
            .FirstOrDefault();

        if (match != null)
            return match.RangeFamily;

        var nearestSmallerAnchor = context.Ranges
            .Select(x => x.SmallerHigherDamageAnchor)
            .OrderBy(x => Distance(snapshot.SizeBytes, x.SizeBytes))
            .FirstOrDefault();

        if (nearestSmallerAnchor != null)
            return NormalizeRangeFamily(nearestSmallerAnchor.Quant.BaseQuant);

        return NormalizeRangeFamily(snapshot.Quant.BaseQuant);
    }

    private static string NormalizeRangeFamily(BaselineQuants baseline)
    {
        if (!string.IsNullOrWhiteSpace(baseline.QuantizeBaseArgumentName))
            return baseline.QuantizeBaseArgumentName;

        return !baseline.Names.IsDefaultOrEmpty ? baseline.Names[0] : "Unknown";
    }

    private static string ResolveExternalProviderToken(BaselineQuants baseline)
    {
        string joined = $"{baseline.ShortSourceName} {baseline.SourceOwner} {baseline.SourceRepository} {baseline.Names[0]}";
        if (joined.Contains("unsloth", StringComparison.OrdinalIgnoreCase))
            return "UD";

        if (!string.IsNullOrWhiteSpace(baseline.ShortSourceName))
            return SanitizeToken(baseline.ShortSourceName!);

        return "EXT";
    }

    private static string NormalizeExternalDisplayName(string displayName, string providerToken)
    {
        if (string.IsNullOrWhiteSpace(displayName))
            return providerToken;

        var value = displayName.Trim();

        if (value.StartsWith("Unsloth_", StringComparison.OrdinalIgnoreCase))
            value = providerToken + value["Unsloth".Length..];

        if (!value.StartsWith(providerToken + "_", StringComparison.OrdinalIgnoreCase) &&
            !value.StartsWith(providerToken + "-", StringComparison.OrdinalIgnoreCase))
        {
            value = $"{providerToken}_{value}";
        }

        return value;
    }

    private static string MakeUniqueFileName(string desiredFileName, ISet<string>? reservedFileNames)
    {
        if (reservedFileNames == null)
            return desiredFileName;

        string candidate = desiredFileName;
        string stem = Path.GetFileNameWithoutExtension(desiredFileName);
        string ext = Path.GetExtension(desiredFileName);
        int i = 1;

        while (!reservedFileNames.Add(candidate))
        {
            i++;
            candidate = $"{stem}_{i}{ext}";
        }

        return candidate;
    }

    public static string SanitizeToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        value = value.Trim().Replace(' ', '_');
        value = UnsafeFileChars.Replace(value, "_");
        while (value.Contains("__", StringComparison.Ordinal))
            value = value.Replace("__", "_", StringComparison.Ordinal);
        return value.Trim('_', '-');
    }

    private static ulong Distance(ulong left, ulong right) => left >= right ? left - right : right - left;
}

public sealed class FinalArtifactNamingContext
{
    private readonly Dictionary<string, int> _hybridOrdinalByFamily = new(StringComparer.OrdinalIgnoreCase);

    public FinalArtifactNamingContext(IReadOnlyList<FinalArtifactRange> ranges)
    {
        Ranges = ranges;
    }

    public IReadOnlyList<FinalArtifactRange> Ranges { get; }

    public int NextHybridOrdinal(string rangeFamily)
    {
        string key = string.IsNullOrWhiteSpace(rangeFamily) ? "Unknown" : rangeFamily;
        _hybridOrdinalByFamily.TryGetValue(key, out int current);
        current++;
        _hybridOrdinalByFamily[key] = current;
        return current;
    }
}

public sealed class FinalArtifactRange
{
    public ulong MinSizeBytes { get; init; }
    public ulong MaxSizeBytes { get; init; }
    public string RangeFamily { get; init; } = string.Empty;
    public BenchmarkSnapshotRecord SmallerHigherDamageAnchor { get; init; } = default!;
    public BenchmarkSnapshotRecord LargerLowerDamageAnchor { get; init; } = default!;
}

public sealed class FinalArtifactName
{
    public string FileName { get; init; } = string.Empty;
    public string DisplayName { get; init; } = string.Empty;
    public string ProviderToken { get; init; } = string.Empty;
    public string QuantFamilyOrBaseline { get; init; } = string.Empty;
}

public sealed class ProviderCredit
{
    public string Name { get; init; } = string.Empty;
    public string Url { get; init; } = string.Empty;
    public string Note { get; init; } = string.Empty;
}
