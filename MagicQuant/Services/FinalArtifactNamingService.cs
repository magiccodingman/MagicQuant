using System.Text.RegularExpressions;
using MagicQuant.Models;
using MQ.DB.Models;

namespace MagicQuant.Services;

/// <summary>
/// Centralizes the public naming rules used by exported GGUF files, README rows,
/// CLI previews, links, and diagnostic logs. Internal tensor-combo display names stay internal.
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
        string prefix = ResolveModelPrefix();

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
        else if (HybridBenchmarkRepository.ResolveSourceBaselineForProvider(snapshot.Quant).IsExternalRepositoryBaseline)
        {
            var sourceBaseline = HybridBenchmarkRepository.ResolveSourceBaselineForProvider(snapshot.Quant);
            string externalProviderToken = ResolveExternalProviderToken(sourceBaseline);
            string externalFamily = NormalizeExternalDisplayName(sourceBaseline.Names[0], externalProviderToken);

            providerToken = externalProviderToken;
            if (Config.ExportExternalLearnedBaselines || snapshot.IsExternalRebuiltBaseline || snapshot.IsMaterializedTensorMapped)
            {
                // This is a MagicQuant rebuilt/re-uploaded copy of an external learned baseline.
                // The artifact name gets an MQ prefix, but the provider remains the upstream source.
                quantFamily = $"MQ-{SanitizeToken(externalFamily)}";
                tag = quantFamily;
            }
            else
            {
                quantFamily = SanitizeToken(externalFamily);
                tag = quantFamily;
            }
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
            ShortDisplayName = ToShortDisplayName(Path.GetFileNameWithoutExtension(fileName)),
            ProviderToken = providerToken,
            QuantFamilyOrBaseline = quantFamily
        };
    }

    public string BuildDisplayLabel(
        BenchmarkSnapshotRecord snapshot,
        FinalArtifactNamingContext context)
    {
        string prefix = ResolveModelPrefix();

        if (snapshot.IsHybrid)
            return $"{prefix}-MQ-{SanitizeToken(ResolveHybridRangeFamily(snapshot, context))}";

        var sourceBaseline = HybridBenchmarkRepository.ResolveSourceBaselineForProvider(snapshot.Quant);
        if (sourceBaseline.IsExternalRepositoryBaseline)
        {
            string providerToken = ResolveExternalProviderToken(sourceBaseline);
            string family = SanitizeToken(NormalizeExternalDisplayName(sourceBaseline.Names[0], providerToken));
            return Config.ExportExternalLearnedBaselines || snapshot.IsExternalRebuiltBaseline || snapshot.IsMaterializedTensorMapped
                ? $"{prefix}-MQ-{family}"
                : $"{prefix}-{family}";
        }

        return $"{prefix}-LM-{SanitizeToken(snapshot.Quant.BaseQuant.Names[0])}";
    }

    public string ToShortDisplayName(string displayNameOrFileName)
    {
        if (string.IsNullOrWhiteSpace(displayNameOrFileName))
            return string.Empty;

        string value = displayNameOrFileName.Trim();

        // Only strip directories. Do NOT blindly call GetFileNameWithoutExtension on
        // extensionless display names like Qwen3.6-35B-A3B-LM-Q8_0, because .NET will
        // treat ".6-35B-A3B-LM-Q8_0" as the extension and return only "Qwen3".
        value = Path.GetFileName(value);

        // Only remove the extension when it is a real GGUF artifact filename.
        if (value.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            value = value[..^".gguf".Length];

        string prefix = ResolveModelPrefix();
        string fullPrefix = prefix + "-";

        if (value.StartsWith(fullPrefix, StringComparison.OrdinalIgnoreCase))
            return value[fullPrefix.Length..];

        return value;
    }

    public string ToPublicArtifactShortName(
        string? displayNameOrFileName,
        string? fileName = null,
        string? providerName = null,
        string? quantFamily = null,
        BenchmarkSnapshotRecord? snapshot = null,
        FinalArtifactNamingContext? context = null)
    {
        string prefix = ResolveModelPrefix();
        string? preferred = !string.IsNullOrWhiteSpace(fileName) ? fileName : displayNameOrFileName;

        if (!string.IsNullOrWhiteSpace(preferred))
        {
            string candidate = Path.GetFileNameWithoutExtension(preferred.Trim());
            string fullPrefix = prefix + "-";

            if (candidate.StartsWith(fullPrefix, StringComparison.OrdinalIgnoreCase))
                candidate = candidate[fullPrefix.Length..];

            if (!LooksLikeModelOnlyLabel(candidate, prefix))
                return candidate;
        }

        return BuildProviderQuantFallback(fileName, providerName, quantFamily, snapshot, context);
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

    public static string ReasonCode(string reason)
    {
        if (reason.Contains("strict", StringComparison.OrdinalIgnoreCase))
            return "STRICT_DOMINANCE";
        if (reason.Contains("near-baseline", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("size premium", StringComparison.OrdinalIgnoreCase))
            return "NEAR_BASELINE_PREMIUM";
        if (reason.Contains("interior", StringComparison.OrdinalIgnoreCase))
            return "INTERIOR_DISCOVERY";
        if (reason.Contains("spacing", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("collapse", StringComparison.OrdinalIgnoreCase))
            return "SPACING_COLLAPSE";
        if (reason.Contains("dominance", StringComparison.OrdinalIgnoreCase))
            return "FINAL_DOMINANCE";

        return "VALIDATED_REPLACEMENT";
    }

    public static string ReasonDescription(string code)
    {
        return code switch
        {
            "STRICT_DOMINANCE" => "The winner was no larger and had lower real KLD than the removed anchor.",
            "NEAR_BASELINE_PREMIUM" => "The winner used only the configured near-baseline size premium and beat the real linear KLD trade line.",
            "INTERIOR_DISCOVERY" => "The winner was selected as a useful interior point inside a size/KLD gap between anchors.",
            "SPACING_COLLAPSE" => "Two candidates were too close in practical output space; the stronger one was kept.",
            "FINAL_DOMINANCE" => "A later validated survivor dominated this artifact in final real benchmark comparison.",
            _ => "A validated survivor replaced or made this artifact redundant."
        };
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

        return SanitizeToken(value);
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

    private static string ResolveModelPrefix()
    {
        string prefix = SanitizeToken(Config.OutputNamePrefix);
        return string.IsNullOrWhiteSpace(prefix) ? "Model" : prefix;
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

    private static bool LooksLikeModelOnlyLabel(string value, string modelPrefix)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;

        string normalized = value.Trim();
        if (string.Equals(normalized, modelPrefix, StringComparison.OrdinalIgnoreCase))
            return true;

        string alnum = new(normalized.Where(char.IsLetterOrDigit).ToArray());
        if (string.IsNullOrWhiteSpace(alnum))
            return true;

        return Regex.IsMatch(alnum, @"^[A-Za-z]+\d*(?:\d)?$", RegexOptions.IgnoreCase);
    }

    private string BuildProviderQuantFallback(
        string? fileName,
        string? providerName,
        string? quantFamily,
        BenchmarkSnapshotRecord? snapshot,
        FinalArtifactNamingContext? context)
    {
        string resolvedProvider = providerName ?? (snapshot != null
            ? (snapshot.IsHybrid ? "MagicQuant" : HybridBenchmarkRepository.ResolveProviderName(snapshot.Quant, exportNaming: false))
            : string.Empty);

        string resolvedFamily = quantFamily;
        if (string.IsNullOrWhiteSpace(resolvedFamily) && snapshot != null)
            resolvedFamily = snapshot.BaselineFamily;
        if (string.IsNullOrWhiteSpace(resolvedFamily) && snapshot != null && context != null)
            resolvedFamily = ResolveHybridRangeFamily(snapshot, context);

        string sanitizedFamily = SanitizeToken(resolvedFamily ?? string.Empty);
        if (string.IsNullOrWhiteSpace(sanitizedFamily))
            return string.Empty;

        if (sanitizedFamily.StartsWith("MQ-", StringComparison.OrdinalIgnoreCase))
            return sanitizedFamily;

        if (snapshot?.IsHybrid == true)
        {
            string ordinal = ExtractOrdinalFromFileName(fileName);
            string baseName = sanitizedFamily.StartsWith("MQ-", StringComparison.OrdinalIgnoreCase)
                ? sanitizedFamily
                : $"MQ-{sanitizedFamily}";
            return string.IsNullOrWhiteSpace(ordinal) ? baseName : $"{baseName}_{ordinal}";
        }

        if (string.Equals(resolvedProvider, "MagicQuant", StringComparison.OrdinalIgnoreCase))
            return sanitizedFamily.StartsWith("MQ-", StringComparison.OrdinalIgnoreCase) ? sanitizedFamily : $"MQ-{sanitizedFamily}";

        if (string.Equals(resolvedProvider, "llama.cpp", StringComparison.OrdinalIgnoreCase))
            return sanitizedFamily.StartsWith("LM-", StringComparison.OrdinalIgnoreCase) ? sanitizedFamily : $"LM-{sanitizedFamily}";

        if (string.Equals(resolvedProvider, "Unsloth", StringComparison.OrdinalIgnoreCase))
        {
            if (sanitizedFamily.StartsWith("UD-", StringComparison.OrdinalIgnoreCase) ||
                sanitizedFamily.StartsWith("Unsloth", StringComparison.OrdinalIgnoreCase))
                return sanitizedFamily;

            return $"UD-{sanitizedFamily}";
        }

        return sanitizedFamily;
    }

    private static string ExtractOrdinalFromFileName(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return string.Empty;

        string stem = Path.GetFileNameWithoutExtension(fileName.Trim());
        var match = Regex.Match(stem, @"_(\d+)$", RegexOptions.CultureInvariant);
        return match.Success ? match.Groups[1].Value : string.Empty;
    }
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
    public string ShortDisplayName { get; init; } = string.Empty;
    public string ProviderToken { get; init; } = string.Empty;
    public string QuantFamilyOrBaseline { get; init; } = string.Empty;
}

public sealed class ProviderCredit
{
    public string Name { get; init; } = string.Empty;
    public string Url { get; init; } = string.Empty;
    public string Note { get; init; } = string.Empty;
}