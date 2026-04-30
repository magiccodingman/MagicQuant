using System.Globalization;
using System.Text;
using MagicQuant.Models;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class ReadmeGenerationService
{
    private readonly FinalArtifactNamingService _namingService = new();

    public async Task<string> GenerateAsync(
        string outputDirectory,
        string modelName,
        IReadOnlyCollection<ExportedArtifactRecord> exportedArtifacts,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        IReadOnlyCollection<BaselineEliminationRecord>? eliminatedBaselines = null,
        BenchmarkSnapshotRecord? pplReference = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(outputDirectory);
        string readmePath = Path.Combine(outputDirectory, "README.md");

        var replacementMap = FinalReleaseMetadataService.BuildReplacementMap(eliminatedBaselines ?? Array.Empty<BaselineEliminationRecord>());
        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);
        var exportedByKey = exportedArtifacts
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Snapshot.Config), StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);

        var sb = new StringBuilder();
        AppendHuggingFaceFrontmatter(sb);

        string resolvedModelName = ResolveReadmeTitleModelName(modelName);
        sb.AppendLine($"# MagicQuant Hybrids (v2.0) - {resolvedModelName}");
        sb.AppendLine();
        sb.AppendLine("MagicQuant is a benchmark driven GGUF hybrid discovery and validation system focused on finding real, practical GGUF quants specific to each architecture.");
        sb.AppendLine();
        sb.AppendLine("Whether it's a pure baseline model built by llama.cpp, learned tensor configurations from Unsloth, or a custom built MagicQuant hybrid, the model table below shows quants that have won dominance checks, survived collapse spaces, and/or were found to be nonlinearly better. Instead of dumping every quant type possible, MagicQuant tests, validates, and brutally murders anything deemed unworthy.");
        sb.AppendLine();
        sb.AppendLine("You can learn more [from the MagicQuant Wiki](https://github.com/magiccodingman/MagicQuant-Wiki). It covers things like nonlinear winners, prediction systems, imatrix generation philosophy, isolated tensor analysis, and more.");
        sb.AppendLine();
        sb.AppendLine("By default, if an external provider like Unsloth is deemed the winner, the repo will generally link directly to the original provider instead of re-hosting the quant. External GGUFs are normally only re-uploaded when a specific winning variant does not already exist (e.g. Heretic models or similar).");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine("## Final surviving downloadable outputs");
        sb.AppendLine();
        AppendDownloadTable(sb, exportedArtifacts, replacementMap, exportedByKey, namingContext);
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();

        sb.AppendLine("## Release metadata");
        sb.AppendLine();
        sb.AppendLine("- [Final survivor metrics](./../../resolve/main/magicquant.final-survivors.json?download=true) — full file names, KLD, PPL delta %, byte sizes, download targets, and replacement lineage. PPL delta % is measured against the native/reference PPL when available; negative is better and larger positive values are worse.");
        sb.AppendLine("- [Hybrid tensor map](./../../resolve/main/magicquant.hybrid-map.json?download=true) — tensor-group assignments and effective-state details for MagicQuant hybrid GGUFs.");
        sb.AppendLine("- [Replacement details](./../../resolve/main/magicquant.replacements.json?download=true) — structured details for baselines or anchors removed from the final download table, including reason codes, KLD deltas, PPL delta %, and size deltas.");
        sb.AppendLine("- [Clone tensor configs](./../../resolve/main/magicquant.clone-configs.json?download=true) — exact per-GGUF tensor quantization maps for reproducing this final output list in repository clone mode.");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
        
        AppendReasonCodeDetails(sb);
        sb.AppendLine();

        AppendProviderCredits(sb, exportedArtifacts);
        sb.AppendLine();

        AppendCollapsible(sb, "Warning", "External/custom baselines are normalized into MagicQuant's controlled comparison flow. MagicQuant may rebuild a learned baseline under native-source / MagicQuant-controlled conditions, including its own imatrix handling, so hybrids can be judged on a more equal footing. That does **not** mean MagicQuant proved the original upstream artifact or upstream imatrix was worse. These comparisons exist for internal hybrid-search consistency, not as a universal judgment of the original creator's exact release artifact.");
        sb.AppendLine();

        sb.AppendLine("## Support");
        sb.AppendLine("I’m a solo developer working full time for myself to achieve my dream. I build open source code on the side. If you like any of my work, buying me a coffee is always appreciated. Otherwise, I hope you enjoy, maybe give me a star or something. Or just send me good vibes. Either way, thank you!");
        sb.AppendLine();
        sb.AppendLine("[Click here to see ways to support](https://sayou.biz/support) - BTC, Paypal, GitHub sponsors.");
        sb.AppendLine();
        
        await File.WriteAllTextAsync(readmePath, sb.ToString(), ct);
        AnsiConsole.MarkupLine($"[green]README generated:[/] {Markup.Escape(readmePath)}");
        return readmePath;
    }


    private static void AppendHuggingFaceFrontmatter(StringBuilder sb)
    {
        var entries = OrderedFrontmatterEntries().ToList();
        if (entries.Count == 0)
            return;

        sb.AppendLine("---");
        foreach (var (key, value) in entries)
        {
            if (TryGetSequence(value, out var values))
            {
                var rendered = values
                    .Where(x => !IsEmptyFrontmatterValue(x))
                    .Select(FormatYamlScalar)
                    .Where(x => !string.IsNullOrWhiteSpace(x))
                    .ToList();

                if (rendered.Count == 0)
                    continue;

                sb.AppendLine($"{key}:");
                foreach (var item in rendered)
                    sb.AppendLine($"- {item}");
            }
            else
            {
                if (IsEmptyFrontmatterValue(value))
                    continue;

                sb.AppendLine($"{key}: {FormatYamlScalar(value)}");
            }
        }

        sb.AppendLine("---");
        sb.AppendLine();
    }

    private static IEnumerable<KeyValuePair<string, object?>> OrderedFrontmatterEntries()
    {
        var frontmatter = Config.Current.Readme.Frontmatter;
        if (frontmatter == null || frontmatter.Count == 0)
            yield break;

        if (frontmatter.TryGetValue("license", out var license) && !IsEmptyFrontmatterValue(license))
            yield return new KeyValuePair<string, object?>("license", license);

        foreach (var entry in frontmatter)
        {
            if (string.IsNullOrWhiteSpace(entry.Key) ||
                string.Equals(entry.Key, "license", StringComparison.OrdinalIgnoreCase) ||
                IsEmptyFrontmatterValue(entry.Value))
            {
                continue;
            }

            yield return new KeyValuePair<string, object?>(entry.Key.Trim(), entry.Value);
        }
    }

    private static string ResolveReadmeTitleModelName(string fallbackModelName)
    {
        if (!string.IsNullOrWhiteSpace(Config.Current.Readme.TitleModelNameOverride))
            return Config.Current.Readme.TitleModelNameOverride.Trim();

        if (!string.IsNullOrWhiteSpace(Config.Current.Identity.ArchitectureFamilyName))
            return Config.Current.Identity.ArchitectureFamilyName.Trim();

        if (!string.IsNullOrWhiteSpace(Cache.CurrentArchitectureFamilyName))
            return Cache.CurrentArchitectureFamilyName.Trim();

        return string.IsNullOrWhiteSpace(fallbackModelName) ? "model" : fallbackModelName.Trim();
    }

    private static bool TryGetSequence(object? value, out IReadOnlyList<object?> values)
    {
        values = Array.Empty<object?>();

        if (value is string || value == null)
            return false;

        if (value is System.Collections.IEnumerable sequence)
        {
            values = sequence.Cast<object?>().ToList();
            return true;
        }

        return false;
    }

    private static bool IsEmptyFrontmatterValue(object? value)
    {
        if (value == null)
            return true;

        if (value is string text)
            return string.IsNullOrWhiteSpace(text);

        if (TryGetSequence(value, out var values))
            return values.All(IsEmptyFrontmatterValue);

        return false;
    }

    private static string FormatYamlScalar(object? value)
    {
        if (value == null)
            return string.Empty;

        if (value is bool boolean)
            return boolean ? "true" : "false";

        if (value is IFormattable formattable && value is not string)
            return formattable.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty;

        string text = value.ToString() ?? string.Empty;
        if (!NeedsYamlQuotes(text))
            return text;

        return "\"" + text
            .Replace("\\", "\\\\", StringComparison.Ordinal)
            .Replace("\"", "\\\"", StringComparison.Ordinal)
            .Replace("\r", "\\r", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal) + "\"";
    }

    private static bool NeedsYamlQuotes(string text)
    {
        if (text.Length == 0)
            return true;

        if (!string.Equals(text, text.Trim(), StringComparison.Ordinal))
            return true;

        if (text.Contains(": ", StringComparison.Ordinal) ||
            text.Contains("#", StringComparison.Ordinal) ||
            text.Contains("\n", StringComparison.Ordinal) ||
            text.Contains("\r", StringComparison.Ordinal))
        {
            return true;
        }

        char first = text[0];
        return first is '-' or '?' or ':' or '@' or '!' or '&' or '*' or '[' or ']' or '{' or '}' or '|' or '>' or '%' or '`' or ',';
    }

    private void AppendDownloadTable(
        StringBuilder sb,
        IReadOnlyCollection<ExportedArtifactRecord> artifacts,
        IReadOnlyDictionary<string, List<BaselineEliminationRecord>> replacementMap,
        IReadOnlyDictionary<string, ExportedArtifactRecord> exportedByKey,
        FinalArtifactNamingContext namingContext)
    {
        sb.AppendLine("| Name | Provider | Quant Family | KLD | Size (GB) | Download |");
        sb.AppendLine("|---|---|---|---:|---:|---|");

        foreach (var artifact in artifacts.OrderBy(x => x.Snapshot.Kld).ThenBy(x => x.Snapshot.SizeBytes))
        {
            string key = TensorConfigIdentity.ToKey(artifact.Snapshot.Config);
            string shortName = _namingService.ToPublicArtifactShortName(
                artifact.DisplayName,
                artifact.FileName,
                artifact.ProviderName,
                artifact.BaselineFamily,
                artifact.Snapshot,
                namingContext);
            var replacements = FinalReleaseMetadataService.ResolveTransitiveReplacements(key, replacementMap);
            string nameCell = BuildNameCell(shortName, replacements, exportedByKey, namingContext);
            string sizeGb = ToGb(artifact.Snapshot.SizeBytes);
            string download = artifact.IsExternalReference
                ? $"[Link]({artifact.DownloadTarget})"
                : $"[Link](./../../resolve/main/{artifact.FileName}?download=true)";

            sb.AppendLine(
                $"| {nameCell} | {EscapePipe(artifact.ProviderName)} | {EscapePipe(artifact.BaselineFamily)} | " +
                $"{artifact.Snapshot.Kld:0.000000} | {sizeGb} | {download} |");
        }
    }

    private string BuildNameCell(
        string shortName,
        IReadOnlyList<BaselineEliminationRecord> replacements,
        IReadOnlyDictionary<string, ExportedArtifactRecord> exportedByKey,
        FinalArtifactNamingContext namingContext)
    {
        if (replacements.Count == 0)
            return EscapePipe(shortName);

        var replacedNames = replacements
            .Select(x => GetPublicShortName(x.Eliminated, exportedByKey, namingContext))
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(5)
            .ToList();

        string tooltip = replacedNames.Count == 0
            ? "Replaced one or more dominated artifacts. See magicquant.replacements.json."
            : $"Replaced: {string.Join(", ", replacedNames)}";

        if (replacements.Count > replacedNames.Count)
            tooltip += $" + {replacements.Count - replacedNames.Count} more";

        return $"[<u>{EscapePipe(shortName)}</u>](#winner-notes \"{EscapeTooltip(tooltip)}\")";
    }

    private string GetPublicShortName(
        BenchmarkSnapshotRecord snapshot,
        IReadOnlyDictionary<string, ExportedArtifactRecord> exportedByKey,
        FinalArtifactNamingContext namingContext)
    {
        string key = TensorConfigIdentity.ToKey(snapshot.Config);
        if (exportedByKey.TryGetValue(key, out var artifact))
            return _namingService.ToPublicArtifactShortName(
                artifact.DisplayName,
                artifact.FileName,
                artifact.ProviderName,
                artifact.BaselineFamily,
                artifact.Snapshot,
                namingContext);

        return _namingService.ToPublicArtifactShortName(
            _namingService.BuildDisplayLabel(snapshot, namingContext),
            null,
            snapshot.IsHybrid ? "MagicQuant" : HybridBenchmarkRepository.ResolveProviderName(snapshot.Quant, exportNaming: false),
            snapshot.BaselineFamily,
            snapshot,
            namingContext);
    }

    private static void AppendReasonCodeDetails(StringBuilder sb)
    {
        sb.AppendLine("<details>");
        sb.AppendLine("<summary>Replacement reason codes</summary>");
        sb.AppendLine();
        sb.AppendLine("- `STRICT_DOMINANCE` — the winner was no larger and had lower real KLD than the removed anchor.");
        sb.AppendLine("- `NEAR_BASELINE_PREMIUM` — the winner used only the configured near-baseline size premium and beat the real linear KLD trade line.");
        sb.AppendLine("- `INTERIOR_DISCOVERY` — the winner was selected as a useful interior point inside a size/KLD gap between anchors.");
        sb.AppendLine("- `SPACING_COLLAPSE` — two candidates were too close in practical output space, so the stronger one was kept.");
        sb.AppendLine("- `FINAL_DOMINANCE` — a later validated survivor dominated this artifact in final real benchmark comparison.");
        sb.AppendLine();
        sb.AppendLine("<a id=\"winner-notes\"></a>");
        sb.AppendLine("Underlined names in the table replaced or ultimately inherited the replacement of another artifact. Hover the name for the short replacement summary, or inspect `magicquant.replacements.json` for exact KLD/PPL/size deltas.");
        sb.AppendLine();
        sb.AppendLine("</details>");
    }

    private void AppendProviderCredits(StringBuilder sb, IReadOnlyCollection<ExportedArtifactRecord> artifacts)
    {
        var credits = _namingService.BuildProviderCredits(artifacts);
        if (credits.Count == 0)
            return;

        sb.AppendLine("<details>");
        sb.AppendLine("<summary>Provider credits</summary>");
        sb.AppendLine();

        foreach (var credit in credits)
        {
            string name = EscapePipe(credit.Name);
            string note = EscapePipe(credit.Note);

            if (!string.IsNullOrWhiteSpace(credit.Url))
                sb.AppendLine($"- [{name}]({credit.Url}) — {note}");
            else
                sb.AppendLine($"- {name} — {note}");
        }

        sb.AppendLine();
        sb.AppendLine("</details>");
    }

    private static void AppendCollapsible(StringBuilder sb, string summary, string body)
    {
        sb.AppendLine("<details>");
        sb.AppendLine($"<summary>{EscapeHtml(summary)}</summary>");
        sb.AppendLine();
        sb.AppendLine(body);
        sb.AppendLine();
        sb.AppendLine("</details>");
    }

    private static string ToGb(ulong bytes) => (bytes / 1024d / 1024d / 1024d).ToString("0.00");
    private static string EscapePipe(string value) => (value ?? string.Empty).Replace("|", "\\|");
    private static string EscapeTooltip(string value) => (value ?? string.Empty).Replace("\"", "&quot;").Replace("|", " ");
    private static string EscapeHtml(string value) => (value ?? string.Empty).Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");
}