using System.Globalization;
using System.Text.Json;
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
        var replacementMap = FinalReleaseMetadataService.BuildReplacementMap(eliminatedBaselines ?? Array.Empty<BaselineEliminationRecord>());
        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);
        var exportedByKey = exportedArtifacts
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Snapshot.Config), StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);

        double? referencePpl = ResolveReferencePpl(pplReference, pureBaselineSnapshots, exportedArtifacts.Select(x => x.Snapshot));

        var rows = exportedArtifacts
            .OrderBy(x => x.Snapshot.Kld)
            .ThenBy(x => x.Snapshot.SizeBytes)
            .Select(artifact =>
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
                string download = artifact.IsExternalReference
                    ? artifact.DownloadTarget
                    : MagicQuantManifestPathService.HuggingFaceGgufResolvePath(artifact.FileName ?? string.Empty);

                return new ReadmeArtifactRow
                {
                    NameCell = nameCell,
                    Provider = artifact.ProviderName,
                    QuantFamily = artifact.BaselineFamily,
                    Kld = artifact.Snapshot.Kld,
                    Ppl = artifact.Snapshot.Ppl,
                    PplDeltaPercent = FinalReleaseMetadataService.CalculatePplDeltaPercent(artifact.Snapshot.Ppl, referencePpl),
                    SizeBytes = artifact.Snapshot.SizeBytes,
                    DownloadTarget = download
                };
            })
            .ToList();

        return await GenerateCoreAsync(
            outputDirectory,
            modelName,
            rows,
            hasReplacementDetails: (eliminatedBaselines?.Count ?? 0) > 0,
            cloneContext: null,
            exportedArtifacts: exportedArtifacts,
            ct: ct);
    }

    public async Task<string> GenerateCloneAsync(
        string outputDirectory,
        string modelName,
        string sourceDescription,
        bool sourceWasHuggingFaceRepo,
        IReadOnlyCollection<CloneArtifactBuildRecord> records,
        IReadOnlyCollection<string>? archivedManifestFileNames = null,
        CancellationToken ct = default)
    {
        var cloneReplacementHints = LoadCloneReplacementHints(outputDirectory);

        var rows = records
            .OrderBy(x => x.Kld ?? double.MaxValue)
            .ThenBy(x => x.ActualSizeBytes)
            .Select(record =>
            {
                var artifact = record.ManifestArtifact;
                string rawName = string.IsNullOrWhiteSpace(artifact.ShortName)
                    ? Path.GetFileNameWithoutExtension(artifact.FileName)
                    : artifact.ShortName;
                string name = BuildCloneNameCell(rawName, artifact.FileName, cloneReplacementHints);

                return new ReadmeArtifactRow
                {
                    NameCell = name,
                    Provider = string.IsNullOrWhiteSpace(artifact.Provider) ? "Cloned config" : artifact.Provider,
                    QuantFamily = string.IsNullOrWhiteSpace(artifact.QuantFamily) ? artifact.BaseQuant : artifact.QuantFamily,
                    Kld = record.Kld,
                    Ppl = record.Ppl,
                    PplDeltaPercent = record.PplDeltaPercent,
                    SizeBytes = record.ActualSizeBytes,
                    DownloadTarget = MagicQuantManifestPathService.HuggingFaceGgufResolvePath(artifact.FileName)
                };
            })
            .ToList();

        var cloneContext = new ReadmeCloneContext
        {
            SourceDescription = sourceDescription,
            SourceWasHuggingFaceRepo = sourceWasHuggingFaceRepo,
            ArchivedManifestFileNames = archivedManifestFileNames?.Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToList()
                                        ?? new List<string>()
        };

        return await GenerateCoreAsync(
            outputDirectory,
            modelName,
            rows,
            hasReplacementDetails: cloneContext.ArchivedManifestFileNames.Contains(MagicQuantManifestPathService.ReplacementsFileName, StringComparer.OrdinalIgnoreCase),
            cloneContext: cloneContext,
            exportedArtifacts: Array.Empty<ExportedArtifactRecord>(),
            ct: ct);
    }

    private async Task<string> GenerateCoreAsync(
        string outputDirectory,
        string modelName,
        IReadOnlyCollection<ReadmeArtifactRow> rows,
        bool hasReplacementDetails,
        ReadmeCloneContext? cloneContext,
        IReadOnlyCollection<ExportedArtifactRecord> exportedArtifacts,
        CancellationToken ct)
    {
        Directory.CreateDirectory(outputDirectory);
        MagicQuantManifestPathService.EnsureManifestDirectory(outputDirectory);

        string readmePath = Path.Combine(outputDirectory, "README.md");
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

        if (cloneContext != null)
            AppendCloneNotice(sb, cloneContext);

        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine("## Final survivors");
        sb.AppendLine();
        AppendDownloadTable(sb, rows);
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();

        AppendReleaseMetadata(sb, cloneContext);
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();

        if (hasReplacementDetails)
        {
            AppendReasonCodeDetails(sb);
            sb.AppendLine();
        }

        if (cloneContext == null)
        {
            AppendProviderCredits(sb, exportedArtifacts);
            sb.AppendLine();

            AppendCollapsible(sb, "Warning", "External/custom baselines are normalized into MagicQuant's controlled comparison flow. MagicQuant may rebuild a learned baseline under native-source / MagicQuant-controlled conditions, including its own imatrix handling, so hybrids can be judged on a more equal footing. That does **not** mean MagicQuant proved the original upstream artifact or upstream imatrix was worse. These comparisons exist for internal hybrid-search consistency, not as a universal judgment of the original creator's exact release artifact.");
            sb.AppendLine();
        }

        sb.AppendLine("## Support");
        sb.AppendLine("I’m a solo developer working full time for myself to achieve my dream. I build open source code on the side. If you like any of my work, buying me a coffee is always appreciated. Otherwise, I hope you enjoy, maybe give me a star or something. Or just send me good vibes. Either way, thank you!");
        sb.AppendLine();
        sb.AppendLine("[Click here to see ways to support](https://sayou.biz/support) - BTC, Paypal, GitHub sponsors.");
        sb.AppendLine();

        await File.WriteAllTextAsync(readmePath, sb.ToString(), ct);
        AnsiConsole.MarkupLine($"[green]README generated:[/] {Markup.Escape(readmePath)}");
        return readmePath;
    }

    private static void AppendCloneNotice(StringBuilder sb, ReadmeCloneContext clone)
    {
        sb.AppendLine("## Clone notice");
        sb.AppendLine();

        string source = clone.SourceWasHuggingFaceRepo
            ? BuildHuggingFaceRepoLink(clone.SourceDescription)
            : $"`{EscapePipe(clone.SourceDescription)}`";

        sb.AppendLine($"This repository did not run through the full MagicQuant evolution/search pipeline. It is a clone of the final survivor tensor configurations from {source}, rebuilt and benchmarked locally for this model.");
        sb.AppendLine();
        sb.AppendLine("The archived MagicQuant JSON files in `magicquant-manifest/` are copied from the source release for durability. The clone benchmark JSON and the table below are from this clone run, so those metrics reflect the rebuilt outputs in this repository.");
        sb.AppendLine();
    }
    private static string BuildCloneNameCell(string rawName, string fileName, IReadOnlyDictionary<string, IReadOnlyList<string>> replacementHints)
    {
        string safeName = EscapePipe(rawName);

        if (!TryGetReplacementHint(replacementHints, fileName, out var replaced) &&
            !TryGetReplacementHint(replacementHints, rawName, out replaced))
        {
            return safeName;
        }

        var shown = replaced.Take(5).Where(x => !string.IsNullOrWhiteSpace(x)).Select(EscapePipe).ToList();
        string tooltip = shown.Count == 0
            ? $"Replaced one or more source artifacts. See {MagicQuantManifestPathService.RelativeManifestPath(MagicQuantManifestPathService.ReplacementsFileName)}."
            : $"Replaced: {string.Join(", ", shown)}";

        if (replaced.Count > shown.Count)
            tooltip += $" + {replaced.Count - shown.Count} more";

        return $"[<u>{safeName}</u>](#winner-notes \"{EscapeTooltip(tooltip)}\")";
    }

    private static bool TryGetReplacementHint(IReadOnlyDictionary<string, IReadOnlyList<string>> replacementHints, string? key, out IReadOnlyList<string> replaced)
    {
        replaced = Array.Empty<string>();
        if (string.IsNullOrWhiteSpace(key))
            return false;

        return replacementHints.TryGetValue(key.Trim(), out replaced) && replaced.Count > 0;
    }

    private static Dictionary<string, IReadOnlyList<string>> LoadCloneReplacementHints(string outputDirectory)
    {
        var output = new Dictionary<string, IReadOnlyList<string>>(StringComparer.OrdinalIgnoreCase);
        string path = MagicQuantManifestPathService.GetManifestFilePath(outputDirectory, MagicQuantManifestPathService.FinalSurvivorsFileName);
        if (!File.Exists(path))
            return output;

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
                return output;

            foreach (var row in doc.RootElement.EnumerateArray())
            {
                var replaced = ReadReplacedShortNames(row);
                if (replaced.Count == 0)
                    continue;

                AddReplacementHint(output, TryGetString(row, "fileName"), replaced);
                AddReplacementHint(output, TryGetString(row, "shortName"), replaced);
                AddReplacementHint(output, TryGetString(row, "displayName"), replaced);
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]Could not read clone replacement hints from archived final-survivors JSON:[/] {Markup.Escape(ex.Message)}");
        }

        return output;
    }

    private static List<string> ReadReplacedShortNames(JsonElement survivorRow)
    {
        if (!survivorRow.TryGetProperty("replacedArtifacts", out var replacedArtifacts) ||
            replacedArtifacts.ValueKind != JsonValueKind.Array)
        {
            return new List<string>();
        }

        return replacedArtifacts.EnumerateArray()
            .Select(x => TryGetString(x, "shortName") ?? TryGetString(x, "displayName") ?? TryGetString(x, "fileName"))
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(x => x!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static void AddReplacementHint(Dictionary<string, IReadOnlyList<string>> output, string? key, IReadOnlyList<string> replaced)
    {
        if (string.IsNullOrWhiteSpace(key) || replaced.Count == 0)
            return;

        output[key.Trim()] = replaced;
    }

    private static string? TryGetString(JsonElement row, string propertyName)
    {
        return row.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
    }


    private static string BuildHuggingFaceRepoLink(string repoId)
    {
        string clean = (repoId ?? string.Empty).Trim().Trim('/');
        if (string.IsNullOrWhiteSpace(clean))
            return "the source Hugging Face repository";

        return $"[{EscapePipe(clean)}](https://huggingface.co/{clean})";
    }

    private static void AppendReleaseMetadata(StringBuilder sb, ReadmeCloneContext? cloneContext)
    {
        sb.AppendLine("## Release metadata");
        sb.AppendLine();
        if (ShouldLinkManifestFile(cloneContext, MagicQuantManifestPathService.FinalSurvivorsFileName))
            sb.AppendLine($"- [Final survivor metrics]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.FinalSurvivorsFileName)}) — full file names, KLD, PPL, PPL delta %, byte sizes, download targets, and replacement lineage. PPL delta % is measured against the native/reference PPL when available; negative is better and larger positive values are worse.");

        if (ShouldLinkManifestFile(cloneContext, MagicQuantManifestPathService.HybridMapFileName))
            sb.AppendLine($"- [Hybrid tensor map]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.HybridMapFileName)}) — tensor-group assignments and effective-state details for MagicQuant hybrid GGUFs.");

        if (ShouldLinkManifestFile(cloneContext, MagicQuantManifestPathService.ReplacementsFileName))
            sb.AppendLine($"- [Replacement details]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.ReplacementsFileName)}) — structured details for baselines or anchors removed from the final download table, including reason codes, KLD deltas, PPL delta %, and size deltas.");

        sb.AppendLine($"- [Clone tensor configs]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.CloneConfigsFileName)}) — exact per-GGUF tensor quantization maps for reproducing this final output list in repository clone mode.");

        if (ShouldLinkManifestFile(cloneContext, MagicQuantManifestPathService.IsolationSamplesFileName))
            sb.AppendLine($"- [Isolation samples]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.IsolationSamplesFileName)}) — isolated base/group probe samples with KLD, PPL, PPL delta %, and size truth.");

        if (ShouldLinkManifestFile(cloneContext, MagicQuantManifestPathService.BadTradesFileName))
            sb.AppendLine($"- [Bad trade details]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.BadTradesFileName)}) — structured bad-trade pruning decisions from the isolation optimizer.");

        if (cloneContext != null)
            sb.AppendLine($"- [Clone benchmark summary]({MagicQuantManifestPathService.HuggingFaceResolvePath(MagicQuantManifestPathService.CloneBenchmarksFileName)}) — fresh benchmark results from this clone run.");
    }

    private static bool ShouldLinkManifestFile(ReadmeCloneContext? cloneContext, string fileName)
    {
        return cloneContext == null ||
               cloneContext.ArchivedManifestFileNames.Contains(fileName, StringComparer.OrdinalIgnoreCase) ||
               string.Equals(fileName, MagicQuantManifestPathService.CloneConfigsFileName, StringComparison.OrdinalIgnoreCase) ||
               string.Equals(fileName, MagicQuantManifestPathService.CloneBenchmarksFileName, StringComparison.OrdinalIgnoreCase);
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

    private static void AppendDownloadTable(StringBuilder sb, IReadOnlyCollection<ReadmeArtifactRow> rows)
    {
        sb.AppendLine("| Name | Provider | Quant Family | KLD | PPL | PPL Δ % | Size (GB) | Download |");
        sb.AppendLine("|---|---|---|---:|---:|---:|---:|---|");

        foreach (var row in rows.OrderBy(x => x.Kld ?? double.MaxValue).ThenBy(x => x.SizeBytes))
        {
            string kld = row.Kld.HasValue ? row.Kld.Value.ToString("0.000000", CultureInfo.InvariantCulture) : "n/a";
            string ppl = row.Ppl.HasValue ? row.Ppl.Value.ToString("0.000000", CultureInfo.InvariantCulture) : "n/a";
            string pplDelta = row.PplDeltaPercent.HasValue ? row.PplDeltaPercent.Value.ToString("0.000", CultureInfo.InvariantCulture) + "%" : "n/a";
            string sizeGb = ToGB(row.SizeBytes);
            string download = string.IsNullOrWhiteSpace(row.DownloadTarget) ? "n/a" : $"[Link]({row.DownloadTarget})";

            sb.AppendLine(
                $"| {row.NameCell} | {EscapePipe(row.Provider)} | {EscapePipe(row.QuantFamily)} | {kld} | {ppl} | {pplDelta} | {sizeGb} | {download} |");
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
            ? $"Replaced one or more dominated artifacts. See {MagicQuantManifestPathService.RelativeManifestPath(MagicQuantManifestPathService.ReplacementsFileName)}."
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
        sb.AppendLine($"Underlined names in the table replaced or ultimately inherited the replacement of another artifact. Hover the name for the short replacement summary, or inspect `{MagicQuantManifestPathService.RelativeManifestPath(MagicQuantManifestPathService.ReplacementsFileName)}` for exact KLD/PPL/size deltas.");
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

    private static double? ResolveReferencePpl(
        BenchmarkSnapshotRecord? pplReference,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        IEnumerable<BenchmarkSnapshotRecord> snapshots)
    {
        if (pplReference is { Ppl: > 0d })
            return pplReference.Ppl;

        var bestPure = pureBaselineSnapshots
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .ThenByDescending(x => x.SizeBytes)
            .FirstOrDefault();

        if (bestPure != null)
            return bestPure.Ppl;

        return snapshots
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .FirstOrDefault()
            ?.Ppl;
    }

    private static string ToGB(ulong bytes) => (bytes / 1000d / 1000d / 1000d).ToString("0.00", CultureInfo.InvariantCulture);
    private static string EscapePipe(string value) => (value ?? string.Empty).Replace("|", "\\|");
    private static string EscapeTooltip(string value) => (value ?? string.Empty).Replace("\"", "&quot;").Replace("|", " ");
    private static string EscapeHtml(string value) => (value ?? string.Empty).Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

    private sealed class ReadmeArtifactRow
    {
        public string NameCell { get; init; } = string.Empty;
        public string Provider { get; init; } = string.Empty;
        public string QuantFamily { get; init; } = string.Empty;
        public double? Kld { get; init; }
        public double? Ppl { get; init; }
        public double? PplDeltaPercent { get; init; }
        public ulong SizeBytes { get; init; }
        public string DownloadTarget { get; init; } = string.Empty;
    }

    private sealed class ReadmeCloneContext
    {
        public string SourceDescription { get; init; } = string.Empty;
        public bool SourceWasHuggingFaceRepo { get; init; }
        public IReadOnlyList<string> ArchivedManifestFileNames { get; init; } = Array.Empty<string>();
    }
}
