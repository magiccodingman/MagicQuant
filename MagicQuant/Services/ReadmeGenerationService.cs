using System.Text;
using MagicQuant.Models;
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
        sb.AppendLine($"# MagicQuant Hybrids (v2.2) - {modelName}");
        sb.AppendLine();
        sb.AppendLine("MagicQuant is **not** a quantization technique by itself.");
        sb.AppendLine();
        sb.AppendLine("It is a search, judging, and hybrid-discovery system that learns from baseline families such as llama.cpp and external/custom baseline sources, then uses isolated empirical truth, rank-safe prediction, and real benchmarking to keep the practical survivors.");
        sb.AppendLine();
        sb.AppendLine("Sometimes a hybrid beats a pure baseline. Sometimes it does not. The point is to pay the real benchmarking cost only where the trade is genuinely worth keeping.");
        sb.AppendLine();

        sb.AppendLine("## Final surviving downloadable outputs");
        sb.AppendLine();
        AppendDownloadTable(sb, exportedArtifacts, replacementMap, exportedByKey, namingContext);
        sb.AppendLine();

        sb.AppendLine("## Release metadata");
        sb.AppendLine();
        sb.AppendLine("- [Final survivor metrics](./../../resolve/main/magicquant.final-survivors.json?download=true) — full file names, KLD, PPL delta %, byte sizes, download targets, and replacement lineage. PPL delta % is measured against the native/reference PPL when available; negative is better and larger positive values are worse.");
        sb.AppendLine("- [Hybrid tensor map](./../../resolve/main/magicquant.hybrid-map.json?download=true) — tensor-group assignments and effective-state details for MagicQuant hybrid GGUFs.");
        sb.AppendLine("- [Replacement details](./../../resolve/main/magicquant.replacements.json?download=true) — structured details for baselines or anchors removed from the final download table, including reason codes, KLD deltas, PPL delta %, and size deltas.");
        sb.AppendLine();
        AppendReasonCodeDetails(sb);
        sb.AppendLine();

        AppendCollapsible(sb, "Method note", "The final chooser uses rank-safe isolation prediction: Q8-carrier single-group isolation measurements provide the additive backbone, a low-bit interaction correction improves numeric KLD closeness, and an isotonic projection keeps the final predicted ordering monotone with the isolation backbone. Predicted candidates still have to validate against real benchmark truth before they can replace a baseline or remain as an interior hybrid.");
        sb.AppendLine();

        AppendProviderCredits(sb, exportedArtifacts);
        sb.AppendLine();

        AppendCollapsible(sb, "Warning", "External/custom baselines are normalized into MagicQuant's controlled comparison flow. MagicQuant may rebuild a learned baseline under native-source / MagicQuant-controlled conditions, including its own imatrix handling, so hybrids can be judged on a more equal footing. That does **not** mean MagicQuant proved the original upstream artifact or upstream imatrix was worse. These comparisons exist for internal hybrid-search consistency, not as a universal judgment of the original creator's exact release artifact.");
        sb.AppendLine();

        AppendCollapsible(sb, "Dive deeper", "Browse the project GitHub/Wiki for benchmark methodology, architecture notes, and planned pipeline improvements. If you spot a mistake, edge case, or a better practical trade, open an issue or share the artifact details so the comparison can be improved.");
        sb.AppendLine();

        AppendCollapsible(sb, "Support", "If this release helped you, a star, issue report, correction, or benchmark reproduction note is genuinely useful. Careful feedback matters more than hype, especially when a hybrid looks surprisingly good or surprisingly bad.");

        await File.WriteAllTextAsync(readmePath, sb.ToString(), ct);
        AnsiConsole.MarkupLine($"[green]README generated:[/] {Markup.Escape(readmePath)}");
        return readmePath;
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
            string shortName = _namingService.ToShortDisplayName(artifact.DisplayName);
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
            return _namingService.ToShortDisplayName(artifact.DisplayName);

        return _namingService.ToShortDisplayName(_namingService.BuildDisplayLabel(snapshot, namingContext));
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
