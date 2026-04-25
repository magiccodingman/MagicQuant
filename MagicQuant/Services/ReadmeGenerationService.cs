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

        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);
        double? referencePpl = ResolveReferencePpl(pplReference, pureBaselineSnapshots, exportedArtifacts);

        var sb = new StringBuilder();
        sb.AppendLine($"# MagicQuant Hybrids (v2.1) - {modelName}");
        sb.AppendLine();
        sb.AppendLine("MagicQuant is **not** a quantization technique by itself.");
        sb.AppendLine();
        sb.AppendLine("It is a search, judging, and hybrid-discovery system that learns from baseline families such as llama.cpp and external/custom baseline sources, then uses isolated empirical truth, rank-safe prediction, and real benchmarking to keep the practical survivors.");
        sb.AppendLine();
        sb.AppendLine("Sometimes a hybrid beats a pure baseline. Sometimes it does not. The point is to pay the real benchmarking cost only where the trade is genuinely worth keeping.");
        sb.AppendLine();

        sb.AppendLine("## Final surviving downloadable outputs");
        sb.AppendLine();
        AppendDownloadTable(sb, exportedArtifacts, referencePpl);
        sb.AppendLine();
        sb.AppendLine("> **PPL Δ % note:** negative is better. Larger positive values are worse. The percentage is measured against the native/reference PPL when available; otherwise it falls back to the best available reference in this release set.");
        sb.AppendLine();

        if (eliminatedBaselines is { Count: > 0 })
        {
            sb.AppendLine("## Baselines / anchors removed from final download table");
            sb.AppendLine();
            sb.AppendLine("These rows are intentionally **not** part of the primary download table. They explain which pure baselines or previously-surviving anchors were beaten, collapsed, or made redundant by a validated artifact.");
            sb.AppendLine();
            AppendEliminationLegend(sb);
            sb.AppendLine();
            AppendEliminationTable(sb, eliminatedBaselines, exportedArtifacts, namingContext);
            sb.AppendLine();
        }

        sb.AppendLine("## Method note");
        sb.AppendLine();
        sb.AppendLine("The final chooser uses rank-safe isolation prediction: Q8-carrier single-group isolation measurements provide the additive backbone, a low-bit interaction correction improves numeric KLD closeness, and an isotonic projection keeps the final predicted ordering monotone with the isolation backbone. Predicted candidates still have to validate against real benchmark truth before they can replace a baseline or remain as an interior hybrid.");
        sb.AppendLine();

        sb.AppendLine("## Dive Deeper");
        sb.AppendLine();
        sb.AppendLine("- Browse the project GitHub/Wiki for benchmark methodology, architecture notes, and planned pipeline improvements.");
        sb.AppendLine("- If you spot a mistake, edge case, or a better practical trade, open an issue or share the artifact details so the comparison can be improved.");
        sb.AppendLine();

        AppendProviderCredits(sb, exportedArtifacts);

        sb.AppendLine("## Warning");
        sb.AppendLine();
        sb.AppendLine("External/custom baselines are normalized into MagicQuant's controlled comparison flow. MagicQuant may rebuild a learned baseline under native-source / MagicQuant-controlled conditions, including its own imatrix handling, so hybrids can be judged on a more equal footing.");
        sb.AppendLine();
        sb.AppendLine("That does **not** mean MagicQuant proved the original upstream artifact or upstream imatrix was worse. These comparisons exist for internal hybrid-search consistency, not as a universal judgment of the original creator's exact release artifact.");
        sb.AppendLine();

        sb.AppendLine("## Support");
        sb.AppendLine();
        sb.AppendLine("If this release helped you, a star, issue report, correction, or benchmark reproduction note is genuinely useful. Careful feedback matters more than hype, especially when a hybrid looks surprisingly good or surprisingly bad.");

        await File.WriteAllTextAsync(readmePath, sb.ToString(), ct);
        AnsiConsole.MarkupLine($"[green]README generated:[/] {Markup.Escape(readmePath)}");
        return readmePath;
    }

    private static void AppendDownloadTable(
        StringBuilder sb,
        IReadOnlyCollection<ExportedArtifactRecord> artifacts,
        double? referencePpl)
    {
        sb.AppendLine("| Name | Provider | Quant Family / Baseline | KLD | PPL Δ % | Size (GB) | Download |");
        sb.AppendLine("|---|---|---|---:|---:|---:|---|");

        foreach (var artifact in artifacts.OrderBy(x => x.Snapshot.Kld).ThenBy(x => x.Snapshot.SizeBytes))
        {
            string sizeGb = ToGb(artifact.Snapshot.SizeBytes);
            string download = artifact.IsExternalReference
                ? $"[Link]({artifact.DownloadTarget})"
                : $"[Link](./../../resolve/main/{artifact.FileName}?download=true)";

            sb.AppendLine(
                $"| {EscapePipe(artifact.DisplayName)} | {EscapePipe(artifact.ProviderName)} | {EscapePipe(artifact.BaselineFamily)} | " +
                $"{artifact.Snapshot.Kld:0.000000} | {FormatPplDeltaPercent(artifact.Snapshot.Ppl, referencePpl)} | {sizeGb} | {download} |");
        }
    }

    private static void AppendEliminationLegend(StringBuilder sb)
    {
        sb.AppendLine("**Reason legend:** 🏆 strict dominance, 📈 near-baseline premium, 🧩 useful interior discovery, 📏 spacing collapse, 🔪 final dominance.");
    }

    private void AppendEliminationTable(
        StringBuilder sb,
        IReadOnlyCollection<BaselineEliminationRecord> eliminations,
        IReadOnlyCollection<ExportedArtifactRecord> artifacts,
        FinalArtifactNamingContext namingContext)
    {
        var exportedByKey = artifacts
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Snapshot.Config), StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);

        sb.AppendLine("| Removed | Winner | KLD Δ | Size Δ (GB) | Why |");
        sb.AppendLine("|---|---|---:|---:|---|");

        foreach (var row in eliminations
                     .DistinctBy(x => $"{TensorConfigIdentity.ToKey(x.Eliminated.Config)}::{TensorConfigIdentity.ToKey(x.Eliminator.Config)}::{x.Reason}")
                     .OrderBy(x => x.Eliminated.Kld)
                     .ThenBy(x => x.Eliminated.SizeBytes))
        {
            string removed = GetPublicName(row.Eliminated, exportedByKey, namingContext);
            string winner = GetPublicName(row.Eliminator, exportedByKey, namingContext);

            double kldDelta = row.Eliminated.Kld - row.Eliminator.Kld;
            double sizeDeltaGb = (row.Eliminated.SizeBytes - (double)row.Eliminator.SizeBytes) / 1024d / 1024d / 1024d;

            sb.AppendLine(
                $"| {EscapePipe(removed)} | {EscapePipe(winner)} | {kldDelta:0.000000} | {sizeDeltaGb:0.00} | {ReasonEmoji(row.Reason)} |");
        }
    }

    private static string GetPublicName(
        BenchmarkSnapshotRecord snapshot,
        IReadOnlyDictionary<string, ExportedArtifactRecord> exportedByKey,
        FinalArtifactNamingContext namingContext)
    {
        string key = TensorConfigIdentity.ToKey(snapshot.Config);
        if (exportedByKey.TryGetValue(key, out var artifact))
            return artifact.DisplayName;

        return new FinalArtifactNamingService().BuildDisplayLabel(snapshot, namingContext);
    }

    private void AppendProviderCredits(StringBuilder sb, IReadOnlyCollection<ExportedArtifactRecord> artifacts)
    {
        var credits = _namingService.BuildProviderCredits(artifacts);
        if (credits.Count == 0)
            return;

        sb.AppendLine();
        sb.AppendLine("### Provider credits");
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
    }

    private static string ReasonEmoji(string reason)
    {
        if (reason.Contains("strict", StringComparison.OrdinalIgnoreCase))
            return "🏆";
        if (reason.Contains("near-baseline", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("size premium", StringComparison.OrdinalIgnoreCase))
            return "📈";
        if (reason.Contains("interior", StringComparison.OrdinalIgnoreCase))
            return "🧩";
        if (reason.Contains("spacing", StringComparison.OrdinalIgnoreCase) ||
            reason.Contains("collapse", StringComparison.OrdinalIgnoreCase))
            return "📏";
        if (reason.Contains("dominance", StringComparison.OrdinalIgnoreCase))
            return "🔪";

        return "✅";
    }

    private static double? ResolveReferencePpl(
        BenchmarkSnapshotRecord? pplReference,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        IReadOnlyCollection<ExportedArtifactRecord> artifacts)
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

        return artifacts
            .Select(x => x.Snapshot)
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .FirstOrDefault()
            ?.Ppl;
    }

    private static string FormatPplDeltaPercent(double ppl, double? referencePpl)
    {
        if (referencePpl is null or <= 0d || ppl <= 0d)
            return "n/a";

        double delta = ((ppl - referencePpl.Value) / referencePpl.Value) * 100d;
        return delta.ToString("0.000");
    }

    private static string ToGb(ulong bytes) => (bytes / 1024d / 1024d / 1024d).ToString("0.00");
    private static string EscapePipe(string value) => (value ?? string.Empty).Replace("|", "\\|");
}
