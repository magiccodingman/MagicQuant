using System.Text;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class ReadmeGenerationService
{
    public async Task<string> GenerateAsync(
        string outputDirectory,
        string modelName,
        IReadOnlyCollection<ExportedArtifactRecord> exportedArtifacts,
        IReadOnlyCollection<BenchmarkSnapshotRecord> benchmarkOverview,
        IReadOnlyCollection<BaselineEliminationRecord>? eliminatedBaselines = null,
        IReadOnlyCollection<CandidateValidationResult>? validationFailures = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(outputDirectory);
        string readmePath = Path.Combine(outputDirectory, "README.md");

        var sb = new StringBuilder();
        sb.AppendLine($"# MagicQuant Hybrids (v2.0) - {modelName}");
        sb.AppendLine();
        sb.AppendLine("MagicQuant is **not** a quantization technique by itself.");
        sb.AppendLine();
        sb.AppendLine("It is a search, judging, and hybrid-discovery system that learns from baseline families such as llama.cpp and external/custom baseline sources, then uses isolated empirical truth, rank-safe prediction, and real benchmarking to keep the practical survivors.");
        sb.AppendLine();
        sb.AppendLine("Sometimes a hybrid beats a pure baseline. Sometimes it does not. That is normal. The point is to pay the real benchmarking cost only where the trade looks genuinely worth it.");
        sb.AppendLine();

        sb.AppendLine("## Final surviving downloadable outputs");
        sb.AppendLine();
        AppendDownloadTable(sb, exportedArtifacts);
        sb.AppendLine();

        if (eliminatedBaselines is { Count: > 0 })
        {
            sb.AppendLine("## Baselines / anchors removed from final download table");
            sb.AppendLine();
            sb.AppendLine("These rows are intentionally **not** part of the primary download table. They explain which pure baselines or previously-surviving anchors were beaten by another validated artifact.");
            sb.AppendLine();
            AppendEliminationTable(sb, eliminatedBaselines);
            sb.AppendLine();
        }

        if (validationFailures is { Count: > 0 })
        {
            sb.AppendLine("## Predicted candidates that did not validate");
            sb.AppendLine();
            sb.AppendLine("The prediction engine is used for choosing what is worth building, but final survival still requires real benchmark validation. These candidates were predicted as interesting, built or checked, and then rejected because the real relationship did not hold.");
            sb.AppendLine();
            AppendValidationFailureTable(sb, validationFailures);
            sb.AppendLine();
        }

        sb.AppendLine("## Benchmark overview");
        sb.AppendLine();
        AppendBenchmarkOverviewTable(sb, benchmarkOverview);
        sb.AppendLine();

        sb.AppendLine("## Method note");
        sb.AppendLine();
        sb.AppendLine("The final chooser uses rank-safe isolation prediction: Q8-carrier single-group isolation measurements provide the additive backbone, a low-bit interaction correction improves numeric KLD closeness, and an isotonic projection keeps the final predicted ordering monotone with the isolation backbone. Predicted candidates still have to validate against real benchmark truth before they can replace a baseline or remain as an interior hybrid.");
        sb.AppendLine();

        sb.AppendLine("## Dive Deeper");
        sb.AppendLine();
        sb.AppendLine("- Browse the project GitHub/Wiki for benchmark methodology, architecture notes, and planned pipeline improvements.");
        sb.AppendLine("- If you spot a mistake, edge case, or a better practical trade, open an issue or share the artifact details so the comparison can be improved.");
        sb.AppendLine();

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

    private static void AppendDownloadTable(StringBuilder sb, IReadOnlyCollection<ExportedArtifactRecord> artifacts)
    {
        sb.AppendLine("| Name | Provider | Quant Family / Baseline | KLD | PPL | Size (GB) | Download |");
        sb.AppendLine("|---|---|---|---:|---:|---:|---|");

        foreach (var artifact in artifacts.OrderBy(x => x.Snapshot.Kld).ThenBy(x => x.Snapshot.SizeBytes))
        {
            string sizeGb = (artifact.Snapshot.SizeBytes / 1024d / 1024d / 1024d).ToString("0.00");
            string download = artifact.IsExternalReference
                ? $"[Link]({artifact.DownloadTarget})"
                : $"[Link](./../../resolve/main/{artifact.FileName}?download=true)";

            sb.AppendLine($"| {EscapePipe(artifact.DisplayName)} | {EscapePipe(artifact.ProviderName)} | {EscapePipe(artifact.BaselineFamily)} | {artifact.Snapshot.Kld:0.000000} | {artifact.Snapshot.Ppl:0.0000} | {sizeGb} | {download} |");
        }
    }

    private static void AppendEliminationTable(StringBuilder sb, IReadOnlyCollection<BaselineEliminationRecord> eliminations)
    {
        sb.AppendLine("| Removed | Removed KLD | Removed Size (GB) | Winner | Winner KLD | Winner Size (GB) | Reason |");
        sb.AppendLine("|---|---:|---:|---|---:|---:|---|");

        foreach (var row in eliminations
                     .DistinctBy(x => $"{TensorConfigIdentity.ToKey(x.Eliminated.Config)}::{TensorConfigIdentity.ToKey(x.Eliminator.Config)}::{x.Reason}")
                     .OrderBy(x => x.Eliminated.Kld)
                     .ThenBy(x => x.Eliminated.SizeBytes))
        {
            sb.AppendLine(
                $"| {EscapePipe(row.Eliminated.DisplayName)} | {row.Eliminated.Kld:0.000000} | {ToGb(row.Eliminated.SizeBytes)} | " +
                $"{EscapePipe(row.Eliminator.DisplayName)} | {row.Eliminator.Kld:0.000000} | {ToGb(row.Eliminator.SizeBytes)} | {EscapePipe(row.Reason)} |");
        }
    }

    private static void AppendValidationFailureTable(StringBuilder sb, IReadOnlyCollection<CandidateValidationResult> failures)
    {
        sb.AppendLine("| Candidate | Reason | Predicted KLD | Predicted Size (GB) | Actual KLD | Actual Size (GB) | Message |");
        sb.AppendLine("|---|---|---:|---:|---:|---:|---|");

        foreach (var failure in failures
                     .Where(x => !x.Accepted)
                     .OrderBy(x => x.Candidate.Reason)
                     .ThenBy(x => x.Candidate.Prediction.PredictedKld)
                     .Take(100))
        {
            var actualKld = failure.Snapshot == null ? "n/a" : failure.Snapshot.Kld.ToString("0.000000");
            var actualSize = failure.Snapshot == null ? "n/a" : ToGb(failure.Snapshot.SizeBytes);
            sb.AppendLine(
                $"| {EscapePipe(failure.Candidate.Prediction.Quant.BaseQuant.Names[0])} | {failure.Candidate.Reason} | " +
                $"{failure.Candidate.Prediction.PredictedKld:0.000000} | {ToGb(failure.Candidate.Prediction.PredictedSizeBytes)} | " +
                $"{actualKld} | {actualSize} | {EscapePipe(failure.Message)} |");
        }
    }

    private static void AppendBenchmarkOverviewTable(StringBuilder sb, IReadOnlyCollection<BenchmarkSnapshotRecord> snapshots)
    {
        sb.AppendLine("| Name | Provider | Quant Family | KLD | PPL | Size (GB) |");
        sb.AppendLine("|---|---|---|---:|---:|---:|");

        foreach (var snap in snapshots
                     .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
                     .OrderBy(x => x.Kld)
                     .ThenBy(x => x.SizeBytes))
        {
            sb.AppendLine($"| {EscapePipe(snap.DisplayName)} | {EscapePipe(snap.ProviderName)} | {EscapePipe(snap.BaselineFamily)} | {snap.Kld:0.000000} | {snap.Ppl:0.0000} | {ToGb(snap.SizeBytes)} |");
        }
    }

    private static string ToGb(ulong bytes) => (bytes / 1024d / 1024d / 1024d).ToString("0.00");
    private static string EscapePipe(string value) => (value ?? string.Empty).Replace("|", "\\|");
}
