using System.Globalization;
using System.Text;
using MagicQuant.Models;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

/// <summary>
/// Offline validator for the rank-safe isolation predictor.
/// It scores every currently benchmarked general-category combo in the active
/// scoped-model/imatrix bucket, compares predicted KLD to real KLD, and reports
/// rank/order accuracy.
/// </summary>
public sealed class PredictionValidationService
{
    private readonly HybridBenchmarkRepository _repository;
    private readonly RankSafeKldPredictionService _predictionService;

    public PredictionValidationService(
        HybridBenchmarkRepository repository,
        RankSafeKldPredictionService predictionService)
    {
        _repository = repository;
        _predictionService = predictionService;
    }

    public async Task<PredictionValidationExportResult> ExportAsync(
        string outputDirectory,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(outputDirectory);

        var actual = await _repository.LoadAllBenchmarkSnapshotsForCurrentContextAsync(
            category: (byte)BenchmarkCategory.General,
            strictImatrixContext: true,
            ct: ct);

        if (actual.Count == 0)
            throw new InvalidOperationException("No category=General benchmark snapshots were found for the active scoped model/imatrix context.");

        var predictions = await _predictionService.PredictAsync(actual.Select(x => x.Config).ToList(), ct);

        var actualByKey = actual.ToDictionary(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal);
        var rows = predictions.Rows
            .Where(x => x.IsPredictable)
            .Where(x => actualByKey.ContainsKey(TensorConfigIdentity.ToKey(x.Config)))
            .ToList();

        foreach (var row in rows)
        {
            var snap = actualByKey[TensorConfigIdentity.ToKey(row.Config)];
            row.ActualKld = snap.Kld;
            row.ActualPpl = snap.Ppl;
            row.ActualSizeBytes = snap.SizeBytes;
        }

        AssignRanks(rows);

        var summary = BuildSummary(rows);
        string csvPath = Path.Combine(outputDirectory, "prediction_validation_general.csv");
        string markdownPath = Path.Combine(outputDirectory, "prediction_validation_general.md");

        await File.WriteAllTextAsync(csvPath, BuildCsv(rows), ct);
        await File.WriteAllTextAsync(markdownPath, BuildMarkdown(summary, rows, predictions.Notes), ct);

        PrintSummary(summary, csvPath, markdownPath);

        return new PredictionValidationExportResult
        {
            Summary = summary,
            CsvPath = csvPath,
            MarkdownPath = markdownPath,
            Rows = rows
                .OrderByDescending(x => x.AbsoluteKldError)
                .ThenByDescending(x => Math.Abs((x.PredictedRank ?? 0) - (x.ActualRank ?? 0)))
                .ToList()
        };
    }

    private static void AssignRanks(IReadOnlyList<RankSafePredictionRow> rows)
    {
        int actualRank = 1;
        foreach (var row in rows.OrderBy(x => x.ActualKld).ThenBy(x => x.ActualSizeBytes ?? ulong.MaxValue))
            row.ActualRank = actualRank++;

        int predictedRank = 1;
        foreach (var row in rows.OrderBy(x => x.PredictedKld).ThenBy(x => x.PredictedSizeBytes))
            row.PredictedRank = predictedRank++;
    }

    private static RankSafeValidationSummary BuildSummary(IReadOnlyList<RankSafePredictionRow> rows)
    {
        if (rows.Count == 0)
            return new RankSafeValidationSummary();

        double mae = rows.Average(x => x.AbsoluteKldError);
        double rmse = Math.Sqrt(rows.Average(x => x.SignedKldError * x.SignedKldError));
        double maxAbs = rows.Max(x => x.AbsoluteKldError);
        double meanSigned = rows.Average(x => x.SignedKldError);

        long concordant = 0;
        long discordant = 0;
        long tiedPred = 0;

        for (int i = 0; i < rows.Count; i++)
        {
            for (int j = i + 1; j < rows.Count; j++)
            {
                double actualDiff = rows[i].ActualKld - rows[j].ActualKld;
                double predDiff = rows[i].PredictedKld - rows[j].PredictedKld;

                int actualSign = Math.Sign(actualDiff);
                int predSign = Math.Sign(predDiff);

                if (predSign == 0)
                {
                    tiedPred++;
                    continue;
                }

                if (actualSign == 0 || actualSign == predSign)
                    concordant++;
                else
                    discordant++;
            }
        }

        long denominator = concordant + discordant;
        double pairwise = denominator == 0 ? 100d : concordant * 100d / denominator;

        int ShiftWithin(int maxShift) =>
            rows.Count(x => x.ActualRank.HasValue && x.PredictedRank.HasValue &&
                            Math.Abs(x.PredictedRank.Value - x.ActualRank.Value) <= maxShift);

        return new RankSafeValidationSummary
        {
            RowCount = rows.Count,
            PredictableCount = rows.Count,
            Mae = mae,
            Rmse = rmse,
            MaxAbsoluteError = maxAbs,
            MeanSignedError = meanSigned,
            PairwiseAccuracyPercent = pairwise,
            ConcordantPairs = concordant,
            DiscordantPairs = discordant,
            TiedPredictedPairs = tiedPred,
            ExactRankMatches = ShiftWithin(0),
            WithinOneRank = ShiftWithin(1),
            WithinTwoRanks = ShiftWithin(2),
            WithinFiveRanks = ShiftWithin(5),
            WithinTenRanks = ShiftWithin(10),
            WithinTwentyRanks = ShiftWithin(20)
        };
    }

    private static string BuildCsv(IReadOnlyList<RankSafePredictionRow> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("config_key,display_name,is_hybrid,base_quant,is_size_predictable,predicted_kld,actual_kld,abs_error,signed_error,predicted_rank,actual_rank,rank_shift,predicted_size_bytes,actual_size_bytes,size_abs_error_bytes,size_abs_error_percent,predicted_ppl,actual_ppl,effective_groups,notes");

        foreach (var row in rows
                     .OrderByDescending(x => x.AbsoluteKldError)
                     .ThenByDescending(x => Math.Abs((x.PredictedRank ?? 0) - (x.ActualRank ?? 0))))
        {
            int shift = (row.PredictedRank ?? 0) - (row.ActualRank ?? 0);
            sb.Append(Csv(TensorConfigIdentity.ToKey(row.Config))).Append(',');
            sb.Append(Csv(HybridBenchmarkRepository.BuildDisplayName(row.Quant))).Append(',');
            sb.Append(row.IsHybrid ? "true" : "false").Append(',');
            sb.Append(Csv(row.Quant.BaseQuant.Names[0])).Append(',');
            sb.Append(row.IsSizePredictable ? "true" : "false").Append(',');
            sb.Append(Format(row.PredictedKld)).Append(',');
            sb.Append(Format(row.ActualKld)).Append(',');
            sb.Append(Format(row.AbsoluteKldError)).Append(',');
            sb.Append(Format(row.SignedKldError)).Append(',');
            sb.Append(row.PredictedRank?.ToString(CultureInfo.InvariantCulture) ?? "").Append(',');
            sb.Append(row.ActualRank?.ToString(CultureInfo.InvariantCulture) ?? "").Append(',');
            sb.Append(shift.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(row.PredictedSizeBytes.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(row.ActualSizeBytes?.ToString(CultureInfo.InvariantCulture) ?? "").Append(',');
            long sizeError = row.ActualSizeBytes.HasValue ? (long)row.PredictedSizeBytes - (long)row.ActualSizeBytes.Value : 0L;
            double sizeErrorPct = row.ActualSizeBytes.HasValue && row.ActualSizeBytes.Value > 0
                ? Math.Abs(sizeError) * 100d / row.ActualSizeBytes.Value
                : double.NaN;
            sb.Append(row.ActualSizeBytes.HasValue ? Math.Abs(sizeError).ToString(CultureInfo.InvariantCulture) : "").Append(',');
            sb.Append(row.ActualSizeBytes.HasValue ? Format(sizeErrorPct) : "").Append(',');
            sb.Append(Format(row.PredictedPpl)).Append(',');
            sb.Append(Format(row.ActualPpl)).Append(',');
            sb.Append(Csv(BuildEffectiveGroupSummary(row.Config))).Append(',');
            sb.Append(Csv(string.Join(" | ", row.Notes)));
            sb.AppendLine();
        }

        return sb.ToString();
    }

    private static string BuildMarkdown(
        RankSafeValidationSummary summary,
        IReadOnlyList<RankSafePredictionRow> rows,
        IReadOnlyList<string> notes)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# MagicQuant Rank-Safe Prediction Validation");
        sb.AppendLine();
        sb.AppendLine("This report compares predicted KLD to real category=General benchmark KLD for the active scoped model/imatrix bucket.");
        sb.AppendLine();

        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine("| Metric | Value |");
        sb.AppendLine("|---|---:|");
        sb.AppendLine($"| Rows | {summary.RowCount:N0} |");
        sb.AppendLine($"| MAE | {summary.Mae:0.000000} |");
        sb.AppendLine($"| RMSE | {summary.Rmse:0.000000} |");
        sb.AppendLine($"| Max abs error | {summary.MaxAbsoluteError:0.000000} |");
        sb.AppendLine($"| Mean signed error | {summary.MeanSignedError:0.000000} |");
        sb.AppendLine($"| Pairwise order accuracy | {summary.PairwiseAccuracyPercent:0.0000}% |");
        sb.AppendLine($"| Concordant pairs | {summary.ConcordantPairs:N0} |");
        sb.AppendLine($"| Discordant pairs | {summary.DiscordantPairs:N0} |");
        sb.AppendLine($"| Tied predicted pairs | {summary.TiedPredictedPairs:N0} |");

        var sizeRows = rows.Where(x => x.ActualSizeBytes.HasValue && x.IsSizePredictable).ToList();
        if (sizeRows.Count > 0)
        {
            double sizeMaePercent = sizeRows.Average(x => Math.Abs((long)x.PredictedSizeBytes - (long)x.ActualSizeBytes!.Value) * 100d / x.ActualSizeBytes!.Value);
            double sizeMaxPercent = sizeRows.Max(x => Math.Abs((long)x.PredictedSizeBytes - (long)x.ActualSizeBytes!.Value) * 100d / x.ActualSizeBytes!.Value);
            sb.AppendLine($"| Size-safe rows | {sizeRows.Count:N0} |");
            sb.AppendLine($"| Size MAE % | {sizeMaePercent:0.0000}% |");
            sb.AppendLine($"| Size max abs % | {sizeMaxPercent:0.0000}% |");
        }
        sb.AppendLine();

        sb.AppendLine("## Rank movement");
        sb.AppendLine();
        sb.AppendLine("| Window | Count | Percent |");
        sb.AppendLine("|---|---:|---:|");
        AppendRankWindow(sb, "Exact", summary.ExactRankMatches, summary.RowCount);
        AppendRankWindow(sb, "Within 1", summary.WithinOneRank, summary.RowCount);
        AppendRankWindow(sb, "Within 2", summary.WithinTwoRanks, summary.RowCount);
        AppendRankWindow(sb, "Within 5", summary.WithinFiveRanks, summary.RowCount);
        AppendRankWindow(sb, "Within 10", summary.WithinTenRanks, summary.RowCount);
        AppendRankWindow(sb, "Within 20", summary.WithinTwentyRanks, summary.RowCount);
        sb.AppendLine();

        if (notes.Count > 0)
        {
            sb.AppendLine("## Prediction notes");
            sb.AppendLine();
            foreach (var note in notes)
                sb.AppendLine($"- {note}");
            sb.AppendLine();
        }

        sb.AppendLine("## Largest size misses");
        sb.AppendLine();
        sb.AppendLine("| Name | Size Safe | Pred Size GB | Actual Size GB | Abs Error MB | Error % | Groups |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---|");
        foreach (var row in rows
                     .Where(x => x.ActualSizeBytes.HasValue)
                     .OrderByDescending(x => Math.Abs((long)x.PredictedSizeBytes - (long)x.ActualSizeBytes!.Value))
                     .Take(50))
        {
            long absBytes = Math.Abs((long)row.PredictedSizeBytes - (long)row.ActualSizeBytes!.Value);
            double pct = row.ActualSizeBytes.Value == 0 ? 0d : absBytes * 100d / row.ActualSizeBytes.Value;
            double mb = absBytes / 1024d / 1024d;
            sb.AppendLine($"| {EscapePipe(HybridBenchmarkRepository.BuildDisplayName(row.Quant))} | {(row.IsSizePredictable ? "yes" : "no")} | {ToGb(row.PredictedSizeBytes)} | {ToGb(row.ActualSizeBytes.Value)} | {mb:0.00} | {pct:0.0000}% | {EscapePipe(BuildEffectiveGroupSummary(row.Config))} |");
        }
        sb.AppendLine();

        sb.AppendLine("## Largest KLD misses");
        sb.AppendLine();
        sb.AppendLine("| Name | Predicted KLD | Actual KLD | Abs Error | Pred Rank | Actual Rank | Shift | Size Pred GB | Size Actual GB | Groups |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");

        foreach (var row in rows
                     .OrderByDescending(x => x.AbsoluteKldError)
                     .ThenByDescending(x => Math.Abs((x.PredictedRank ?? 0) - (x.ActualRank ?? 0)))
                     .Take(100))
        {
            int shift = (row.PredictedRank ?? 0) - (row.ActualRank ?? 0);
            sb.AppendLine(
                $"| {EscapePipe(HybridBenchmarkRepository.BuildDisplayName(row.Quant))} | {row.PredictedKld:0.000000} | {row.ActualKld:0.000000} | {row.AbsoluteKldError:0.000000} | " +
                $"{row.PredictedRank} | {row.ActualRank} | {shift:+#;-#;0} | {ToGb(row.PredictedSizeBytes)} | {ToGb(row.ActualSizeBytes ?? 0)} | {EscapePipe(BuildEffectiveGroupSummary(row.Config))} |");
        }

        return sb.ToString();
    }

    private static void PrintSummary(RankSafeValidationSummary summary, string csvPath, string markdownPath)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Validation[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[green]Rows:[/] [cyan]{summary.RowCount:N0}[/]");
        AnsiConsole.MarkupLine($"[green]MAE:[/] [cyan]{summary.Mae:0.000000}[/]  [green]RMSE:[/] [cyan]{summary.Rmse:0.000000}[/]  [green]MaxAbs:[/] [cyan]{summary.MaxAbsoluteError:0.000000}[/]");
        AnsiConsole.MarkupLine($"[green]Pairwise order accuracy:[/] [cyan]{summary.PairwiseAccuracyPercent:0.0000}%[/]  [grey]discordant={summary.DiscordantPairs:N0} tied-pred={summary.TiedPredictedPairs:N0}[/]");
        AnsiConsole.MarkupLine($"[green]CSV:[/] [blue]{Markup.Escape(csvPath)}[/]");
        AnsiConsole.MarkupLine($"[green]Markdown:[/] [blue]{Markup.Escape(markdownPath)}[/]");
    }

    private static void AppendRankWindow(StringBuilder sb, string label, int count, int total)
    {
        double pct = total == 0 ? 0d : count * 100d / total;
        sb.AppendLine($"| {label} | {count:N0} | {pct:0.00}% |");
    }

    private static string BuildEffectiveGroupSummary(TensorConfig config)
    {
        return string.Join("; ",
            RankSafeKldPredictionService.EnumerateEffectiveBaselines(config)
                .Select(x =>
                {
                    var baseline = BaselineQuants.FromId(x.EffectiveBaselineId);
                    return $"{x.Group.Name}={baseline.Names[0]}";
                }));
    }

    private static string Format(double value) =>
        double.IsNaN(value) || double.IsInfinity(value)
            ? ""
            : value.ToString("0.000000########", CultureInfo.InvariantCulture);

    private static string Csv(string value)
    {
        value ??= string.Empty;
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
            return value;

        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string ToGb(ulong bytes) => (bytes / 1024d / 1024d / 1024d).ToString("0.00", CultureInfo.InvariantCulture);
    private static string EscapePipe(string value) => (value ?? string.Empty).Replace("|", "\\|");
}
