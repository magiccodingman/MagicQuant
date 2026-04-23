using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class CombinationSurvivalPipelineService
{
    private readonly QuantizationService _quantizationService;
    private readonly RemainingCombinationStore _combinationStore;
    private readonly HybridBenchmarkRepository _benchmarkRepository;
    private readonly EffectiveCandidateStateResolverService _effectiveResolver;
    private readonly PredictedCandidateEvaluationService _predictionService;
    private readonly BitRangeBucketBuilderService _bucketBuilder;
    private readonly BucketLocalPruningService _bucketPruner;
    private readonly FinalRealBenchmarkEliminationService _finalEliminator;
    private readonly FinalSurvivorSelectionCliService _selectionCli;
    private readonly HybridArtifactExportService _exportService;
    private readonly ReadmeGenerationService _readmeService;
    private readonly HybridMapGenerationService _hybridMapService;

    public CombinationSurvivalPipelineService(QuantizationService quantizationService)
    {
        _quantizationService = quantizationService;
        _combinationStore = new RemainingCombinationStore();
        _benchmarkRepository = new HybridBenchmarkRepository();
        _effectiveResolver = new EffectiveCandidateStateResolverService(_benchmarkRepository);
        _predictionService = new PredictedCandidateEvaluationService(_benchmarkRepository, _effectiveResolver);
        _bucketBuilder = new BitRangeBucketBuilderService(_benchmarkRepository);
        _bucketPruner = new BucketLocalPruningService();
        _finalEliminator = new FinalRealBenchmarkEliminationService();
        _selectionCli = new FinalSurvivorSelectionCliService();
        _exportService = new HybridArtifactExportService(_quantizationService, _effectiveResolver);
        _readmeService = new ReadmeGenerationService();
        _hybridMapService = new HybridMapGenerationService();
    }

    public async Task<CombinationSurvivalExecutionResult> RunAsync(CancellationToken ct = default)
    {
        var report = new SurvivalStageReport();
        report.StartingCount = await _combinationStore.CountAsync(ct);

        AnsiConsole.Write(new Rule("[yellow]Prediction / Survival Pipeline[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[green]Starting remaining combinations:[/] {report.StartingCount:N0}");

        if (report.StartingCount > Config.BruteForceFinalCombinationThreshold)
        {
            var current = await _combinationStore.LoadAllAsync(ct);
            var predicted = await _predictionService.EvaluateAsync(current, ct);
            var bucketBuild = await _bucketBuilder.BuildAsync(predicted, ct);

            PrintBucketAnchors(bucketBuild.Buckets);

            var pureBaselineSnapshots = await _benchmarkRepository.LoadPureBaselineSnapshotsAsync(ct);
            AnsiConsole.MarkupLine($"[grey]Pure baseline context loaded for bucket pruning:[/] [cyan]{pureBaselineSnapshots.Count:N0}[/]");
            var bucketPruneResult = _bucketPruner.Prune(bucketBuild, pureBaselineSnapshots);
            foreach (var diag in bucketPruneResult.Diagnostics)
                report.BucketDiagnostics.Add(diag);

            var survivors = bucketPruneResult.Survivors
                .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
                .ToList();

            int beforeBalance = survivors.Count;
            if (survivors.Count > Config.BruteForceFinalCombinationThreshold)
            {
                survivors = BalanceDownToThreshold(survivors, bucketBuild, Config.BruteForceFinalCombinationThreshold);
                report.AddRemoval("bucket-balance", beforeBalance - survivors.Count);
            }

            if (survivors.Count > Config.BruteForceFinalCombinationThreshold)
            {
                var globalCut = survivors
                    .OrderBy(x => x.CompositeScore)
                    .ThenBy(x => x.PredictedKldCost)
                    .ThenBy(x => x.PredictedSizeBytes)
                    .Take(Config.BruteForceFinalCombinationThreshold)
                    .ToList();

                report.AddRemoval("stage-7-global-cut", survivors.Count - globalCut.Count);
                survivors = globalCut;
            }

            if (survivors.Count == 0)
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Survival pipeline produced zero kept candidates after bucket pruning. Check pure-baseline-shadowed / bonus-hybrid-kept diagnostics.");

            await _combinationStore.ReplaceAllAsync(survivors.Select(x => x.Config).ToList(), "prediction-survival", ct);

            foreach (var diag in report.BucketDiagnostics)
            {
                AnsiConsole.MarkupLine(
                    $"[grey]Bucket {Markup.Escape(diag.BucketKey)}:[/] anchors=({diag.LowerAnchorSizeBytes:N0}..{diag.UpperAnchorSizeBytes:N0}) " +
                    $"incoming=[cyan]{diag.IncomingCount:N0}[/] removed=[red]{diag.RemovedCount:N0}[/] kept=[green]{diag.KeptCount:N0}[/]");

                foreach (var reason in diag.RemovalReasons.OrderByDescending(x => x.Value))
                    AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(reason.Key)}:[/] {reason.Value:N0}");
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]Remaining combinations are already at or under threshold. Skipping additional predictive narrowing.[/]");
        }

        report.EndingCount = await _combinationStore.CountAsync(ct);
        AnsiConsole.MarkupLine($"[green]Combinations after survival pipeline:[/] {report.EndingCount:N0}");

        if (report.EndingCount > Config.BruteForceFinalCombinationThreshold)
        {
            throw new InvalidOperationException(
                $"Survival pipeline completed but still left {report.EndingCount:N0} combinations, which is above the brute-force threshold of {Config.BruteForceFinalCombinationThreshold:N0}. Diagnostics were emitted above.");
        }

        AnsiConsole.Write(new Rule("[yellow]Final Brute Force Benchmark Phase[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine(
            $"[green]Remaining combination count[/] [cyan]{report.EndingCount:N0}[/] [grey]is at or below the brute-force threshold of[/] [yellow]{Config.BruteForceFinalCombinationThreshold:N0}[/].");

        var finalConfigs = await _combinationStore.LoadAllAsync(ct);
        var finalQuants = finalConfigs.Select(x => (HybridQuant)x).ToList();
        var finalSummary = await _quantizationService.ProcessHybridBatchAsync(finalQuants);

        AnsiConsole.MarkupLine("[bold green]Final brute force benchmarking complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Requested:[/] {finalSummary.Requested:N0}");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {finalSummary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {finalSummary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {finalSummary.Failed:N0}");

        var benchmarkSnapshots = (await _benchmarkRepository.LoadBenchmarkSnapshotsAsync(finalConfigs, ct)).Values.ToList();
        var pureBaselines = await _benchmarkRepository.LoadPureBaselineSnapshotsAsync(ct);

        var brutalInput = benchmarkSnapshots
            .Concat(pureBaselines)
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal)
            .Select(g => g.First())
            .ToList();

        var brutal = _finalEliminator.Eliminate(brutalInput);
        AnsiConsole.MarkupLine($"[green]Final brutal elimination removals:[/] [red]{brutal.Eliminated.Count:N0}[/]");

        var selectedRows = _selectionCli.Prompt(brutal.Survivors);

        var exportedArtifacts = await _exportService.ExportAsync(selectedRows, ct);

        string modelName = string.IsNullOrWhiteSpace(Cache.ModelDirectory)
            ? "model"
            : new DirectoryInfo(Cache.ModelDirectory!).Name;

        await _readmeService.GenerateAsync(Cache.OutputDirectory!, modelName, exportedArtifacts, brutalInput, ct);
        await _hybridMapService.GenerateAsync(Cache.OutputDirectory!, exportedArtifacts, ct);

        return new CombinationSurvivalExecutionResult
        {
            BenchmarkedSnapshots = benchmarkSnapshots,
            BrutalSurvivors = brutal.Survivors,
            SelectedRows = selectedRows,
            ExportedArtifacts = exportedArtifacts,
            BucketDiagnostics = report.BucketDiagnostics,
            SurvivalReport = report
        };
    }

    private static void PrintBucketAnchors(IReadOnlyList<BitRangeBucketDefinition> buckets)
    {
        foreach (var bucket in buckets)
        {
            AnsiConsole.MarkupLine(
                $"[grey]BitRange bucket {Markup.Escape(bucket.Key)}[/] -> lower_anchor=[cyan]{bucket.LowerAnchorSizeBytes:N0}[/] upper_anchor=[cyan]{bucket.UpperAnchorSizeBytes:N0}[/]");
        }
    }

    private static List<PredictedCandidateEvaluation> BalanceDownToThreshold(
        IReadOnlyList<PredictedCandidateEvaluation> survivors,
        BitRangeBucketBuildResult bucketBuild,
        int threshold)
    {
        var byBucket = bucketBuild.Buckets
            .ToDictionary(
                bucket => bucket.Key,
                bucket => survivors
                    .Where(x => bucketBuild.BucketedCandidates.Any(bc => bc.Bucket.Key == bucket.Key && TensorConfigIdentity.ToKey(bc.Evaluation.Config) == TensorConfigIdentity.ToKey(x.Config)))
                    .OrderBy(x => x.CompositeScore)
                    .ThenBy(x => x.PredictedKldCost)
                    .ThenBy(x => x.PredictedSizeBytes)
                    .ToList(),
                StringComparer.Ordinal);

        var fallback = survivors
            .Where(x => !bucketBuild.BucketedCandidates.Any(bc => TensorConfigIdentity.ToKey(bc.Evaluation.Config) == TensorConfigIdentity.ToKey(x.Config)))
            .OrderBy(x => x.CompositeScore)
            .ThenBy(x => x.PredictedKldCost)
            .ThenBy(x => x.PredictedSizeBytes)
            .ToList();

        var balanced = new List<PredictedCandidateEvaluation>(threshold);
        int pass = 0;
        while (balanced.Count < threshold)
        {
            bool addedAny = false;

            foreach (var bucket in byBucket.OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                if (pass >= bucket.Value.Count)
                    continue;

                balanced.Add(bucket.Value[pass]);
                addedAny = true;

                if (balanced.Count >= threshold)
                    break;
            }

            if (!addedAny)
                break;

            pass++;
        }

        foreach (var item in fallback)
        {
            if (balanced.Count >= threshold)
                break;

            if (balanced.Any(x => TensorConfigIdentity.ToKey(x.Config) == TensorConfigIdentity.ToKey(item.Config)))
                continue;

            balanced.Add(item);
        }

        return balanced
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .Take(threshold)
            .ToList();
    }
}