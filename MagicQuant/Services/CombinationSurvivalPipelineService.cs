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
    private readonly RankSafeKldPredictionService _predictionService;
    private readonly FinalRealBenchmarkEliminationService _finalEliminator;
    private readonly PredictionGuidedHybridSelectionService _selectionEngine;
    private readonly FinalSurvivorSelectionCliService _selectionCli;
    private readonly HybridArtifactExportService _exportService;
    private readonly ReadmeGenerationService _readmeService;
    private readonly HybridMapGenerationService _hybridMapService;
    private readonly SelectionDiagnosticsLogService _diagnosticsLogService;
    private readonly FinalReleaseMetadataService _releaseMetadataService;
    private readonly FinalArtifactNamingService _namingService;

    public CombinationSurvivalPipelineService(QuantizationService quantizationService)
    {
        _quantizationService = quantizationService;
        _combinationStore = new RemainingCombinationStore();
        _benchmarkRepository = new HybridBenchmarkRepository();
        _effectiveResolver = new EffectiveCandidateStateResolverService(_benchmarkRepository);
        _predictionService = new RankSafeKldPredictionService(_benchmarkRepository, _effectiveResolver);
        _finalEliminator = new FinalRealBenchmarkEliminationService();
        _selectionEngine = new PredictionGuidedHybridSelectionService(_quantizationService, _benchmarkRepository, _finalEliminator);
        _selectionCli = new FinalSurvivorSelectionCliService();
        _exportService = new HybridArtifactExportService(_quantizationService, _effectiveResolver);
        _readmeService = new ReadmeGenerationService();
        _hybridMapService = new HybridMapGenerationService();
        _diagnosticsLogService = new SelectionDiagnosticsLogService();
        _releaseMetadataService = new FinalReleaseMetadataService();
        _namingService = new FinalArtifactNamingService();
    }

    public async Task<CombinationSurvivalExecutionResult> RunAsync(CancellationToken ct = default)
    {
        var report = new SurvivalStageReport
        {
            StartingCount = await _combinationStore.CountAsync(ct)
        };

        AnsiConsole.Write(new Rule("[yellow]Rank-Safe Prediction / Hybrid Selection Pipeline[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[green]Remaining DuckDB combinations available to score:[/] [cyan]{report.StartingCount:N0}[/]");
        AnsiConsole.MarkupLine("[grey]Old MDA bucket survival is disabled. DuckDB now defines the allowed search space; rank-safe isolation prediction selects what deserves real benchmarking.[/]");

        var remainingConfigs = await _combinationStore.LoadAllAsync(ct);
        var pureBaselines = await _benchmarkRepository.LoadPureBaselineSnapshotsAsync(ct);

        if (pureBaselines.Count == 0)
            throw new InvalidOperationException("No pure baseline benchmark snapshots were available. Run the baseline/isolation phases before final hybrid selection.");

        AnsiConsole.MarkupLine($"[green]Pure baseline snapshots loaded:[/] [cyan]{pureBaselines.Count:N0}[/]");

        var predictionInput = remainingConfigs
            .Concat(pureBaselines.Select(x => x.Config))
            .DistinctBy(TensorConfigIdentity.ToKey)
            .ToList();

        var predictions = await _predictionService.PredictAsync(predictionInput, ct);

        foreach (var note in predictions.Notes)
            AnsiConsole.MarkupLine($"[grey]Prediction note:[/] {Markup.Escape(note)}");

        var selection = await _selectionEngine.RunAsync(
            predictions.PredictableRows.ToList(),
            pureBaselines,
            ct);

        report.EndingCount = selection.Survivors.Count;
        report.AddRemoval("prediction-guided-non-selected", Math.Max(0L, report.StartingCount - report.EndingCount));

        AnsiConsole.MarkupLine($"[green]Final candidate/anchor survivors before manual enablement:[/] [cyan]{selection.Survivors.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[yellow]Recorded baseline/anchor eliminations:[/] [cyan]{selection.Eliminations.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[yellow]Prediction validation misses:[/] [cyan]{selection.ValidationFailures.Count:N0}[/]");
        RenderEliminationSummary(selection.Eliminations, pureBaselines);

        var nativeReference = await _benchmarkRepository.LoadBenchmarkSnapshotAsync(
            (TensorConfig)HybridQuant.CreatePureBaseline(BaselineQuants.GetNativeQuant()),
            ct);

        var selectedRows = _selectionCli.Prompt(selection.Survivors, pureBaselines, nativeReference);

        var exportedArtifacts = await _exportService.ExportAsync(selectedRows, pureBaselines, ct);

        string modelName = string.IsNullOrWhiteSpace(Cache.ModelDirectory)
            ? "model"
            : new DirectoryInfo(Cache.ModelDirectory!).Name;

        var benchmarkOverview = selection.Survivors
            .Concat(pureBaselines)
            .Concat(selection.ValidationFailures.Select(x => x.Snapshot).OfType<BenchmarkSnapshotRecord>())
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ToList();

        await _diagnosticsLogService.WriteAsync(benchmarkOverview, selection.ValidationFailures, ct);

        await _hybridMapService.GenerateAsync(Cache.OutputDirectory!, exportedArtifacts, ct);

        await _releaseMetadataService.GenerateAsync(
            Cache.OutputDirectory!,
            exportedArtifacts,
            selection.Eliminations,
            pureBaselines,
            nativeReference,
            ct);

        await _readmeService.GenerateAsync(
            Cache.OutputDirectory!,
            modelName,
            exportedArtifacts,
            pureBaselines,
            selection.Eliminations,
            nativeReference,
            ct);

        return new CombinationSurvivalExecutionResult
        {
            BenchmarkedSnapshots = benchmarkOverview,
            BrutalSurvivors = selection.Survivors,
            SelectedRows = selectedRows,
            ExportedArtifacts = exportedArtifacts,
            BucketDiagnostics = Array.Empty<BucketPruneDiagnostics>(),
            SurvivalReport = report,
            Eliminations = selection.Eliminations,
            ValidationFailures = selection.ValidationFailures
        };
    }

    private void RenderEliminationSummary(
        IReadOnlyCollection<BaselineEliminationRecord> eliminations,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots)
    {
        if (eliminations.Count == 0)
            return;

        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);

        AnsiConsole.Write(new Rule("[yellow]Baseline / Anchor Eliminations[/]") { Justification = Justify.Left });

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Removed");
        table.AddColumn("Winner");
        table.AddColumn("KLD Δ");
        table.AddColumn("Size Δ (GB)");
        table.AddColumn("Code");

        foreach (var row in eliminations
                     .DistinctBy(x => $"{TensorConfigIdentity.ToKey(x.Eliminated.Config)}::{TensorConfigIdentity.ToKey(x.Eliminator.Config)}::{x.Reason}")
                     .OrderBy(x => x.Eliminated.Kld)
                     .ThenBy(x => x.Eliminated.SizeBytes)
                     .Take(25))
        {
            double kldDelta = row.Eliminated.Kld - row.Eliminator.Kld;
            double sizeDeltaGb = (row.Eliminated.SizeBytes - (double)row.Eliminator.SizeBytes) / 1024d / 1024d / 1024d;
            string removed = _namingService.ToShortDisplayName(_namingService.BuildDisplayLabel(row.Eliminated, namingContext));
            string winner = _namingService.ToShortDisplayName(_namingService.BuildDisplayLabel(row.Eliminator, namingContext));
            string code = FinalArtifactNamingService.ReasonCode(row.Reason);

            table.AddRow(
                Markup.Escape(removed),
                Markup.Escape(winner),
                kldDelta.ToString("0.000000"),
                sizeDeltaGb.ToString("0.00"),
                Markup.Escape(code));
        }

        AnsiConsole.Write(table);

        if (eliminations.Count > 25)
            AnsiConsole.MarkupLine($"[grey]Showing first 25 of {eliminations.Count:N0} elimination records. Full details are in magicquant.replacements.json.[/]");
    }

}
