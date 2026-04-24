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

        var selectedRows = _selectionCli.Prompt(selection.Survivors);

        var exportedArtifacts = await _exportService.ExportAsync(selectedRows, ct);

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

        await _readmeService.GenerateAsync(
            Cache.OutputDirectory!,
            modelName,
            exportedArtifacts,
            benchmarkOverview,
            selection.Eliminations,
            selection.ValidationFailures,
            ct);

        await _hybridMapService.GenerateAsync(Cache.OutputDirectory!, exportedArtifacts, ct);

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
}
