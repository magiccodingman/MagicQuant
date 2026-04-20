using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Commands;

public class BuildHybrids : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            ShowHelp();
            return;
        }

        string? modelDirRaw = args.FirstOrDefault(a =>
            string.Equals(a.Name, "model-dir", StringComparison.OrdinalIgnoreCase))?.Value;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
            throw new InvalidOperationException("Missing required argument --model-dir.");

        string fullModelPath = Path.GetFullPath(modelDirRaw);
        if (!Directory.Exists(fullModelPath))
            throw new DirectoryNotFoundException($"The directory '{fullModelPath}' does not exist.");

        Cache.ModelDirectory = fullModelPath;
        Cache.ModelMagicQuantDirectory = Path.Combine(fullModelPath, "MagicQuant");
        Cache.UseImatrix = args.Any(a => string.Equals(a.Name, "use-imatrix", StringComparison.OrdinalIgnoreCase));
        Cache.ForceImatrixRebuild = args.Any(a => string.Equals(a.Name, "imatrix-force-rebuild", StringComparison.OrdinalIgnoreCase));
        RuntimeSearchSpace.AllowHighPrecisionHybrids = args.Any(a =>
            string.Equals(a.Name, "allow-high-precision-hybrids", StringComparison.OrdinalIgnoreCase));

        Directory.CreateDirectory(Cache.ModelMagicQuantDirectory);
        JsonHelper.DetectAndSetTorchType(Cache.ModelDirectory);
        Cache.CurrentModelId = MagicQuantModelId.GetOrCreateModelId(Cache.ModelDirectory);

        await using (var db = new MagicQuantContext())
        {
            _ = db.AiModelHashes.Count();
        }

        var pyManager = new PythonManager(Cache.MagicQuantDirectory);
        var benchmarkService = new BenchmarkService(pyManager);
        var quantizationService = new QuantizationService(benchmarkService);
        var imatrixService = new ImatrixService();
        var dbService = new QuantDatabaseService();

        var imatrixRequest = new ImatrixRequest
        {
            UseImatrix = Cache.UseImatrix,
            ForceRebuild = Cache.ForceImatrixRebuild,
            ImatrixUrl = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-url", StringComparison.OrdinalIgnoreCase))?.Value,
            DatasetRepo = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-dataset-repo", StringComparison.OrdinalIgnoreCase))?.Value,
            DatasetSplit = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-dataset-split", StringComparison.OrdinalIgnoreCase))?.Value,
            DatasetConfig = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-dataset-config", StringComparison.OrdinalIgnoreCase))?.Value,
            LocalDatasetFile = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-dataset-local-file", StringComparison.OrdinalIgnoreCase))?.Value,
            ModelDirectory = Cache.ModelDirectory!,
            MagicQuantDirectory = Cache.ModelMagicQuantDirectory!
        };

        await imatrixService.EnsureImatrixAsync(imatrixRequest);

        bool loadedPlanFromCache = await benchmarkService.TryInitializeExecutionPlanFromCacheAsync(
            quantizationKey: BaselineQuants.Q8_0.Names[0]);

        if (!loadedPlanFromCache)
        {
            var q8ModelGgufPath = await quantizationService.EnsurePureQ8ModelAsync();
            await benchmarkService.EnsureExecutionPlanAsync(q8ModelGgufPath, quantizationKey: BaselineQuants.Q8_0.Names[0]);
            await quantizationService.CleanupPureQ8ModelAsync();
        }

        await dbService.InitializeAsync();
        var remaining = await dbService.GetRemainingTensorConfigsAsync();

        if (remaining.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No remaining hybrid combinations to build.[/]");
            return;
        }

        var quants = remaining.Select(x => (MQ.DB.Models.HybridQuant)x).ToList();
        var summary = await quantizationService.ProcessHybridBatchAsync(quants);

        AnsiConsole.MarkupLine("[bold green]Build-hybrids complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Requested:[/] {summary.Requested:N0}");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {summary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped:[/] {summary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {summary.Failed:N0}");
    }

    private static void ShowHelp()
    {
        AnsiConsole.MarkupLine("[bold yellow]Command: build-hybrids[/]");
        AnsiConsole.MarkupLine("Builds/benchmarks remaining hybrid combinations from the current candidate-based search space.");
        AnsiConsole.MarkupLine("Usage: mq build-hybrids --model-dir \"<path>\" [--use-imatrix] [--allow-high-precision-hybrids]");
    }
}
