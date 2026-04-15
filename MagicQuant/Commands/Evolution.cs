using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Commands;

public class Evolution : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => a.Name?.ToLower() == "help"))
        {
            ShowEvolutionHelp();
            return;
        }

        string? modelDirRaw = args.FirstOrDefault(a => a.Name?.ToLower() == "model-dir")?.Value;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
        {
            string msg = "[red]Error:[/] Missing required argument [yellow]--model-dir[/].";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new Exception(msg);
        }

        string fullModelPath = Path.GetFullPath(modelDirRaw);

        if (!Directory.Exists(fullModelPath))
        {
            string msg = $"[red]Error:[/] The directory [yellow]'{fullModelPath}'[/] does not exist.";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new Exception(msg);
        }

        var safeTensorFiles = Directory.GetFiles(fullModelPath, "*.safetensors", SearchOption.TopDirectoryOnly);

        if (safeTensorFiles.Length == 0)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] No [yellow].safetensors[/] files found in [blue]{fullModelPath}[/].");
            AnsiConsole.MarkupLine("[grey]Please ensure this is a valid HuggingFace model directory.[/]");
            throw new Exception();
        }

        Cache.ModelDirectory = fullModelPath;
        Cache.ModelMagicQuantDirectory = Path.Combine(fullModelPath, "MagicQuant");
        JsonHelper.DetectAndSetTorchType(Cache.ModelDirectory);

        if (!Directory.Exists(Cache.ModelMagicQuantDirectory))
            Directory.CreateDirectory(Cache.ModelMagicQuantDirectory);

        AnsiConsole.MarkupLine("[green]✔ Model Directory Validated[/]");
        AnsiConsole.Write(new Rule("[yellow]Evolution Configuration[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Model Path:   [blue]{Cache.ModelDirectory}[/]");
        AnsiConsole.MarkupLine($"Output Path:  [blue]{Cache.ModelMagicQuantDirectory}[/]");
        AnsiConsole.MarkupLine($"Files Found:  [green]{safeTensorFiles.Length}[/] safe tensors");

        if (string.IsNullOrEmpty(Cache.LlamaBin))
            AnsiConsole.MarkupLine("[yellow]Warning: Llama binaries path not set in Cache. (Did Initialization run?)[/]");

        Console.WriteLine("Acquiring unique model ID...");
        Cache.CurrentModelId = MagicQuantModelId.GetOrCreateModelId(Cache.ModelDirectory);
        AnsiConsole.MarkupLine($"[green] Model ID Created/Found: {Cache.CurrentModelId}[/]");

        var pyManager = new PythonManager(Cache.MagicQuantDirectory);
        var benchmarkService = new BenchmarkService(pyManager);
        var quantizationService = new QuantizationService(benchmarkService);

        var bf16ModelGgufPath = await quantizationService.EnsureBaseModelFileAsync(true);
        var q8ModelGgufPath = await quantizationService.EnsurePureQ8ModelAsync();

        await benchmarkService.EnsureExecutionPlanAsync(q8ModelGgufPath);
        await benchmarkService.ClampStaticNglWithBaseModelAsync(bf16ModelGgufPath);

        var baseTypeName = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        var baseBenchDir = Path.Combine(Cache.ModelMagicQuantDirectory!, "Benchmarks", baseTypeName);
        var baseLogitsDir = Path.Combine(baseBenchDir, "logits");

        var baseModelQuant = HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant());

        await benchmarkService.RunAllBenchmarksAsync(
            quantConfig: baseModelQuant,
            modelPath: bf16ModelGgufPath,
            benchDir: baseBenchDir,
            klLogitsDir: baseLogitsDir,
            saveLogits: true,
            domainsOverride: new[] { "general", "code", "math" });

        var compatibilityService = new ModelCompatibilityService(pyManager);
        await compatibilityService.RunCompatibilityCheckAsync(bf16ModelGgufPath);

        CliHelpers.ValidateCombinationLogicWorks(true);

        var dbService = new QuantDatabaseService();
        await dbService.InitializeAsync();

        AnsiConsole.Write(new Rule("[yellow]Initial Isolation Startup Samples[/]") { Justification = Justify.Left });

        var initialPlan = TensorConfigGenerator.GenerateInitialIsolationSamplePlan(Cache.UnusedTensorGroups);
        AnsiConsole.MarkupLine($"[grey]Queued initial startup samples:[/] [cyan]{initialPlan.TotalCount:N0}[/]");

        var initialSummary = await quantizationService.ProcessHybridBatchAsync(initialPlan.Plans);

        AnsiConsole.MarkupLine("[bold green]Initial startup sampling complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {initialSummary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {initialSummary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {initialSummary.Failed:N0}");

        var isolationOptimizer = new IsolationOptimizationService();

        AnsiConsole.Write(new Rule("[yellow]Initial Probe Analysis[/]") { Justification = Justify.Left });
        var initialAnalysis = await isolationOptimizer.AnalyzeInitialIsolationProbesAsync(initialPlan);

        foreach (var note in initialAnalysis.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Initial Probe Analysis");

        AnsiConsole.Write(new Rule("[yellow]Continuation Isolation Samples[/]") { Justification = Justify.Left });

        var continuationPlan = TensorConfigGenerator.GenerateContinuationIsolationSamplePlan(
            initialAnalysis.GroupsToContinue,
            Cache.UnusedTensorGroups);

        if (continuationPlan.TotalCount > 0)
        {
            AnsiConsole.MarkupLine($"[grey]Queued continuation samples:[/] [cyan]{continuationPlan.TotalCount:N0}[/]");

            var continuationSummary = await quantizationService.ProcessHybridBatchAsync(continuationPlan.Plans);

            AnsiConsole.MarkupLine("[bold green]Continuation sampling complete.[/]");
            AnsiConsole.MarkupLine($"  [green]Completed:[/] {continuationSummary.Completed:N0}");
            AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {continuationSummary.Skipped:N0}");
            AnsiConsole.MarkupLine($"  [red]Failed:[/] {continuationSummary.Failed:N0}");
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]No continuation samples were required after smallest-first gating.[/]");
        }

        var mergedPlan = initialPlan.MergeWith(continuationPlan);

        var comboCountBefore = ComboCounter.CountAll();

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space Before Final Isolation Optimization");

        AnsiConsole.Write(new Rule("[yellow]Final Isolation Optimization[/]") { Justification = Justify.Left });
        var isolationResult = await isolationOptimizer.AnalyzeAndApplyFinalAsync(mergedPlan);

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Final Isolation Optimization");

        foreach (var gd in isolationResult.GroupDetails.OrderBy(x => x.GroupName))
        {
            AnsiConsole.Write(new Rule($"[yellow]Isolation Group: {Markup.Escape(gd.GroupName)}[/]") { Justification = Justify.Left });
            AnsiConsole.MarkupLine($"[green]Best savings:[/] {gd.BestReductionRatio:P2}");
            AnsiConsole.MarkupLine($"[green]Winning scheme:[/] {Markup.Escape(gd.WinningScheme ?? "n/a")}");
            AnsiConsole.MarkupLine($"[green]Explicit quant banned:[/] {(gd.ExplicitQuantBanned ? "[red]yes[/]" : "[green]no[/]")}");
            AnsiConsole.MarkupLine($"[green]BF16 suppressed:[/] {(gd.Bf16Suppressed ? "[yellow]yes[/]" : "[green]no[/]")}");

            foreach (var line in gd.Candidates)
                AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(line)}[/]");
        }

        var comboCountAfterRulePruning = ComboCounter.CountAll();

        await dbService.InitializeAsync(forceRebuild: true);

        long predictedSizePruned = await dbService.PrunePredictedLargerThanQ8Async(mergedPlan);

        AnsiConsole.MarkupLine($"[green]Groups reduced to BF16-only:[/] {isolationResult.ExplicitQuantBannedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]BF16-suppressed groups:[/] {isolationResult.Bf16SuppressedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]Hard damage eliminations:[/] {isolationResult.HardDamageEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Dominance eliminations:[/] {isolationResult.DominatedGroupSchemesBanned:N0}");
        AnsiConsole.MarkupLine($"[green]Bad trade eliminations:[/] {isolationResult.BadTradeEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Disabled combination baselines:[/] {isolationResult.DisabledBaselines:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count before pruning:[/] {comboCountBefore:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count after rule pruning:[/] {comboCountAfterRulePruning:N0}");
        AnsiConsole.MarkupLine($"[green]Predicted-size combo removals:[/] {predictedSizePruned:N0}");

        foreach (var note in isolationResult.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");
    }

    private void ShowEvolutionHelp()
    {
        AnsiConsole.MarkupLine("[bold yellow]Command: evolution[/]");
        AnsiConsole.WriteLine("Runs the full evolutionary quantization search algorithm on a target model.");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Usage:[/]");
        AnsiConsole.WriteLine("  mq evolution --model-dir \"<path>\" [options]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Arguments:[/]");
        AnsiConsole.MarkupLine("  [green]--model-dir[/]    Path to the model directory containing .safetensors files (Required)");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Example:[/]");
        AnsiConsole.WriteLine("  mq evolution --model-dir \"C:\\Models\\Mistral-7B\"");
    }
}