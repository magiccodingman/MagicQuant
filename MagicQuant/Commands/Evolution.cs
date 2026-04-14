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

        AnsiConsole.Write(new Rule("[yellow]Required Sample Generation[/]") { Justification = Justify.Left });

        var samplePlan = TensorConfigGenerator.GenerateRequiredSamplePlan(Cache.UnusedTensorGroups);
        AnsiConsole.MarkupLine($"[grey]Queued required samples:[/] [cyan]{samplePlan.TotalCount:N0}[/]");
        AnsiConsole.MarkupLine("[grey]SQLite will be treated as the source of truth for completed samples.[/]");

        var sampleSummary = await quantizationService.ProcessHybridBatchAsync(samplePlan.Plans);

        AnsiConsole.MarkupLine("[bold green]Sample generation phase complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {sampleSummary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {sampleSummary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {sampleSummary.Failed:N0}");
        
        
        

        var comboCountBefore = ComboCounter.CountAll();

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space Before Isolation Optimization");
        
        AnsiConsole.Write(new Rule("[yellow]Isolation Optimization[/]") { Justification = Justify.Left });
        var isolationOptimizer = new IsolationOptimizationService();
        var isolationResult = await isolationOptimizer.AnalyzeAndApplyAsync(samplePlan);
        
        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Isolation Optimization");
        
        foreach (var gd in isolationResult.GroupDetails.OrderBy(x => x.GroupName))
        {
            AnsiConsole.Write(new Rule($"[yellow]Isolation Group: {Markup.Escape(gd.GroupName)}[/]") { Justification = Justify.Left });

            AnsiConsole.MarkupLine($"[green]Best reduction:[/] {gd.BestReductionRatio:P2}");
            AnsiConsole.MarkupLine($"[green]Winning scheme:[/] {Markup.Escape(gd.WinningScheme ?? "n/a")}");
            AnsiConsole.MarkupLine($"[green]Locked to native:[/] {(gd.LockedToNative ? "[red]yes[/]" : "[green]no[/]")}");

            foreach (var line in gd.Candidates)
                AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(line)}[/]");
        }
        

        var comboCountAfter = ComboCounter.CountAll();

        await dbService.InitializeAsync(forceRebuild: true);

        AnsiConsole.MarkupLine($"[green]Native-locked groups:[/] {isolationResult.NativeLockedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]Dominated group-scheme bans applied:[/] {isolationResult.DominatedGroupSchemesBanned:N0}");
        AnsiConsole.MarkupLine($"[green]Disabled combination baselines:[/] {isolationResult.DisabledBaselines:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count before pruning:[/] {comboCountBefore:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count after pruning:[/] {comboCountAfter:N0}");

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