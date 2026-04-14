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

        TensorWeightScheme.ValidateSmallestConfiguration();

        AnsiConsole.MarkupLine("[green]✔ Model Directory Validated[/]");
        AnsiConsole.Write(new Rule("[yellow]Evolution Configuration[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Model Path:   [blue]{Cache.ModelDirectory}[/]");
        AnsiConsole.MarkupLine($"Output Path:  [blue]{Cache.ModelMagicQuantDirectory}[/]");
        AnsiConsole.MarkupLine($"Files Found:  [green]{safeTensorFiles.Length}[/] safe tensors");
        AnsiConsole.MarkupLine($"Smallest non-imatrix scheme: [cyan]{TensorWeightScheme.GetSmallestNonImatrix().Names[0]}[/]");

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

        // ---------------------------------------------------------
        // INITIAL ISOLATION PLAN
        // ---------------------------------------------------------
        AnsiConsole.Write(new Rule("[yellow]Initial Isolation Sample Generation[/]") { Justification = Justify.Left });

        var initialPlan = TensorConfigGenerator.GenerateRequiredSamplePlan(Cache.UnusedTensorGroups);
        AnsiConsole.MarkupLine($"[grey]Queued initial samples:[/] [cyan]{initialPlan.TotalCount:N0}[/]");
        AnsiConsole.MarkupLine("[grey]SQLite will be treated as the source of truth for completed samples.[/]");

        var initialSummary = await quantizationService.ProcessHybridBatchAsync(initialPlan.Plans);

        AnsiConsole.MarkupLine("[bold green]Initial sample generation phase complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {initialSummary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {initialSummary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {initialSummary.Failed:N0}");

        var comboCountBeforeGate = ComboCounter.CountAll();
        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space Before Smallest-Probe Gate");

        // ---------------------------------------------------------
        // SMALLEST-PROBE GATE
        // ---------------------------------------------------------
        AnsiConsole.Write(new Rule("[yellow]Smallest-Probe Isolation Gate[/]") { Justification = Justify.Left });
        var isolationOptimizer = new IsolationOptimizationService();
        var gateResult = await isolationOptimizer.ApplyInitialSamplingGateAsync(initialPlan);

        foreach (var gd in gateResult.GroupDetails.OrderBy(x => x.GroupName))
        {
            AnsiConsole.Write(new Rule($"[yellow]Smallest Probe: {Markup.Escape(gd.GroupName)}[/]") { Justification = Justify.Left });
            AnsiConsole.MarkupLine($"[green]Probe scheme:[/] {Markup.Escape(gd.ProbeScheme)}");
            AnsiConsole.MarkupLine($"[green]Reduction:[/] {gd.ReductionRatio:P2}");
            AnsiConsole.MarkupLine($"[green]Continue sampling:[/] {(gd.ContinueSampling ? "[green]yes[/]" : "[red]no[/]")}");
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(gd.Reason)}[/]");
        }

        var comboCountAfterGate = ComboCounter.CountAll();
        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Smallest-Probe Gate");

        // ---------------------------------------------------------
        // CONTINUATION ISOLATION PLAN
        // ---------------------------------------------------------
        var continuationPlan = new RequiredSampleGenerationResult();
        SampleProcessingSummary? continuationSummary = null;

        if (gateResult.GroupIdsToContinue.Count > 0)
        {
            AnsiConsole.Write(new Rule("[yellow]Continuation Isolation Sample Generation[/]") { Justification = Justify.Left });

            continuationPlan = TensorConfigGenerator.GenerateContinuationIsolationPlan(
                gateResult.GroupIdsToContinue,
                Cache.UnusedTensorGroups);

            if (continuationPlan.TotalCount > 0)
            {
                AnsiConsole.MarkupLine($"[grey]Queued continuation samples:[/] [cyan]{continuationPlan.TotalCount:N0}[/]");
                continuationSummary = await quantizationService.ProcessHybridBatchAsync(continuationPlan.Plans);

                AnsiConsole.MarkupLine("[bold green]Continuation sample generation phase complete.[/]");
                AnsiConsole.MarkupLine($"  [green]Completed:[/] {continuationSummary.Completed:N0}");
                AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {continuationSummary.Skipped:N0}");
                AnsiConsole.MarkupLine($"  [red]Failed:[/] {continuationSummary.Failed:N0}");
            }
            else
            {
                AnsiConsole.MarkupLine("[grey]No continuation isolation samples were required after the smallest-probe gate.[/]");
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]No tensor groups survived the smallest-probe gate. Skipping continuation isolation sampling.[/]");
        }

        // ---------------------------------------------------------
        // FINAL ISOLATION PRUNING
        // ---------------------------------------------------------
        var comboCountBeforeFinalPruning = ComboCounter.CountAll();
        var fullPlan = RequiredSampleGenerationResult.Merge(initialPlan, continuationPlan);

        AnsiConsole.Write(new Rule("[yellow]Final Isolation Optimization[/]") { Justification = Justify.Left });
        var isolationResult = await isolationOptimizer.AnalyzeAndApplyAsync(fullPlan, gateResult);

        var comboCountAfterFinalPruning = ComboCounter.CountAll();
        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Final Isolation Optimization");

        foreach (var gd in isolationResult.GroupDetails.OrderBy(x => x.GroupName))
        {
            AnsiConsole.Write(new Rule($"[yellow]Isolation Group: {Markup.Escape(gd.GroupName)}[/]") { Justification = Justify.Left });
            AnsiConsole.MarkupLine($"[green]Stopped early:[/] {(gd.StoppedEarly ? "[yellow]yes[/]" : "[green]no[/]")}");
            AnsiConsole.MarkupLine($"[green]Winning scheme:[/] {Markup.Escape(gd.WinningScheme ?? "n/a")}");

            foreach (var line in gd.Candidates)
                AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(line)}[/]");

            foreach (var line in gd.Eliminations)
                AnsiConsole.MarkupLine($"  [red]- {Markup.Escape(line)}[/]");
        }

        await dbService.InitializeAsync(forceRebuild: true);

        AnsiConsole.MarkupLine($"[green]Groups stopped early:[/] {gateResult.GroupsStoppedEarly:N0}");
        AnsiConsole.MarkupLine($"[green]Hard-damage eliminations:[/] {isolationResult.HardDamageEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Dominance eliminations:[/] {isolationResult.DominatedGroupSchemesBanned:N0}");
        AnsiConsole.MarkupLine($"[green]Disabled combination baselines:[/] {isolationResult.DisabledBaselines:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count before smallest-probe gate:[/] {comboCountBeforeGate:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count after smallest-probe gate:[/] {comboCountAfterGate:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count before final pruning:[/] {comboCountBeforeFinalPruning:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count after final pruning:[/] {comboCountAfterFinalPruning:N0}");

        foreach (var note in gateResult.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");

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