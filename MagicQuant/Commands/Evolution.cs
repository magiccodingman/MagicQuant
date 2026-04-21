using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Commands;

public class Evolution : ICommand
{
    private const int BruteForceFinalCombinationThreshold = 2_000;

    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            ShowEvolutionHelp();
            return;
        }

        string? modelDirRaw = args.FirstOrDefault(a =>
            string.Equals(a.Name, "model-dir", StringComparison.OrdinalIgnoreCase))?.Value;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
        {
            const string msg = "[red]Error:[/] Missing required argument [yellow]--model-dir[/].";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new InvalidOperationException("Missing required argument --model-dir.");
        }

        string fullModelPath = Path.GetFullPath(modelDirRaw);

        if (!Directory.Exists(fullModelPath))
        {
            string msg =
                $"[red]Error:[/] The directory [yellow]{Markup.Escape(fullModelPath)}[/] does not exist.";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new DirectoryNotFoundException($"The directory '{fullModelPath}' does not exist.");
        }

        var safeTensorFiles = Directory.GetFiles(fullModelPath, "*.safetensors", SearchOption.TopDirectoryOnly);

        if (safeTensorFiles.Length == 0)
        {
            AnsiConsole.MarkupLine(
                $"[red]Error:[/] No [yellow].safetensors[/] files found in [blue]{Markup.Escape(fullModelPath)}[/].");
            AnsiConsole.MarkupLine("[grey]Please ensure this is a valid HuggingFace model directory.[/]");
            throw new InvalidOperationException("No .safetensors files were found in the provided model directory.");
        }

        Cache.ModelDirectory = fullModelPath;
        Cache.ModelMagicQuantDirectory = Path.Combine(fullModelPath, "MagicQuant");
        Cache.ForceRelearnBaselineTensorMappings = args.Any(a =>
            string.Equals(a.Name, "relearn-baseline-mappings", StringComparison.OrdinalIgnoreCase));
        Cache.ForceRefreshHardwareProbe = args.Any(a =>
            string.Equals(a.Name, "recheck-hardware-probe", StringComparison.OrdinalIgnoreCase));
        Cache.UseImatrix = args.Any(a => string.Equals(a.Name, "use-imatrix", StringComparison.OrdinalIgnoreCase));
        Cache.ForceImatrixRebuild = args.Any(a => string.Equals(a.Name, "imatrix-force-rebuild", StringComparison.OrdinalIgnoreCase));
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(false);
        RuntimeSearchSpace.AllowHighPrecisionHybrids = args.Any(a => string.Equals(a.Name, "allow-high-precision-hybrids", StringComparison.OrdinalIgnoreCase));

        JsonHelper.DetectAndSetTorchType(Cache.ModelDirectory);

        if (!Directory.Exists(Cache.ModelMagicQuantDirectory))
            Directory.CreateDirectory(Cache.ModelMagicQuantDirectory);

        AnsiConsole.MarkupLine("[green]✔ Model Directory Validated[/]");
        AnsiConsole.Write(new Rule("[yellow]Evolution Configuration[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Model Path:   [blue]{Markup.Escape(Cache.ModelDirectory)}[/]");
        AnsiConsole.MarkupLine($"Output Path:  [blue]{Markup.Escape(Cache.ModelMagicQuantDirectory)}[/]");
        AnsiConsole.MarkupLine($"Files Found:  [green]{safeTensorFiles.Length:N0}[/] safe tensors");

        if (string.IsNullOrEmpty(Cache.LlamaBin))
            AnsiConsole.MarkupLine("[yellow]Warning:[/] Llama binaries path not set in Cache. (Did Initialization run?)");

        AnsiConsole.MarkupLine("[grey]Acquiring unique model ID...[/]");
        Cache.CurrentModelId = MagicQuantModelId.GetOrCreateModelId(Cache.ModelDirectory);
        AnsiConsole.MarkupLine($"[green]Model ID Created/Found:[/] [cyan]{Markup.Escape(Cache.CurrentModelId)}[/]");

        await EnsureSqliteReadyAsync();

        var pyManager = new PythonManager(Cache.MagicQuantDirectory);
        var benchmarkService = new BenchmarkService(pyManager);
        var quantizationService = new QuantizationService(benchmarkService);
        var imatrixService = new ImatrixService();

        if (Cache.ForceRelearnBaselineTensorMappings)
        {
            await quantizationService.InvalidateBaselineArtifactsAsync();
            AnsiConsole.MarkupLine("[yellow]Forced relearn is ON:[/] pure baseline samples will be rebuilt and relearned.");
        }

        string q8QuantizationKey = BaselineQuants.Q8_0.Names[0];
        var bf16ModelGgufPath = await quantizationService.EnsureBaseModelFileAsync(true);

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

        var imatrixEnsureResult = await imatrixService.EnsureImatrixAsync(imatrixRequest, ct: default);
        if (imatrixEnsureResult.Enabled)
        {
            string canonicalPath = imatrixEnsureResult.CanonicalImatrixPath ?? "n/a";
            string rebuiltText = imatrixEnsureResult.Rebuilt ? "yes" : "no";
            AnsiConsole.MarkupLine(
                $"[green]Imatrix active:[/] {Markup.Escape(canonicalPath)} (rebuilt={rebuiltText})");
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]Imatrix disabled for this run.[/]");
        }

        // Re-assert the live runtime flag from the imatrix resolution result so later phases
        // cannot accidentally inherit a stale default.
        RuntimeSearchSpace.SetImatrixAvailability(imatrixEnsureResult.Enabled);

        bool loadedPlanFromCache = !Cache.ForceRefreshHardwareProbe &&
                                   await benchmarkService.TryInitializeExecutionPlanFromCacheAsync(
                                       quantizationKey: q8QuantizationKey);

        if (!loadedPlanFromCache)
        {
            AnsiConsole.MarkupLine("[grey]Cache not usable, preparing probe-only Q8 baseline...[/]");
            var q8ModelGgufPath = await quantizationService.EnsurePureQ8ModelAsync();
            await benchmarkService.EnsureExecutionPlanAsync(
                q8ModelGgufPath,
                quantizationKey: q8QuantizationKey,
                forceRediscovery: Cache.ForceRefreshHardwareProbe);
        }

        bool nativeTruthAlreadyLearned = !Cache.ForceRelearnBaselineTensorMappings &&
                                         await quantizationService.HasNativeSourceLearnedTruthAsync();

        if (!nativeTruthAlreadyLearned || !loadedPlanFromCache)
        {
            await benchmarkService.ClampStaticNglWithBaseModelAsync(bf16ModelGgufPath);
        }
        else
        {
            AnsiConsole.MarkupLine(
                "[grey]Skipping base-model ngl clamp because native-source truth already exists and execution plan cache was loaded.[/]");
        }

        await quantizationService.CleanupPureQ8ModelAsync();

        var baseTypeName = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        var baseBenchDir = Path.Combine(Cache.ModelMagicQuantDirectory!, "Benchmarks", baseTypeName);
        var baseLogitsDir = Path.Combine(baseBenchDir, "logits");

        var baseModelQuant = HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant());

        if (!nativeTruthAlreadyLearned)
        {
            await benchmarkService.RunAllBenchmarksAsync(
                quantConfig: baseModelQuant,
                modelPath: bf16ModelGgufPath,
                benchDir: baseBenchDir,
                klLogitsDir: baseLogitsDir,
                saveLogits: true,
                domainsOverride: new[] { "general", "code", "math" });

            await quantizationService.LearnNativeSourceTruthAsync(bf16ModelGgufPath);
        }
        else
        {
            AnsiConsole.MarkupLine(
                "[grey]Skipping native BF16 baseline benchmark + relearn because learned native-source truth already exists. Use --relearn-baseline-mappings to force rebuild.[/]");
        }

        var compatibilityService = new ModelCompatibilityService(pyManager);
        await compatibilityService.RunCompatibilityCheckAsync(bf16ModelGgufPath);

        // Compatibility must not be allowed to silently downgrade the live policy flags for the
        // remainder of the evolution run. Re-assert them here as a final safeguard.
        RuntimeSearchSpace.SetImatrixAvailability(imatrixEnsureResult.Enabled);
        RuntimeSearchSpace.AllowHighPrecisionHybrids = args.Any(a =>
            string.Equals(a.Name, "allow-high-precision-hybrids", StringComparison.OrdinalIgnoreCase));

        CliHelpers.ValidateCombinationLogicWorks(true);

        var dbService = new QuantDatabaseService();
        await dbService.InitializeAsync();

        var comboCountBefore = ComboCounter.CountAll();
        var learnedBaselinePruner = new LearnedBaselinePruningService();

        var totalLearnedPruningResult = new LearnedBaselinePruningResult();

        if (!Cache.ForceRelearnBaselineTensorMappings)
        {
            var coverageStatus = await learnedBaselinePruner.GetCoverageStatusAsync();
            if (coverageStatus.SafeToApplyBeforeStartup)
            {
                AnsiConsole.Write(new Rule("[yellow]Pre-Startup Learned Baseline Pruning[/]") { Justification = Justify.Left });
                AnsiConsole.MarkupLine(
                    $"[grey]Using existing learned baseline coverage before startup sampling:[/] [cyan]{coverageStatus.PresentCandidateGroupPairs:N0}[/]/[cyan]{coverageStatus.ExpectedCandidateGroupPairs:N0}[/] candidate-group pairs.");

                SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space Before Pre-Startup Learned-Baseline Pruning");
                var preStartupLearnedPruningResult = await learnedBaselinePruner.AnalyzeAndApplyAsync();
                MergeLearnedPruningResults(totalLearnedPruningResult, preStartupLearnedPruningResult);
                SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Pre-Startup Learned-Baseline Pruning");

                foreach (var note in preStartupLearnedPruningResult.Notes)
                    AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");
            }
            else if (coverageStatus.HasAnyLearnedRows)
            {
                AnsiConsole.MarkupLine(
                    $"[grey]Skipping pre-startup learned pruning because learned coverage is incomplete for the current explicit candidate universe ({coverageStatus.PresentCandidateGroupPairs:N0}/{coverageStatus.ExpectedCandidateGroupPairs:N0} candidate-group pairs present).[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[grey]Skipping pre-startup learned pruning because no learned baseline rows exist yet for this model.[/]");
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]Skipping pre-startup learned pruning because --relearn-baseline-mappings was requested.[/]");
        }

        AnsiConsole.Write(new Rule("[yellow]Initial Isolation Startup Samples[/]") { Justification = Justify.Left });

        var isolationPlanner = new IsolationPlanningService();
        var initialPlan = isolationPlanner.BuildInitialPlan(Cache.UnusedTensorGroups);
        AnsiConsole.MarkupLine($"[grey]Queued initial startup samples:[/] [cyan]{initialPlan.TotalCount:N0}[/]");

        var initialSummary = await quantizationService.ProcessHybridBatchAsync(initialPlan.Plans);

        AnsiConsole.MarkupLine("[bold green]Initial startup sampling complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {initialSummary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {initialSummary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {initialSummary.Failed:N0}");

        AnsiConsole.MarkupLine("[bold magenta]Evolution flow marker:[/] startup sampling finished, refreshing learned-baseline pruning before initial probe analysis.");
        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space Before Learned-Baseline Pruning Refresh");

        AnsiConsole.Write(new Rule("[yellow]Learned Baseline Pruning Refresh[/]") { Justification = Justify.Left });
        var learnedPruningResult = await learnedBaselinePruner.AnalyzeAndApplyAsync();
        MergeLearnedPruningResults(totalLearnedPruningResult, learnedPruningResult);

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Learned-Baseline Pruning Refresh");

        foreach (var note in learnedPruningResult.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");

        var isolationOptimizer = new IsolationOptimizationService();

        AnsiConsole.Write(new Rule("[yellow]Initial Probe Analysis[/]") { Justification = Justify.Left });
        var initialAnalysis = await isolationOptimizer.AnalyzeInitialIsolationProbesAsync(initialPlan);

        AnsiConsole.Write(new Rule("[yellow]Initial Probe Group Decisions[/]") { Justification = Justify.Left });
        PrintIsolationGroupDecisions(initialAnalysis.GroupDetails);

        foreach (var note in initialAnalysis.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Initial Probe Analysis");

        AnsiConsole.Write(new Rule("[yellow]Continuation Isolation Samples[/]") { Justification = Justify.Left });

        AnsiConsole.MarkupLine($"[grey]Groups continuing after early probe:[/] [cyan]{initialAnalysis.GroupsToContinue.Count:N0}[/]");

        var continuationPlan = isolationPlanner.BuildContinuationPlan(
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

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space Before Final Isolation Optimization");

        AnsiConsole.Write(new Rule("[yellow]Final Isolation Optimization[/]") { Justification = Justify.Left });
        var isolationResult = await isolationOptimizer.AnalyzeAndApplyFinalAsync(mergedPlan);

        SearchSpaceDebugPrinter.PrintCurrentSearchSpace("Search Space After Final Isolation Optimization");

        foreach (var gd in isolationResult.GroupDetails.OrderBy(x => x.GroupName))
        {
            AnsiConsole.Write(
                new Rule($"[yellow]Isolation Group: {Markup.Escape(gd.GroupName)}[/]")
                {
                    Justification = Justify.Left
                });

            AnsiConsole.MarkupLine($"[green]Best savings:[/] {gd.BestReductionRatio:P2}");
            AnsiConsole.MarkupLine($"[green]Winning candidate:[/] {Markup.Escape(gd.WinningCandidate ?? "n/a")}");
            AnsiConsole.MarkupLine($"[green]Explicit quant banned:[/] {(gd.ExplicitQuantBanned ? "[red]yes[/]" : "[green]no[/]")}");
            AnsiConsole.MarkupLine($"[green]BF16 suppressed:[/] {(gd.Bf16Suppressed ? "[yellow]yes[/]" : "[green]no[/]")}");

            foreach (var line in gd.Candidates)
                AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(line)}[/]");
        }

        var comboCountAfterRulePruning = ComboCounter.CountAll();

        await dbService.InitializeAsync(forceRebuild: true);

        long predictedSizePruned = await dbService.PrunePredictedLargerThanQ8Async(mergedPlan);
        long highPrecisionPruned = await dbService.PruneHighPrecisionHybridCandidatesAsync();

        AnsiConsole.MarkupLine($"[green]Learned-baseline eliminations:[/] {totalLearnedPruningResult.GroupCandidateEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Baselines skipped without learned rows:[/] {totalLearnedPruningResult.BaselinesSkippedWithoutLearnedRows:N0}");
        AnsiConsole.MarkupLine($"[green]Groups reduced to explicit-banned->Q8-fallback:[/] {isolationResult.ExplicitQuantBannedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]BF16-suppressed groups:[/] {isolationResult.Bf16SuppressedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]Hard damage eliminations:[/] {isolationResult.HardDamageEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Dominance eliminations:[/] {isolationResult.DominatedGroupCandidatesBanned:N0}");
        AnsiConsole.MarkupLine($"[green]Bad trade eliminations:[/] {isolationResult.BadTradeEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Disabled combination baselines:[/] {isolationResult.DisabledBaselines:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count before pruning:[/] {comboCountBefore:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count after rule pruning:[/] {comboCountAfterRulePruning:N0}");
        AnsiConsole.MarkupLine($"[green]Predicted-size combo removals:[/] {predictedSizePruned:N0}");
        AnsiConsole.MarkupLine($"[green]Late-stage high-precision combo removals:[/] {highPrecisionPruned:N0}");

        foreach (var note in isolationResult.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");

        long finalRemainingCombinationCount = await dbService.GetRemainingCombinationCountAsync();

        AnsiConsole.MarkupLine($"[green]Final surviving combinations:[/] {finalRemainingCombinationCount:N0}");

        if (finalRemainingCombinationCount <= BruteForceFinalCombinationThreshold)
        {
            AnsiConsole.Write(new Rule("[yellow]Final Brute Force Benchmark Phase[/]") { Justification = Justify.Left });

            AnsiConsole.MarkupLine(
                $"[green]Final combination count[/] [cyan]{finalRemainingCombinationCount:N0}[/] " +
                $"is at or below the brute-force threshold of [yellow]{BruteForceFinalCombinationThreshold:N0}[/].");

            var finalConfigs = await dbService.GetRemainingTensorConfigsAsync();
            var finalQuants = finalConfigs
                .Select(x => (HybridQuant)x)
                .ToList();

            var finalSummary = await quantizationService.ProcessHybridBatchAsync(finalQuants);

            AnsiConsole.MarkupLine("[bold green]Final brute force benchmarking complete.[/]");
            AnsiConsole.MarkupLine($"  [green]Requested:[/] {finalSummary.Requested:N0}");
            AnsiConsole.MarkupLine($"  [green]Completed:[/] {finalSummary.Completed:N0}");
            AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {finalSummary.Skipped:N0}");
            AnsiConsole.MarkupLine($"  [red]Failed:[/] {finalSummary.Failed:N0}");
            AnsiConsole.MarkupLine("[yellow]Note:[/] Final model creation/export functionality is still being implemented.");
        }
        else
        {
            AnsiConsole.MarkupLine("[yellow]Note:[/] Final model creation/export functionality is still being implemented.");

            throw new InvalidOperationException(
                $"Prediction engine not created yet. Final surviving combinations were {finalRemainingCombinationCount:N0}, " +
                $"which is above the brute-force threshold of {BruteForceFinalCombinationThreshold:N0}.");
        }
    }

    private static void MergeLearnedPruningResults(LearnedBaselinePruningResult target, LearnedBaselinePruningResult source)
    {
        target.GroupCandidateEliminations += source.GroupCandidateEliminations;
        target.BaselinesSkippedWithoutLearnedRows += source.BaselinesSkippedWithoutLearnedRows;

        foreach (var note in source.Notes)
            target.Notes.Add(note);
    }

    private static void PrintIsolationGroupDecisions(IEnumerable<IsolationGroupDecision> decisions)
    {
        foreach (var gd in decisions.OrderBy(x => x.GroupName))
        {
            AnsiConsole.Write(
                new Rule($"[yellow]Isolation Group: {Markup.Escape(gd.GroupName)}[/]")
                {
                    Justification = Justify.Left
                });

            AnsiConsole.MarkupLine($"[green]Best savings:[/] {gd.BestReductionRatio:P2}");
            AnsiConsole.MarkupLine($"[green]Winning candidate:[/] {Markup.Escape(gd.WinningCandidate ?? "n/a")}");
            AnsiConsole.MarkupLine($"[green]Explicit quant banned:[/] {(gd.ExplicitQuantBanned ? "[red]yes[/]" : "[green]no[/]")}");
            AnsiConsole.MarkupLine($"[green]BF16 suppressed:[/] {(gd.Bf16Suppressed ? "[yellow]yes[/]" : "[green]no[/]")}");

            foreach (var line in gd.Candidates)
                AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(line)}[/]");
        }
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
        AnsiConsole.MarkupLine("  [green]--relearn-baseline-mappings[/]    Delete and relearn baseline tensor mappings (Optional)");
        AnsiConsole.MarkupLine("  [green]--recheck-hardware-probe[/]    Force hardware/Q8 probe and update cached plan in SQLite (Optional)");
        AnsiConsole.MarkupLine("  [green]--use-imatrix[/]    Enable imatrix acquisition/build and allow imatrix-required search candidates (Optional)");
        AnsiConsole.MarkupLine("  [green]--allow-high-precision-hybrids[/]    Keep BF16/F16 explicit group candidates in final surviving combos (Optional, default false)");

        AnsiConsole.MarkupLine("  [green]--imatrix-force-rebuild[/]    Delete/rebuild canonical imatrix artifacts before run (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-url[/]    HTTPS URL for direct imatrix artifact download (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-repo[/]    Hugging Face dataset repo ID for imatrix generation (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-split[/]    Dataset split for HF/local dataset source metadata/build (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-config[/]    Optional dataset config name for HF datasets (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-local-file[/]    Full path to local .json/.jsonl dataset source (Optional)");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Example:[/]");
        AnsiConsole.WriteLine("  mq evolution --model-dir \"C:\\Models\\Mistral-7B\"");
    }

    private static async Task EnsureSqliteReadyAsync(CancellationToken ct = default)
    {
        await using var db = new MagicQuantContext();
        await db.Database.MigrateAsync(ct);

        var model = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model != null)
            return;

        db.AiModelHashes.Add(new AiModelHash { UniqueHash = Cache.CurrentModelId });
        await db.SaveChangesAsync(ct);
    }
}