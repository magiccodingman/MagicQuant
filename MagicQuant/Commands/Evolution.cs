using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using MagicQuant.Services.Progress;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Commands;

public class Evolution : ICommand
{
    private static readonly string[] RequiredNativeKldDomains = ["general", "code", "math"];

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
            modelDirRaw = Config.Current.Paths.ModelDir;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
        {
            const string msg = "[red]Error:[/] Missing required model directory. Provide [yellow]--model-dir[/] or set [yellow]paths.model_dir[/] in YAML.";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new InvalidOperationException("Missing required model directory.");
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
        ModelRuntimePathService.InitializeForCurrentModel();
        await new ScratchStorageService(new ModelArtifactPathService()).CleanupStaleScratchArtifactsAsync();
        Cache.ForceRefreshHardwareProbe = Config.Current.Flags.ForceRefreshHardwareProbe;
        Cache.UseImatrix = Config.Current.Flags.UseImatrix;
        Cache.ForceImatrixRebuild = Config.Current.Flags.ForceImatrixRebuild;

        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(false);
        RuntimeSearchSpace.AllowHighPrecisionHybrids = Config.Current.Flags.AllowHighPrecisionHybrids;

        JsonHelper.DetectAndSetTorchType(Cache.ModelDirectory);

        if (!Directory.Exists(Cache.ModelMagicQuantDirectory))
            Directory.CreateDirectory(Cache.ModelMagicQuantDirectory);

        Cache.OutputDirectory = ResolveAndValidateOutputDirectory();

        AnsiConsole.MarkupLine("[green]✔ Model Directory Validated[/]");
        AnsiConsole.Write(new Rule("[yellow]Evolution Configuration[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Model Path:   [blue]{Markup.Escape(Cache.ModelDirectory)}[/]");
        AnsiConsole.MarkupLine($"Work Path:    [blue]{Markup.Escape(Cache.ModelMagicQuantDirectory)}[/]");
        AnsiConsole.MarkupLine($"Export Path:  [blue]{Markup.Escape(Cache.OutputDirectory ?? "n/a")}[/]");
        AnsiConsole.MarkupLine($"Files Found:  [green]{safeTensorFiles.Length:N0}[/] safe tensors");

        if (string.IsNullOrEmpty(Cache.LlamaBin))
            AnsiConsole.MarkupLine("[yellow]Warning:[/] Llama binaries path not set in Cache. (Did Initialization run?)");

        AnsiConsole.MarkupLine("[grey]Acquiring unique model ID...[/]");
        Cache.CurrentModelId = MagicQuantModelId.GetOrCreateModelId(Cache.ModelDirectory);
        AnsiConsole.MarkupLine($"[green]Model ID Created/Found:[/] [cyan]{Markup.Escape(Cache.CurrentModelId)}[/]");

        var pyManager = new PythonManager(Cache.MagicQuantDirectory!);

        await EnsureSqliteReadyAsync();

        var benchmarkService = new BenchmarkService(pyManager);
        var quantizationService = new QuantizationService(benchmarkService);
        var imatrixService = new ImatrixService();

        string q8QuantizationKey = BaselineQuants.Q8_0.Names[0];
        var bf16ModelGgufPath = await quantizationService.EnsureBaseModelFileAsync(true);

        var sidecarService = new ModelSidecarArtifactService(pyManager);
        await sidecarService.EnsureMmprojArtifactAvailableAsync();

        var architectureFamilyService = new ArchitectureFamilyService(pyManager);
        await architectureFamilyService.EnsureCurrentArchitectureFamilyAsync(bf16ModelGgufPath);

        var tensorGroupProfileService = new TensorGroupProfileService();
        await tensorGroupProfileService.EnsureCurrentProfileAsync();

        var customBaselineService = new HuggingFaceBaselineService(pyManager);
        var resolvedCustomBaselines = await customBaselineService.PrecheckAndRegisterConfiguredBaselinesAsync();

        if (Config.Current.Baselines.CustomRepositories.Any(x => x.Enabled) && resolvedCustomBaselines.Count == 0)
        {
            throw new InvalidOperationException(
                "Custom baseline repositories were enabled, but no custom baselines resolved into the runtime registry.");
        }

        await new TargetedRelearnService().PlanConfirmAndExecuteAsync(resolvedCustomBaselines);

        var imatrixRequest = new ImatrixRequest
        {
            UseImatrix = Cache.UseImatrix,
            ForceRebuild = Cache.ForceImatrixRebuild,
            ImatrixUrl = Config.Current.Imatrix.ImatrixUrl,
            DatasetRepo = Config.Current.Imatrix.DatasetRepo,
            DatasetSplit = Config.Current.Imatrix.DatasetSplit,
            DatasetConfig = Config.Current.Imatrix.DatasetConfig,
            LocalDatasetFile = Config.Current.Imatrix.DatasetLocalFile,
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

        string baseTypeName = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        bool loadedPlanFromCache = !Cache.ForceRefreshHardwareProbe &&
                                   await benchmarkService.TryInitializeDynamicExecutionPlanFromCacheAsync(
                                       q8QuantizationKey: q8QuantizationKey,
                                       nativeModelPath: bf16ModelGgufPath,
                                       nativeQuantizationKey: baseTypeName);

        if (!loadedPlanFromCache)
        {
            AnsiConsole.MarkupLine("[grey]Dynamic execution-plan cache not usable; probing Q8 + native anchors...[/]");
            await using var q8Lease = await quantizationService.BuildPureQ8ProbeLeaseAsync();
            await benchmarkService.EnsureDynamicExecutionPlanAsync(
                q8ModelPath: q8Lease.GgufPath,
                nativeModelPath: bf16ModelGgufPath,
                q8QuantizationKey: q8QuantizationKey,
                nativeQuantizationKey: baseTypeName,
                forceRediscovery: Cache.ForceRefreshHardwareProbe);
        }

        bool nativeTruthAlreadyLearned =
            await quantizationService.HasNativeSourceLearnedTruthAsync();
        if (nativeTruthAlreadyLearned && loadedPlanFromCache)
        {
            AnsiConsole.MarkupLine(
                "[grey]Native-source truth already exists and dynamic plan loaded from cache.[/]");
        }
        var benchmarkRootDir = Path.Combine(Cache.ModelMagicQuantDirectory!, "Benchmarks");
        var baseBenchDir = Path.Combine(benchmarkRootDir, baseTypeName);
        var baseLogitsDir = Path.Combine(baseBenchDir, "logits");
        var pplCorporaDir = Path.Combine(benchmarkRootDir, "_ppl_corpora");

        var baseModelQuant = HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant());

        await EnsureNativeBenchmarkEnvironmentReadyAsync(
            benchmarkService: benchmarkService,
            quantizationService: quantizationService,
            baseModelQuant: baseModelQuant,
            bf16ModelGgufPath: bf16ModelGgufPath,
            baseBenchDir: baseBenchDir,
            baseLogitsDir: baseLogitsDir,
            pplCorporaDir: pplCorporaDir,
            nativeTruthAlreadyLearned: nativeTruthAlreadyLearned);

        var compatibilityService = new ModelCompatibilityService(pyManager);
        await compatibilityService.RunCompatibilityCheckAsync(bf16ModelGgufPath);

        // Compatibility must not be allowed to silently downgrade the live policy flags for the
        // remainder of the evolution run. Re-assert them here as a final safeguard.
        RuntimeSearchSpace.SetImatrixAvailability(imatrixEnsureResult.Enabled);
        RuntimeSearchSpace.AllowHighPrecisionHybrids = Config.Current.Flags.AllowHighPrecisionHybrids;

        PrintCustomBaselineRuntimeSummary(resolvedCustomBaselines, imatrixEnsureResult.Enabled);
        
        // No longer needed
        //CliHelpers.ValidateCombinationLogicWorks(true);

        var comboCountBefore = ComboCounter.CountAll();
        var totalLearnedPruningResult = new LearnedBaselinePruningResult();

        AnsiConsole.MarkupLine("[grey]Learned-baseline early pruning is disabled for this build. Startup sampling will proceed without learned-scheme candidate elimination.[/]");

        AnsiConsole.Write(new Rule("[yellow]Initial Isolation Startup Samples[/]") { Justification = Justify.Left });

        var isolationPlanner = new IsolationPlanningService();
        var initialPlan = isolationPlanner.BuildInitialPlan(Cache.UnusedTensorGroups);

        AnsiConsole.MarkupLine($"[grey]Queued initial startup samples:[/] [cyan]{initialPlan.TotalCount:N0}[/]");

        var initialSummary = await quantizationService.ProcessHybridBatchAsync(
            initialPlan.Plans,
            new StageProgressOptions
            {
                StageName = "Initial isolation startup samples",
                Total = initialPlan.TotalCount,
                MinimumNonSkippedSamplesBeforeEta = 2,
                ShowEta = true,
                CountSkippedForEta = false
            },
            default);

        AnsiConsole.MarkupLine("[bold green]Initial startup sampling complete.[/]");
        AnsiConsole.MarkupLine($"  [green]Completed:[/] {initialSummary.Completed:N0}");
        AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {initialSummary.Skipped:N0}");
        AnsiConsole.MarkupLine($"  [red]Failed:[/] {initialSummary.Failed:N0}");

        AnsiConsole.MarkupLine("[bold magenta]Evolution flow marker:[/] startup sampling finished. Learned-baseline early pruning remains disabled for subsequent phases.");

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

            var continuationSummary = await quantizationService.ProcessHybridBatchAsync(
                continuationPlan.Plans,
                new StageProgressOptions
                {
                    StageName = "Continuation isolation samples",
                    Total = continuationPlan.TotalCount,
                    MinimumNonSkippedSamplesBeforeEta = 2,
                    ShowEta = true,
                    CountSkippedForEta = false
                },
                default);

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

        var archivalGroupIds = TReg.All
            .Where(x => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == x.UniqueId))
            .Select(x => x.UniqueId)
            .Except(initialAnalysis.GroupsToContinue)
            .OrderBy(x => x)
            .ToList();

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

        var dbService = new QuantDatabaseService();
        await dbService.InitializeAsync(forceRebuild: true);

        // The old MDA/predicted-size ceiling pass is intentionally removed.
        // DuckDB now stays as the allowed candidate universe, and the rank-safe
        // isolation predictor chooses which candidates deserve real validation.
        long predictedSizePruned = 0;
        long highPrecisionPruned = await dbService.PruneHighPrecisionHybridCandidatesAsync();

        AnsiConsole.MarkupLine($"[green]Learned-baseline eliminations:[/] {totalLearnedPruningResult.GroupCandidateEliminations:N0} [grey](early pruning disabled)[/]");
        AnsiConsole.MarkupLine($"[green]Baselines skipped without learned rows:[/] {totalLearnedPruningResult.BaselinesSkippedWithoutLearnedRows:N0} [grey](early pruning disabled)[/]");
        AnsiConsole.MarkupLine($"[green]Groups reduced to explicit-banned->Q8-fallback:[/] {isolationResult.ExplicitQuantBannedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]BF16-suppressed groups:[/] {isolationResult.Bf16SuppressedGroups:N0}");
        AnsiConsole.MarkupLine($"[green]Hard damage eliminations:[/] {isolationResult.HardDamageEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Dominance eliminations:[/] {isolationResult.DominatedGroupCandidatesBanned:N0}");
        AnsiConsole.MarkupLine($"[green]Bad trade eliminations:[/] {isolationResult.BadTradeEliminations:N0}");
        AnsiConsole.MarkupLine($"[green]Disabled combination baselines:[/] {isolationResult.DisabledBaselines:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count before pruning:[/] {comboCountBefore:N0}");
        AnsiConsole.MarkupLine($"[green]Combination count after rule pruning:[/] {comboCountAfterRulePruning:N0}");
        AnsiConsole.MarkupLine($"[green]Predicted-size combo removals:[/] {predictedSizePruned:N0} [grey](obsolete MDA ceiling pruning removed)[/]");
        AnsiConsole.MarkupLine($"[green]Late-stage high-precision combo removals:[/] {highPrecisionPruned:N0}");

        foreach (var note in isolationResult.Notes)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(note)}[/]");

        long finalRemainingCombinationCount = await dbService.GetRemainingCombinationCountAsync();

        AnsiConsole.MarkupLine($"[green]Final surviving combinations after stage-1 pruning:[/] {finalRemainingCombinationCount:N0}");

        AnsiConsole.Write(new Rule("[yellow]Archival Isolation Coverage[/]") { Justification = Justify.Left });

        var archivalCoveragePlan = isolationPlanner.BuildArchivalCoveragePlan(
            groupIdsToArchive: archivalGroupIds,
            existingPlanKeys: mergedPlan.Plans.Select(x => x.Key),
            missingTensorGroups: Cache.UnusedTensorGroups);

        var archivalCoverageGroups = archivalCoveragePlan.Plans
            .Where(x => x.TargetGroupId.HasValue)
            .Select(x => x.TargetGroupId!.Value)
            .Distinct()
            .Count();

        AnsiConsole.MarkupLine($"[grey]Groups queued for archival coverage:[/] [cyan]{archivalCoverageGroups:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]Non-continuing groups targeted for archival fill:[/] [cyan]{archivalGroupIds.Count:N0}[/]");
        AnsiConsole.MarkupLine("[grey]This pass does not feed current-run pruning; it only fills missing isolated-sample coverage in the database for groups that were fixed/collapsed out of combo exploration.[/]");

        if (archivalCoveragePlan.TotalCount > 0)
        {
            AnsiConsole.MarkupLine($"[grey]Queued archival isolation samples:[/] [cyan]{archivalCoveragePlan.TotalCount:N0}[/]");

            var archivalCoverageSummary = await quantizationService.ProcessHybridBatchAsync(
                archivalCoveragePlan.Plans,
                new StageProgressOptions
                {
                    StageName = "Archival isolation coverage samples",
                    Total = archivalCoveragePlan.TotalCount,
                    MinimumNonSkippedSamplesBeforeEta = 2,
                    ShowEta = true,
                    CountSkippedForEta = false
                },
                default);

            AnsiConsole.MarkupLine("[bold green]Archival isolation coverage complete.[/]");
            AnsiConsole.MarkupLine($"  [green]Completed:[/] {archivalCoverageSummary.Completed:N0}");
            AnsiConsole.MarkupLine($"  [yellow]Skipped existing:[/] {archivalCoverageSummary.Skipped:N0}");
            AnsiConsole.MarkupLine($"  [red]Failed:[/] {archivalCoverageSummary.Failed:N0}");
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]No archival isolation coverage samples were required.[/]");
        }

        var finalIsolationManifestPlan = mergedPlan.MergeWith(archivalCoveragePlan);

        var survivalPipeline = new CombinationSurvivalPipelineService(quantizationService);
        var finalizationResult = await survivalPipeline.RunAsync(
            isolationSamplePlan: finalIsolationManifestPlan,
            isolationOptimizationResult: isolationResult,
            ct: default);

        AnsiConsole.Write(new Rule("[yellow]Export Summary[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[green]Export directory:[/] [blue]{Markup.Escape(Cache.OutputDirectory ?? "n/a")}[/]");
        AnsiConsole.MarkupLine($"[green]Final brutal survivors:[/] [cyan]{finalizationResult.BrutalSurvivors.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[green]Selected survivors:[/] [cyan]{finalizationResult.SelectedRows.Count(x => x.Enabled):N0}[/]");
        AnsiConsole.MarkupLine($"[green]Exported/linkable artifacts:[/] [cyan]{finalizationResult.ExportedArtifacts.Count:N0}[/]");
    }

    private static async Task EnsureNativeBenchmarkEnvironmentReadyAsync(
        BenchmarkService benchmarkService,
        QuantizationService quantizationService,
        HybridQuant baseModelQuant,
        string bf16ModelGgufPath,
        string baseBenchDir,
        string baseLogitsDir,
        string pplCorporaDir,
        bool nativeTruthAlreadyLearned)
    {
        var status = ValidateNativeBenchmarkEnvironment(
            baseBenchDir: baseBenchDir,
            baseLogitsDir: baseLogitsDir,
            pplCorporaDir: pplCorporaDir,
            requiredDomains: RequiredNativeKldDomains);

        bool mustRegenerateNativeBenchmarkArtifacts =
            !status.IsValid;

        if (mustRegenerateNativeBenchmarkArtifacts)
        {
            AnsiConsole.Write(new Rule("[yellow]Native BF16 Benchmark/KLD Artifact Validation[/]") { Justification = Justify.Left });

            AnsiConsole.MarkupLine("[yellow]Native BF16 benchmark/KLD artifacts are missing or incomplete.[/] Regenerating required artifacts.");

            PrintNativeBenchmarkEnvironmentIssues(status);

            await ForceRegenerateNativeBenchmarkArtifactsAsync(
                benchmarkService: benchmarkService,
                baseModelQuant: baseModelQuant,
                bf16ModelGgufPath: bf16ModelGgufPath,
                baseBenchDir: baseBenchDir,
                baseLogitsDir: baseLogitsDir);

            status = ValidateNativeBenchmarkEnvironment(
                baseBenchDir: baseBenchDir,
                baseLogitsDir: baseLogitsDir,
                pplCorporaDir: pplCorporaDir,
                requiredDomains: RequiredNativeKldDomains);

            if (!status.IsValid)
            {
                var details = string.Join(
                    Environment.NewLine,
                    status.MissingOrInvalidArtifacts.Select(x => $"- {x}"));

                throw new InvalidOperationException(
                    "Native BF16 benchmark/logit generation completed, but required native benchmark artifacts are still missing or invalid. " +
                    "This is fatal because every non-base benchmark requires complete native KLD logits." +
                    Environment.NewLine +
                    details);
            }

            AnsiConsole.MarkupLine("[green]Native BF16 benchmark/KLD artifacts validated.[/]");
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]Native BF16 benchmark/KLD artifacts already exist and passed validation.[/]");
        }

        // Disk artifact validation is not enough. Native tensor learning is tied to the
        // persisted TensorCombo/AiBenchmark identity. The repair path above may run in
        // transient mode so it can regenerate logits even when stale DB truth exists; after
        // the artifacts are valid, explicitly hydrate/validate the SQLite benchmark row
        // from those artifacts before native-source learning tries to attach to it.
        await EnsureNativeBenchmarkDbTruthAsync(
            benchmarkService: benchmarkService,
            baseModelQuant: baseModelQuant,
            bf16ModelGgufPath: bf16ModelGgufPath,
            baseBenchDir: baseBenchDir,
            baseLogitsDir: baseLogitsDir);

        if (!nativeTruthAlreadyLearned)
        {
            await quantizationService.LearnNativeSourceTruthAsync(bf16ModelGgufPath);
        }
        else
        {
            AnsiConsole.MarkupLine(
                "[grey]Skipping native-source tensor relearn because learned native-source truth already exists.[/]");
        }
    }

    private static async Task EnsureNativeBenchmarkDbTruthAsync(
        BenchmarkService benchmarkService,
        HybridQuant baseModelQuant,
        string bf16ModelGgufPath,
        string baseBenchDir,
        string baseLogitsDir)
    {
        bool previousSuppressBenchmarkPersistence = Cache.SuppressBenchmarkPersistence;

        try
        {
            Cache.SuppressBenchmarkPersistence = false;

            await benchmarkService.RunAllBenchmarksAsync(
                quantConfig: baseModelQuant,
                modelPath: bf16ModelGgufPath,
                benchDir: baseBenchDir,
                klLogitsDir: baseLogitsDir,
                saveLogits: true,
                domainsOverride: RequiredNativeKldDomains);

            AnsiConsole.MarkupLine("[grey]Native BF16 benchmark DB truth hydrated/validated.[/]");
        }
        finally
        {
            Cache.SuppressBenchmarkPersistence = previousSuppressBenchmarkPersistence;
        }
    }

    private static async Task ForceRegenerateNativeBenchmarkArtifactsAsync(
        BenchmarkService benchmarkService,
        HybridQuant baseModelQuant,
        string bf16ModelGgufPath,
        string baseBenchDir,
        string baseLogitsDir)
    {
        if (Directory.Exists(baseBenchDir))
        {
            AnsiConsole.MarkupLine(
                $"[grey]Clearing incomplete/stale native benchmark directory:[/] {Markup.Escape(baseBenchDir)}");

            Directory.Delete(baseBenchDir, recursive: true);
        }

        Directory.CreateDirectory(baseBenchDir);
        Directory.CreateDirectory(baseLogitsDir);

        bool previousSuppressBenchmarkPersistence = Cache.SuppressBenchmarkPersistence;

        try
        {
            // This is intentional.
            //
            // If persisted native BF16 benchmark rows already exist in SQLite, the normal
            // BenchmarkService path may return DB truth without actually running llama-perplexity,
            // which means missing KLD logits would stay missing forever.
            //
            // Transient mode forces this artifact-repair pass to rely on disk execution instead
            // of DB benchmark truth. The native tensor truth is learned separately below.
            Cache.SuppressBenchmarkPersistence = true;

            await benchmarkService.RunAllBenchmarksAsync(
                quantConfig: baseModelQuant,
                modelPath: bf16ModelGgufPath,
                benchDir: baseBenchDir,
                klLogitsDir: baseLogitsDir,
                saveLogits: true,
                domainsOverride: RequiredNativeKldDomains);
        }
        finally
        {
            Cache.SuppressBenchmarkPersistence = previousSuppressBenchmarkPersistence;
        }
    }

    private static NativeBenchmarkEnvironmentStatus ValidateNativeBenchmarkEnvironment(
        string baseBenchDir,
        string baseLogitsDir,
        string pplCorporaDir,
        IReadOnlyCollection<string> requiredDomains)
    {
        var issues = new List<string>();

        if (string.IsNullOrWhiteSpace(baseBenchDir))
        {
            issues.Add("Native benchmark directory path is null/empty.");
        }
        else if (!Directory.Exists(baseBenchDir))
        {
            issues.Add($"Native benchmark directory does not exist: {baseBenchDir}");
        }

        if (string.IsNullOrWhiteSpace(baseLogitsDir))
        {
            issues.Add("Native KLD logits directory path is null/empty.");
        }
        else if (!Directory.Exists(baseLogitsDir))
        {
            issues.Add($"Native KLD logits directory does not exist: {baseLogitsDir}");
        }

        if (string.IsNullOrWhiteSpace(pplCorporaDir))
        {
            issues.Add("_ppl_corpora directory path is null/empty.");
        }
        else if (!Directory.Exists(pplCorporaDir))
        {
            issues.Add($"_ppl_corpora directory does not exist: {pplCorporaDir}");
        }
        else if (!Directory.EnumerateFiles(pplCorporaDir, "*", SearchOption.AllDirectories).Any())
        {
            issues.Add($"_ppl_corpora directory exists but contains no files: {pplCorporaDir}");
        }

        foreach (var domain in requiredDomains.OrderBy(x => x, StringComparer.Ordinal))
        {
            if (!string.IsNullOrWhiteSpace(baseBenchDir) && Directory.Exists(baseBenchDir))
            {
                var pplLog = Path.Combine(baseBenchDir, $"perplexity_{domain}.log");

                if (!File.Exists(pplLog))
                {
                    issues.Add($"Missing native BF16 perplexity log for domain '{domain}': {pplLog}");
                }
                else if (new FileInfo(pplLog).Length <= 0)
                {
                    issues.Add($"Native BF16 perplexity log is empty for domain '{domain}': {pplLog}");
                }
            }

            if (!string.IsNullOrWhiteSpace(baseLogitsDir) && Directory.Exists(baseLogitsDir))
            {
                var logitsFile = Path.Combine(baseLogitsDir, $"kld_logits_{domain}.bin");

                if (!File.Exists(logitsFile))
                {
                    issues.Add($"Missing native KLD logits for domain '{domain}': {logitsFile}");
                }
                else if (new FileInfo(logitsFile).Length <= 0)
                {
                    issues.Add($"Native KLD logits file is empty for domain '{domain}': {logitsFile}");
                }
            }
        }

        return new NativeBenchmarkEnvironmentStatus(
            IsValid: issues.Count == 0,
            MissingOrInvalidArtifacts: issues);
    }

    private static void PrintNativeBenchmarkEnvironmentIssues(NativeBenchmarkEnvironmentStatus status)
    {
        if (status.IsValid)
            return;

        foreach (var issue in status.MissingOrInvalidArtifacts.Take(20))
            AnsiConsole.MarkupLine($"[grey]- {Markup.Escape(issue)}[/]");

        if (status.MissingOrInvalidArtifacts.Count > 20)
        {
            AnsiConsole.MarkupLine(
                $"[grey]- ...and {status.MissingOrInvalidArtifacts.Count - 20:N0} more issue(s).[/]");
        }
    }

    private sealed record NativeBenchmarkEnvironmentStatus(
        bool IsValid,
        IReadOnlyList<string> MissingOrInvalidArtifacts);

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

    private static void PrintCustomBaselineRuntimeSummary(
        IReadOnlyCollection<ResolvedCustomBaselineSpec> resolvedCustomBaselines,
        bool hasUsableImatrix)
    {
        AnsiConsole.Write(new Rule("[yellow]Custom Baseline Runtime Summary[/]") { Justification = Justify.Left });

        var learning = BaselineQuants.GetLearningBaselines(hasUsableImatrix);
        var carriers = BaselineQuants.GetCombinationCarrierBaselines(hasUsableImatrix);
        var explicitCandidates = BaselineQuants.GetGroupCombinationCandidates(hasUsableImatrix, Config.Current.Flags.AllowHighPrecisionHybrids);

        AnsiConsole.MarkupLine($"[grey]Learning baselines in runtime registry:[/] [cyan]{learning.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]Combination carriers in runtime registry:[/] [cyan]{carriers.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]Explicit group candidates in runtime registry:[/] [cyan]{explicitCandidates.Count:N0}[/]");

        if (resolvedCustomBaselines.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No custom baselines were resolved for this run.[/]");
            return;
        }

        AnsiConsole.MarkupLine($"[green]Custom baselines registered:[/] [cyan]{resolvedCustomBaselines.Count:N0}[/]");

        foreach (var custom in resolvedCustomBaselines.OrderBy(x => x.DynamicBaselineId))
        {
            bool inLearning = learning.Any(x => x.UniqueId == custom.DynamicBaselineId);
            bool inCarriers = carriers.Any(x => x.UniqueId == custom.DynamicBaselineId);
            bool inExplicit = explicitCandidates.Any(x => x.UniqueId == custom.DynamicBaselineId);

            AnsiConsole.MarkupLine(
                $"  [cyan]{custom.DynamicBaselineId}[/] [yellow]{Markup.Escape(custom.DisplayName)}[/] family={Markup.Escape(custom.BaselineFamily)} file={Markup.Escape(custom.SourceFileName)} learning={inLearning} carrier={inCarriers} explicit={inExplicit}");
        }
    }

    private void ShowEvolutionHelp()
    {
        AnsiConsole.MarkupLine("[bold yellow]Command: evolution[/]");
        AnsiConsole.WriteLine("Runs the full quantization search on a target model, then uses rank-safe isolation prediction to choose validated final hybrids.");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Usage:[/]");
        AnsiConsole.WriteLine("  mq evolution --model-dir \"<path>\" [options]");
        AnsiConsole.WriteLine("  mq evolution --config \"./config.default.yaml\"");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Arguments:[/]");
        AnsiConsole.MarkupLine("  [green]--model-dir[/]    Path to the model directory containing .safetensors files (Optional if set in YAML)");
        AnsiConsole.MarkupLine("  [green]--recheck-hardware-probe[/]    Force hardware/Q8 probe and update cached plan in SQLite (Optional)");
        AnsiConsole.MarkupLine("  [green]--use-imatrix[/]    Enable imatrix acquisition/build and allow imatrix-required search candidates (Optional)");
        AnsiConsole.MarkupLine("  [green]--allow-high-precision-hybrids[/]    Keep BF16/F16 explicit group candidates in final surviving combos (Optional, default false)");
        AnsiConsole.MarkupLine("  [green]--imatrix-force-rebuild[/]    Delete/rebuild canonical imatrix artifacts before run (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-url[/]    HTTPS URL for direct imatrix artifact download (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-repo[/]    Hugging Face dataset repo ID for imatrix generation (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-split[/]    Dataset split for HF/local dataset source metadata/build (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-config[/]    Optional dataset config name for HF datasets (Optional)");
        AnsiConsole.MarkupLine("  [green]--imatrix-dataset-local-file[/]    Full path to local .json/.jsonl dataset source (Optional)");
        AnsiConsole.MarkupLine("  [green]--selection-near-baseline-max-size-growth-percent[/]    Phase-2 size premium for replacing a smaller/higher-damage anchor (Optional; default = 1.0)");
        AnsiConsole.MarkupLine("  [green]--selection-interior-window-fractions[/]    Comma-separated phase-3 interior windows, e.g. 0.35,0.35 (Optional)");
        AnsiConsole.MarkupLine("  [green]--prediction-bit-stress-threshold-candidates[/]    Comma-separated interaction-fit thresholds, e.g. 4,5,6,7,8,9,10,11,12 (Optional)");
        AnsiConsole.MarkupLine("  [green]--output-dir[/]    Final export/output directory for selected survivor artifacts (Optional; default = <model>/MagicQuant/Final_Outputs)");
        AnsiConsole.MarkupLine("  [green]--output-name-prefix[/]    Output filename prefix for exported GGUF files (Optional; default = Model)");
        AnsiConsole.MarkupLine("  [green]--reuse-existing-final-artifacts[/]    Reuse valid final GGUFs only when exact file name + benchmark byte size match (Optional; default false)");
        AnsiConsole.MarkupLine("  [green]--allow-eight-bit-anchor-replacements[/]    Permit final prediction to try replacing 8-bit anchors like Q8_0 (Optional; default false)");
        AnsiConsole.MarkupLine("  [green]--export-external-learned-baselines[/]    Also locally rebuild/export pure learned external baselines such as Unsloth (Optional; default false)");
        AnsiConsole.MarkupLine("  [green]--selection-max-candidates-per-interior-window[/]    Candidate count retained per interior window (Optional; default = 1)");
        AnsiConsole.MarkupLine("  [green]--config[/]    Path to YAML runtime config. CLI flags override YAML values.");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Example:[/]");
        AnsiConsole.WriteLine("  mq evolution --model-dir \"C:\\Models\\Mistral-7B\"");
    }

    private static string ResolveAndValidateOutputDirectory()
    {
        string resolved;

        if (!string.IsNullOrWhiteSpace(Config.Current.Output.OutputDir))
        {
            resolved = Path.IsPathRooted(Config.Current.Output.OutputDir)
                ? Path.GetFullPath(Config.Current.Output.OutputDir)
                : Path.GetFullPath(Path.Combine(Cache.ModelMagicQuantDirectory!, Config.Current.Output.OutputDir));
        }
        else
        {
            resolved = Path.Combine(Cache.ModelMagicQuantDirectory!, "Final_Outputs");
        }

        Directory.CreateDirectory(resolved);

        string probe = Path.Combine(resolved, $".write_test_{Guid.NewGuid():N}.tmp");
        File.WriteAllText(probe, "ok");
        File.Delete(probe);

        return resolved;
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
