using System.Text.Json;
using System.Text.Json.Nodes;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Commands;

public sealed class CloneRepositoryQuants : ICommand
{
    private static readonly string[] ModelAdjacentFiles =
    [
        "generation_config.json",
        "config.json",
        "chat_template.jinja",
        "added_tokenizer.json",
        "LICENSE",
        "merges.txt",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json"
    ];

    private static readonly string[] CloneBenchmarkDomains = ["general"];
    private const string CloneTensorPolicyPropertyName = "cloneTensorPolicy";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            ShowHelp();
            return;
        }

        bool allowMissingManifestTensors = args.Any(a => string.Equals(a.Name, CloneManifestTensorMapBuildService.AllowMissingManifestTensorsFlag, StringComparison.OrdinalIgnoreCase));
        string? missingManifestBaseQuantName = Get(args, CloneManifestTensorMapBuildService.MissingManifestBaseQuantFlag);
        bool hasMissingManifestBaseQuantOverride = !string.IsNullOrWhiteSpace(missingManifestBaseQuantName);

        string? modelDirRaw = Get(args, "model-dir");
        if (string.IsNullOrWhiteSpace(modelDirRaw))
            modelDirRaw = Config.Current.Paths.ModelDir;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
            throw new InvalidOperationException("Clone mode requires --model-dir or paths.model_dir in YAML.");

        string fullModelPath = Path.GetFullPath(modelDirRaw);
        if (!Directory.Exists(fullModelPath))
            throw new DirectoryNotFoundException($"Model directory does not exist: {fullModelPath}");

        var safeTensorFiles = Directory.GetFiles(fullModelPath, "*.safetensors", SearchOption.TopDirectoryOnly);
        if (safeTensorFiles.Length == 0)
            throw new InvalidOperationException($"No .safetensors files were found in model directory: {fullModelPath}");

        Cache.ModelDirectory = fullModelPath;
        Cache.ModelMagicQuantDirectory = Path.Combine(fullModelPath, "MagicQuant");
        ModelRuntimePathService.InitializeForCurrentModel();
        await new ScratchStorageService(new ModelArtifactPathService()).CleanupStaleScratchArtifactsAsync();
        Cache.ForceRefreshHardwareProbe = Config.Current.Flags.ForceRefreshHardwareProbe;
        Cache.UseImatrix = Config.Current.Flags.UseImatrix;
        Cache.ForceImatrixRebuild = Config.Current.Flags.ForceImatrixRebuild;
        Cache.SuppressBenchmarkPersistence = true;

        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(false);
        RuntimeSearchSpace.AllowHighPrecisionHybrids = Config.Current.Flags.AllowHighPrecisionHybrids;

        JsonHelper.DetectAndSetTorchType(Cache.ModelDirectory);
        Directory.CreateDirectory(Cache.ModelMagicQuantDirectory);

        Cache.OutputDirectory = ResolveAndValidateOutputDirectory(args);

        AnsiConsole.Write(new Rule("[yellow]Repository Quant Clone Mode[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Model Path:   [blue]{Markup.Escape(Cache.ModelDirectory)}[/]");
        AnsiConsole.MarkupLine($"Work Path:    [blue]{Markup.Escape(Cache.ModelMagicQuantDirectory)}[/]");
        AnsiConsole.MarkupLine($"Export Path:  [blue]{Markup.Escape(Cache.OutputDirectory ?? "n/a")}[/]");
        AnsiConsole.MarkupLine($"Reuse final artifacts: {(Config.ReuseExistingFinalArtifacts ? "[green]yes[/]" : "[grey]no[/]")}");
        AnsiConsole.MarkupLine($"Allow missing manifest tensors: {(allowMissingManifestTensors || hasMissingManifestBaseQuantOverride ? "[yellow]yes[/]" : "[grey]no[/]")}");
        AnsiConsole.MarkupLine(hasMissingManifestBaseQuantOverride
            ? $"Missing-manifest base quant override: [yellow]{Markup.Escape(missingManifestBaseQuantName!)}[/]"
            : "Missing-manifest base quant override: [grey]none[/]");

        AnsiConsole.MarkupLine("Getting safetensors hash. This may take a bit, please wait...");
        Cache.CurrentModelId = MagicQuantModelId.GetOrCreateModelId(Cache.ModelDirectory);
        AnsiConsole.MarkupLine($"[green]Model ID Created/Found:[/] [cyan]{Markup.Escape(Cache.CurrentModelId)}[/]");

        await EnsureSqliteReadyAsync();

        var pyManager = new PythonManager(Cache.MagicQuantDirectory!);
        var hf = new HuggingFaceBaselineService(pyManager);
        var manifestService = new RepositoryCloneManifestService(hf);

        string? sourceRepo = Get(args, "source-repo") ?? Get(args, "clone-repo");
        string? sourceJson = Get(args, "source-json") ?? Get(args, "clone-json");

        var (manifest, manifestLocalPath, sourceDescription) = await manifestService.ResolveAsync(
            sourceRepo,
            sourceJson,
            Cache.ModelMagicQuantDirectory!,
            CancellationToken.None);

        var sourceClonePolicies = LoadCloneArtifactPolicies(manifestLocalPath);

        var benchmarkService = new BenchmarkService(pyManager);
        var quantizationService = new QuantizationService(benchmarkService);
        var imatrixService = new ImatrixService();
        var cloneBuildService = new CloneManifestTensorMapBuildService(quantizationService, imatrixService);

        string baseModelGgufPath = await quantizationService.EnsureBaseModelFileAsync(true);

        var sidecarService = new ModelSidecarArtifactService(pyManager);
        await sidecarService.EnsureMmprojArtifactAvailableAsync();

        var architectureFamilyService = new ArchitectureFamilyService(pyManager);
        await architectureFamilyService.EnsureCurrentArchitectureFamilyAsync(baseModelGgufPath);

        var tensorGroupProfileService = new TensorGroupProfileService();
        await tensorGroupProfileService.EnsureCurrentProfileAsync();

        var resolvedCustomBaselines = await hf.PrecheckAndRegisterConfiguredBaselinesAsync();
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

        var imatrixEnsureResult = await imatrixService.EnsureImatrixAsync(imatrixRequest);
        RuntimeSearchSpace.SetImatrixAvailability(imatrixEnsureResult.Available);

        var preCleanBenchmarkCache = Config.ReuseExistingFinalArtifacts
            ? LoadReusableCloneBenchmarkRows(Cache.OutputDirectory!)
            : new Dictionary<string, CloneBenchmarkCacheRow>(StringComparer.OrdinalIgnoreCase);

        bool canReuseEverything = TryLoadFullyReusableCloneRecords(
            outputDirectory: Cache.OutputDirectory!,
            manifest: manifest,
            benchmarkCache: preCleanBenchmarkCache,
            records: out var reusableRecords);

        await CleanOutputDirectoryAsync(Cache.OutputDirectory!, Config.ReuseExistingFinalArtifacts);

        var archivedManifestFiles = await CopySourceManifestFilesAsync(
            outputDirectory: Cache.OutputDirectory!,
            sourceManifestLocalPath: manifestLocalPath,
            sourceRepo: sourceRepo,
            sourceJson: sourceJson,
            huggingFace: hf,
            ct: CancellationToken.None);

        // Always place the clone source manifest in the output manifest folder, stamped with this clone source.
        manifest.SourceRepository = sourceRepo;
        manifest.SourceJson = string.IsNullOrWhiteSpace(sourceRepo) ? sourceDescription : manifest.SourceJson;
        string outputCloneManifestPath = MagicQuantManifestPathService.GetManifestFilePath(Cache.OutputDirectory!, MagicQuantManifestPathService.CloneConfigsFileName);
        await File.WriteAllTextAsync(outputCloneManifestPath, JsonSerializer.Serialize(manifest, JsonOptions));
        archivedManifestFiles.Add(MagicQuantManifestPathService.CloneConfigsFileName);

        var cloneBuildResults = new Dictionary<string, CloneManifestTensorMapBuildResult>(StringComparer.OrdinalIgnoreCase);

        var records = canReuseEverything
            ? reusableRecords
            : new List<CloneArtifactBuildRecord>();

        if (records.Count == manifest.Artifacts.Count)
        {
            AnsiConsole.MarkupLine($"[green]Reused clone artifacts and benchmark summary:[/] all {records.Count:N0} artifact(s) matched existing GGUF byte sizes and {MagicQuantManifestPathService.CloneBenchmarksFileName}.");
        }
        else
        {
            records.Clear();
            var benchmarkCache = preCleanBenchmarkCache;

            string q8QuantizationKey = BaselineQuants.Q8_0.Names[0];
            string nativeQuantizationKey = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
            string cloneBenchmarkRootDir = Path.Combine(Cache.ModelMagicQuantDirectory!, "CloneBenchmarks");
            string nativeBenchDir = Path.Combine(cloneBenchmarkRootDir, nativeQuantizationKey);
            string nativeLogitsDir = Path.Combine(nativeBenchDir, "logits");
            string pplCorporaDir = Path.Combine(cloneBenchmarkRootDir, "_ppl_corpora");

            bool loadedPlanFromCache = !Cache.ForceRefreshHardwareProbe &&
                                       await benchmarkService.TryInitializeDynamicExecutionPlanFromCacheAsync(
                                           q8QuantizationKey: q8QuantizationKey,
                                           nativeModelPath: baseModelGgufPath,
                                           nativeQuantizationKey: nativeQuantizationKey);

            if (loadedPlanFromCache)
            {
                AnsiConsole.MarkupLine("[grey]Clone mode reused the DB-backed hardware execution plan; no Q8 probe rebuild was needed.[/]");
            }
            else
            {
                AnsiConsole.MarkupLine(Cache.ForceRefreshHardwareProbe
                    ? "[yellow]Hardware probe refresh requested; rebuilding Q8 probe and updating SQLite execution-plan cache.[/]"
                    : "[grey]No reusable hardware execution-plan cache row found; building one Q8 probe and saving it to SQLite.[/]");

                await using var q8Lease = await quantizationService.BuildPureQ8ProbeLeaseAsync();
                await benchmarkService.EnsureDynamicExecutionPlanAsync(
                    q8ModelPath: q8Lease.GgufPath,
                    nativeModelPath: baseModelGgufPath,
                    q8QuantizationKey: q8QuantizationKey,
                    nativeQuantizationKey: nativeQuantizationKey,
                    discoveryTokenTarget: 8192,
                    forceRediscovery: Cache.ForceRefreshHardwareProbe);
            }

            await EnsureCloneNativeBenchmarkArtifactsReadyAsync(
                benchmarkService: benchmarkService,
                nativeModelQuant: HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()),
                nativeModelPath: baseModelGgufPath,
                nativeBenchDir: nativeBenchDir,
                nativeLogitsDir: nativeLogitsDir,
                pplCorporaDir: pplCorporaDir);

            foreach (var artifact in manifest.Artifacts)
            {
                string outputFile = Path.Combine(Cache.OutputDirectory!, artifact.FileName);
                string baseQuantName = string.IsNullOrWhiteSpace(artifact.BaseQuant)
                    ? artifact.QuantFamily
                    : artifact.BaseQuant;

                var sourcePolicy = sourceClonePolicies.GetValueOrDefault(artifact.FileName);
                string? artifactMissingManifestBaseQuantName = hasMissingManifestBaseQuantOverride
                    ? missingManifestBaseQuantName
                    : sourcePolicy?.MissingManifestBaseQuantName;
                bool artifactAllowMissingManifestTensors = allowMissingManifestTensors ||
                                                           hasMissingManifestBaseQuantOverride ||
                                                           sourcePolicy?.AllowMissingManifestTensors == true ||
                                                           !string.IsNullOrWhiteSpace(artifactMissingManifestBaseQuantName);

                AnsiConsole.Write(new Rule($"[yellow]Clone Artifact: {Markup.Escape(artifact.FileName)}[/]") { Justification = Justify.Left });

                if (TryReuseExistingCloneArtifactAndBenchmark(outputFile, artifact, benchmarkCache, out var cachedRecord))
                {
                    AnsiConsole.MarkupLine($"[green]Reused existing clone artifact + benchmark:[/] {Markup.Escape(outputFile)}");
                    records.Add(cachedRecord);
                    continue;
                }

                bool artifactExists = Config.ReuseExistingFinalArtifacts && File.Exists(outputFile) && new FileInfo(outputFile).Length > 0;
                if (artifactExists)
                {
                    AnsiConsole.MarkupLine($"[green]Reused existing clone GGUF:[/] {Markup.Escape(outputFile)} [grey](benchmark cache missing/stale; rebenchmarking only)[/]");
                }
                else
                {
                    var cloneBuildResult = await cloneBuildService.BuildAsync(
                        tensorTypes: artifact.TensorTypes,
                        outputPath: outputFile,
                        baseQuantName: baseQuantName,
                        allowMissingManifestTensors: artifactAllowMissingManifestTensors,
                        missingManifestBaseQuantName: artifactMissingManifestBaseQuantName,
                        forceRebuild: true);

                    cloneBuildResults[artifact.FileName] = cloneBuildResult;
                }

                var benchmarkBaseline = ResolveCloneBenchmarkBaseline(baseQuantName, artifact.QuantFamily);
                var quantForBenchmark = HybridQuant.CreatePureBaseline(benchmarkBaseline);
                bool benchmarkRequiresKld = benchmarkBaseline.UniqueId != BaselineQuants.NativeSourceUniqueId;

                var bench = await benchmarkService.RunAllBenchmarksAsync(
                    quantConfig: quantForBenchmark,
                    modelPath: outputFile,
                    benchDir: Path.Combine(cloneBenchmarkRootDir, Path.GetFileNameWithoutExtension(artifact.FileName)),
                    klLogitsDir: benchmarkRequiresKld ? nativeLogitsDir : null,
                    domainsOverride: CloneBenchmarkDomains);

                var general = bench.Perplexity.TryGetValue("general", out var ppl) ? ppl : null;

                records.Add(new CloneArtifactBuildRecord
                {
                    ManifestArtifact = artifact,
                    OutputPath = outputFile,
                    ActualSizeBytes = File.Exists(outputFile) ? (ulong)new FileInfo(outputFile).Length : 0UL,
                    Kld = general?.Kld,
                    Ppl = general?.Ppl,
                    PplDeltaPercent = null
                });
            }
        }

        ApplyCloneReferencePplDeltas(records);

        await CopyModelAdjacentFilesAsync(Cache.OutputDirectory!);
        await CopyImatrixArtifactsAsync(Cache.OutputDirectory!);
        await sidecarService.CopyMmprojArtifactsAsync(Cache.OutputDirectory!);

        await WriteCloneBenchmarkSummaryAsync(Cache.OutputDirectory!, records);
        await WriteResolvedCloneConfigManifestAsync(
            outputCloneManifestPath,
            records,
            sourceClonePolicies,
            cloneBuildResults,
            missingManifestBaseQuantName,
            hasMissingManifestBaseQuantOverride);
        archivedManifestFiles.Add(MagicQuantManifestPathService.CloneBenchmarksFileName);

        await new CloneReadmeGenerationService().GenerateAsync(
            Cache.OutputDirectory!,
            new DirectoryInfo(Cache.ModelDirectory!).Name,
            sourceDescription,
            !string.IsNullOrWhiteSpace(sourceRepo),
            records,
            archivedManifestFiles);

        await CleanCloneExportSidecarsAsync(Cache.OutputDirectory!);

        AnsiConsole.MarkupLine("[bold green]Repository quant clone complete.[/]");
    }

    private static async Task EnsureCloneNativeBenchmarkArtifactsReadyAsync(
        BenchmarkService benchmarkService,
        HybridQuant nativeModelQuant,
        string nativeModelPath,
        string nativeBenchDir,
        string nativeLogitsDir,
        string pplCorporaDir)
    {
        var status = ValidateCloneNativeBenchmarkEnvironment(
            nativeBenchDir: nativeBenchDir,
            nativeLogitsDir: nativeLogitsDir,
            pplCorporaDir: pplCorporaDir);

        if (status.IsValid)
        {
            AnsiConsole.MarkupLine("[grey]Clone native benchmark/KLD artifacts already exist and passed validation.[/]");
            return;
        }

        AnsiConsole.Write(new Rule("[yellow]Clone Native Benchmark/KLD Artifact Validation[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine("[yellow]Clone native benchmark/KLD artifacts are missing or incomplete.[/] Regenerating required artifacts.");
        PrintCloneNativeBenchmarkEnvironmentIssues(status);

        await ForceRegenerateCloneNativeBenchmarkArtifactsAsync(
            benchmarkService: benchmarkService,
            nativeModelQuant: nativeModelQuant,
            nativeModelPath: nativeModelPath,
            nativeBenchDir: nativeBenchDir,
            nativeLogitsDir: nativeLogitsDir);

        status = ValidateCloneNativeBenchmarkEnvironment(
            nativeBenchDir: nativeBenchDir,
            nativeLogitsDir: nativeLogitsDir,
            pplCorporaDir: pplCorporaDir);

        if (!status.IsValid)
        {
            var details = string.Join(
                Environment.NewLine,
                status.MissingOrInvalidArtifacts.Select(x => $"- {x}"));

            throw new InvalidOperationException(
                "Clone native benchmark/logit generation completed, but required native benchmark artifacts are still missing or invalid. " +
                "This is fatal because every cloned non-native benchmark requires complete native KLD logits." +
                Environment.NewLine +
                details);
        }

        AnsiConsole.MarkupLine("[green]Clone native benchmark/KLD artifacts validated.[/]");
    }

    private static async Task ForceRegenerateCloneNativeBenchmarkArtifactsAsync(
        BenchmarkService benchmarkService,
        HybridQuant nativeModelQuant,
        string nativeModelPath,
        string nativeBenchDir,
        string nativeLogitsDir)
    {
        if (Directory.Exists(nativeBenchDir))
        {
            AnsiConsole.MarkupLine(
                $"[grey]Clearing incomplete/stale clone native benchmark directory:[/] {Markup.Escape(nativeBenchDir)}");

            Directory.Delete(nativeBenchDir, recursive: true);
        }

        Directory.CreateDirectory(nativeBenchDir);
        Directory.CreateDirectory(nativeLogitsDir);

        bool previousSuppressBenchmarkPersistence = Cache.SuppressBenchmarkPersistence;

        try
        {
            Cache.SuppressBenchmarkPersistence = true;

            await benchmarkService.RunAllBenchmarksAsync(
                quantConfig: nativeModelQuant,
                modelPath: nativeModelPath,
                benchDir: nativeBenchDir,
                klLogitsDir: nativeLogitsDir,
                saveLogits: true,
                domainsOverride: CloneBenchmarkDomains);
        }
        finally
        {
            Cache.SuppressBenchmarkPersistence = previousSuppressBenchmarkPersistence;
        }
    }

    private static CloneNativeBenchmarkEnvironmentStatus ValidateCloneNativeBenchmarkEnvironment(
        string nativeBenchDir,
        string nativeLogitsDir,
        string pplCorporaDir)
    {
        var issues = new List<string>();

        if (string.IsNullOrWhiteSpace(nativeBenchDir))
            issues.Add("Clone native benchmark directory path is null/empty.");
        else if (!Directory.Exists(nativeBenchDir))
            issues.Add($"Clone native benchmark directory does not exist: {nativeBenchDir}");

        if (string.IsNullOrWhiteSpace(nativeLogitsDir))
            issues.Add("Clone native KLD logits directory path is null/empty.");
        else if (!Directory.Exists(nativeLogitsDir))
            issues.Add($"Clone native KLD logits directory does not exist: {nativeLogitsDir}");

        if (string.IsNullOrWhiteSpace(pplCorporaDir))
            issues.Add("Clone _ppl_corpora directory path is null/empty.");
        else if (!Directory.Exists(pplCorporaDir))
            issues.Add($"Clone _ppl_corpora directory does not exist: {pplCorporaDir}");
        else if (!Directory.EnumerateFiles(pplCorporaDir, "*", SearchOption.AllDirectories).Any())
            issues.Add($"Clone _ppl_corpora directory exists but contains no files: {pplCorporaDir}");

        foreach (var domain in CloneBenchmarkDomains.OrderBy(x => x, StringComparer.Ordinal))
        {
            if (!string.IsNullOrWhiteSpace(nativeBenchDir) && Directory.Exists(nativeBenchDir))
            {
                var pplLog = Path.Combine(nativeBenchDir, $"perplexity_{domain}.log");

                if (!File.Exists(pplLog))
                    issues.Add($"Missing clone native perplexity log for domain '{domain}': {pplLog}");
                else if (new FileInfo(pplLog).Length <= 0)
                    issues.Add($"Clone native perplexity log is empty for domain '{domain}': {pplLog}");
            }

            if (!string.IsNullOrWhiteSpace(nativeLogitsDir) && Directory.Exists(nativeLogitsDir))
            {
                var logitsFile = Path.Combine(nativeLogitsDir, $"kld_logits_{domain}.bin");

                if (!File.Exists(logitsFile))
                    issues.Add($"Missing clone native KLD logits for domain '{domain}': {logitsFile}");
                else if (new FileInfo(logitsFile).Length <= 0)
                    issues.Add($"Clone native KLD logits file is empty for domain '{domain}': {logitsFile}");
            }
        }

        return new CloneNativeBenchmarkEnvironmentStatus(
            IsValid: issues.Count == 0,
            MissingOrInvalidArtifacts: issues);
    }

    private static void PrintCloneNativeBenchmarkEnvironmentIssues(CloneNativeBenchmarkEnvironmentStatus status)
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

    private static BaselineQuants ResolveCloneBenchmarkBaseline(string? baseQuantName, string? quantFamily)
    {
        var native = BaselineQuants.GetBF16Quant();

        foreach (var raw in new[] { baseQuantName, quantFamily })
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;

            string name = raw.Trim();

            if (IsNativeBaselineName(name, native))
                return native;

            var standard = BaselineQuants.ResolveBuiltInStandardBaseline(name);
            if (standard != null)
                return standard;

            var recognized = BaselineQuants.GetAllRecognizedBaselines()
                .FirstOrDefault(x => BaselineNameMatches(x, name));

            if (recognized != null)
                return recognized;
        }

        return BaselineQuants.Q8_0;
    }

    private static bool IsNativeBaselineName(string name, BaselineQuants native)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        string normalized = name.Trim();

        return string.Equals(normalized, "native", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(normalized, "native_source", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(normalized, "source", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(normalized, (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString(), StringComparison.OrdinalIgnoreCase) ||
               BaselineNameMatches(native, normalized);
    }

    private static bool BaselineNameMatches(BaselineQuants baseline, string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        string normalized = name.Trim();

        if (baseline.Names.Any(x => string.Equals(x, normalized, StringComparison.OrdinalIgnoreCase)))
            return true;

        if (string.Equals(baseline.QuantizeBaseArgumentName, normalized, StringComparison.OrdinalIgnoreCase))
            return true;

        if (string.Equals(baseline.CanonicalKey, normalized, StringComparison.OrdinalIgnoreCase))
            return true;

        if (!string.IsNullOrWhiteSpace(baseline.ShortSourceName) &&
            string.Equals(baseline.ShortSourceName, normalized, StringComparison.OrdinalIgnoreCase))
            return true;

        if (!string.IsNullOrWhiteSpace(baseline.SourceFileName) &&
            string.Equals(Path.GetFileNameWithoutExtension(baseline.SourceFileName), normalized, StringComparison.OrdinalIgnoreCase))
            return true;

        if (baseline.PrimaryTensorWeightScheme.Names.Any(x => string.Equals(x, normalized, StringComparison.OrdinalIgnoreCase)))
            return true;

        return false;
    }

    private static void ApplyCloneReferencePplDeltas(IReadOnlyList<CloneArtifactBuildRecord> records)
    {
        if (records.Count == 0)
            return;

        var reference = records.FirstOrDefault(IsCloneQ8ReferenceRecord);
        if (reference == null || !reference.Ppl.HasValue || reference.Ppl.Value <= 0d)
        {
            AnsiConsole.MarkupLine("[grey]Clone PPL delta reference unavailable; keeping any cached/source PPL delta values as-is.[/]");
            return;
        }

        double referencePpl = reference.Ppl.Value;
        foreach (var record in records)
        {
            if (record.Ppl.HasValue && record.Ppl.Value > 0d)
                record.PplDeltaPercent = FinalReleaseMetadataService.CalculatePplDeltaPercent(record.Ppl.Value, referencePpl);
        }

        AnsiConsole.MarkupLine(
            $"[grey]Clone PPL deltas calculated from final Q8 artifact:[/] {Markup.Escape(reference.ManifestArtifact.FileName)}");
    }

    private static bool IsCloneQ8ReferenceRecord(CloneArtifactBuildRecord record)
    {
        var artifact = record.ManifestArtifact;

        foreach (var raw in new[]
                 {
                     artifact.BaseQuant,
                     artifact.QuantFamily,
                     artifact.DisplayName,
                     Path.GetFileNameWithoutExtension(artifact.FileName)
                 })
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;

            if (BaselineNameMatches(BaselineQuants.Q8_0, raw.Trim()))
                return true;
        }

        return false;
    }

    private static bool TryLoadFullyReusableCloneRecords(
        string outputDirectory,
        MagicQuantCloneManifest manifest,
        IReadOnlyDictionary<string, CloneBenchmarkCacheRow> benchmarkCache,
        out List<CloneArtifactBuildRecord> records)
    {
        records = new List<CloneArtifactBuildRecord>();

        if (!Config.ReuseExistingFinalArtifacts || benchmarkCache.Count == 0)
            return false;

        foreach (var artifact in manifest.Artifacts)
        {
            string outputFile = Path.Combine(outputDirectory, artifact.FileName);
            if (!TryReuseExistingCloneArtifactAndBenchmark(outputFile, artifact, benchmarkCache, out var record))
            {
                records.Clear();
                return false;
            }

            records.Add(record);
        }

        return records.Count == manifest.Artifacts.Count;
    }

    private static bool TryReuseExistingCloneArtifactAndBenchmark(
        string outputFile,
        MagicQuantCloneArtifact artifact,
        IReadOnlyDictionary<string, CloneBenchmarkCacheRow> benchmarkCache,
        out CloneArtifactBuildRecord record)
    {
        record = default!;

        if (!Config.ReuseExistingFinalArtifacts)
            return false;

        if (!File.Exists(outputFile))
            return false;

        var info = new FileInfo(outputFile);
        if (info.Length <= 0)
            return false;

        if (!benchmarkCache.TryGetValue(artifact.FileName, out var cached))
            return false;

        if (cached.SizeBytes != (ulong)info.Length)
            return false;

        record = new CloneArtifactBuildRecord
        {
            ManifestArtifact = artifact,
            OutputPath = outputFile,
            ActualSizeBytes = cached.SizeBytes,
            Kld = cached.Kld,
            Ppl = cached.Ppl,
            PplDeltaPercent = cached.PplDeltaPercent
        };

        return true;
    }

    private static Dictionary<string, CloneBenchmarkCacheRow> LoadReusableCloneBenchmarkRows(string outputDirectory)
    {
        string path = MagicQuantManifestPathService.GetManifestFilePath(outputDirectory, MagicQuantManifestPathService.CloneBenchmarksFileName);
        if (!File.Exists(path))
            return new Dictionary<string, CloneBenchmarkCacheRow>(StringComparer.OrdinalIgnoreCase);

        try
        {
            var rows = JsonSerializer.Deserialize<List<CloneBenchmarkCacheRow>>(File.ReadAllText(path), JsonOptions)
                       ?? new List<CloneBenchmarkCacheRow>();

            return rows
                .Where(x => !string.IsNullOrWhiteSpace(x.FileName) && x.SizeBytes > 0)
                .GroupBy(x => x.FileName, StringComparer.OrdinalIgnoreCase)
                .ToDictionary(g => g.Key, g => g.First(), StringComparer.OrdinalIgnoreCase);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]Existing clone benchmark summary could not be reused:[/] {Markup.Escape(ex.Message)}");
            return new Dictionary<string, CloneBenchmarkCacheRow>(StringComparer.OrdinalIgnoreCase);
        }
    }

    private static Dictionary<string, CloneArtifactPolicy> LoadCloneArtifactPolicies(string manifestPath)
    {
        var policies = new Dictionary<string, CloneArtifactPolicy>(StringComparer.OrdinalIgnoreCase);

        if (string.IsNullOrWhiteSpace(manifestPath) || !File.Exists(manifestPath))
            return policies;

        try
        {
            var root = JsonNode.Parse(File.ReadAllText(manifestPath)) as JsonObject;
            var artifacts = TryGetProperty(root, "artifacts") as JsonArray;
            if (artifacts == null)
                return policies;

            foreach (var node in artifacts.OfType<JsonObject>())
            {
                string? fileName = TryGetString(node, "fileName");
                if (string.IsNullOrWhiteSpace(fileName))
                    continue;

                var policyNode = TryGetProperty(node, CloneTensorPolicyPropertyName) as JsonObject;
                if (policyNode == null)
                    continue;

                policies[fileName] = new CloneArtifactPolicy(
                    AllowMissingManifestTensors: TryGetBool(policyNode, "allowMissingManifestTensors"),
                    MissingManifestBaseQuantName: TryGetString(policyNode, "missingManifestBaseQuant"));
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]Clone tensor policy metadata could not be read from source manifest:[/] {Markup.Escape(ex.Message)}");
        }

        return policies;
    }

    private static async Task WriteResolvedCloneConfigManifestAsync(
        string outputCloneManifestPath,
        IReadOnlyCollection<CloneArtifactBuildRecord> records,
        IReadOnlyDictionary<string, CloneArtifactPolicy> sourcePolicies,
        IReadOnlyDictionary<string, CloneManifestTensorMapBuildResult> cloneBuildResults,
        string? cliMissingManifestBaseQuantName,
        bool hasCliMissingManifestBaseQuantOverride)
    {
        if (!File.Exists(outputCloneManifestPath))
            return;

        var root = JsonNode.Parse(await File.ReadAllTextAsync(outputCloneManifestPath)) as JsonObject;
        var artifacts = TryGetProperty(root, "artifacts") as JsonArray;
        if (root == null || artifacts == null)
            return;

        int policiesWritten = 0;
        var recordByFileName = records
            .Where(x => !string.IsNullOrWhiteSpace(x.ManifestArtifact.FileName))
            .GroupBy(x => x.ManifestArtifact.FileName, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.OrdinalIgnoreCase);

        foreach (var artifactNode in artifacts.OfType<JsonObject>())
        {
            string? fileName = TryGetString(artifactNode, "fileName");
            if (string.IsNullOrWhiteSpace(fileName))
                continue;

            recordByFileName.TryGetValue(fileName, out var record);
            cloneBuildResults.TryGetValue(fileName, out var buildResult);
            sourcePolicies.TryGetValue(fileName, out var sourcePolicy);

            bool allowMissing = buildResult?.UsedManifestSubset == true ||
                                sourcePolicy?.AllowMissingManifestTensors == true ||
                                hasCliMissingManifestBaseQuantOverride;
            string? missingBaseQuant = buildResult?.UsedManifestSubset == true
                ? buildResult.EffectiveBaseQuantName
                : hasCliMissingManifestBaseQuantOverride
                    ? cliMissingManifestBaseQuantName
                    : sourcePolicy?.MissingManifestBaseQuantName;

            if (!allowMissing && string.IsNullOrWhiteSpace(missingBaseQuant))
            {
                artifactNode.Remove(CloneTensorPolicyPropertyName);
                continue;
            }

            var policyNode = new JsonObject
            {
                ["allowMissingManifestTensors"] = allowMissing,
                ["missingManifestBaseQuant"] = string.IsNullOrWhiteSpace(missingBaseQuant) ? null : missingBaseQuant,
                ["generatedAtUtc"] = DateTimeOffset.UtcNow.ToString("O")
            };

            if (record?.ActualSizeBytes > 0)
                policyNode["actualSizeBytes"] = (long)Math.Min(record.ActualSizeBytes, long.MaxValue);

            if (buildResult?.UsedManifestSubset == true)
            {
                policyNode["missingManifestTensorCount"] = buildResult.MissingInManifest.Count;
                policyNode["missingManifestTensors"] = new JsonArray(buildResult.MissingInManifest.Select(x => JsonValue.Create(x)).ToArray<JsonNode?>());
            }
            else if (sourcePolicy?.AllowMissingManifestTensors == true || !string.IsNullOrWhiteSpace(sourcePolicy?.MissingManifestBaseQuantName))
            {
                policyNode["inheritedFromSourceManifest"] = true;
            }
            else if (hasCliMissingManifestBaseQuantOverride)
            {
                policyNode["createdFromCliOverride"] = true;
            }

            artifactNode[CloneTensorPolicyPropertyName] = policyNode;
            policiesWritten++;
        }

        await File.WriteAllTextAsync(outputCloneManifestPath, root.ToJsonString(JsonOptions));
        AnsiConsole.MarkupLine($"[green]Resolved clone config manifest updated:[/] {Markup.Escape(outputCloneManifestPath)} [grey](policies={policiesWritten:N0})[/]");
    }

    private static JsonNode? TryGetProperty(JsonObject? obj, string name)
    {
        if (obj == null)
            return null;

        foreach (var kv in obj)
        {
            if (string.Equals(kv.Key, name, StringComparison.OrdinalIgnoreCase))
                return kv.Value;
        }

        return null;
    }

    private static string? TryGetString(JsonObject obj, string name)
        => TryGetProperty(obj, name)?.GetValue<string>();

    private static bool TryGetBool(JsonObject obj, string name)
    {
        var node = TryGetProperty(obj, name);
        if (node == null)
            return false;

        try
        {
            return node.GetValue<bool>();
        }
        catch
        {
            return bool.TryParse(node.ToString(), out var value) && value;
        }
    }

    private static async Task<HashSet<string>> CopySourceManifestFilesAsync(
        string outputDirectory,
        string sourceManifestLocalPath,
        string? sourceRepo,
        string? sourceJson,
        HuggingFaceBaselineService huggingFace,
        CancellationToken ct)
    {
        string targetManifestDir = MagicQuantManifestPathService.EnsureManifestDirectory(outputDirectory);
        var copied = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(sourceRepo))
        {
            foreach (var fileName in MagicQuantManifestPathService.KnownManifestFileNames)
            {
                if (string.Equals(fileName, MagicQuantManifestPathService.CloneBenchmarksFileName, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (await TryDownloadOptionalSourceManifestFileAsync(sourceRepo.Trim(), fileName, Path.Combine(targetManifestDir, fileName), huggingFace, ct))
                    copied.Add(fileName);
            }

            return copied;
        }

        string sourceDir = Path.GetDirectoryName(sourceManifestLocalPath) ?? string.Empty;
        if (Directory.Exists(sourceDir))
        {
            foreach (var file in Directory.EnumerateFiles(sourceDir, "magicquant*.json", SearchOption.TopDirectoryOnly))
            {
                string fileName = Path.GetFileName(file);
                if (string.Equals(fileName, MagicQuantManifestPathService.CloneBenchmarksFileName, StringComparison.OrdinalIgnoreCase))
                    continue;

                File.Copy(file, Path.Combine(targetManifestDir, fileName), overwrite: true);
                copied.Add(fileName);
            }
        }

        return copied;
    }

    private static async Task<bool> TryDownloadOptionalSourceManifestFileAsync(
        string repoId,
        string fileName,
        string destinationPath,
        HuggingFaceBaselineService huggingFace,
        CancellationToken ct)
    {
        var candidates = new[]
        {
            MagicQuantManifestPathService.RelativeManifestPath(fileName),
            fileName
        };

        foreach (var candidate in candidates)
        {
            try
            {
                await huggingFace.DownloadRepositoryFileAsync(
                    repoId: repoId,
                    fileName: candidate,
                    destinationPath: destinationPath,
                    forceRedownload: true,
                    ct: ct);

                AnsiConsole.MarkupLine($"[green]Archived source manifest file:[/] {Markup.Escape(candidate)}");
                return true;
            }
            catch
            {
                // Optional source manifest sidecars may not exist, especially in older repos.
            }
        }

        AnsiConsole.MarkupLine($"[grey]Optional source manifest file unavailable:[/] {Markup.Escape(fileName)}");
        return false;
    }

    private static async Task WriteCloneBenchmarkSummaryAsync(string outputDirectory, IReadOnlyCollection<CloneArtifactBuildRecord> records)
    {
        string path = MagicQuantManifestPathService.GetManifestFilePath(outputDirectory, MagicQuantManifestPathService.CloneBenchmarksFileName);

        var payload = records
            .OrderBy(x => x.Kld ?? double.MaxValue)
            .ThenBy(x => x.ActualSizeBytes)
            .Select(x => new CloneBenchmarkCacheRow
            {
                FileName = x.ManifestArtifact.FileName,
                DisplayName = x.ManifestArtifact.DisplayName,
                Provider = x.ManifestArtifact.Provider,
                QuantFamily = x.ManifestArtifact.QuantFamily,
                BaseQuant = x.ManifestArtifact.BaseQuant,
                Kld = x.Kld,
                Ppl = x.Ppl,
                PplDeltaPercent = x.PplDeltaPercent,
                SizeBytes = x.ActualSizeBytes,
                SizeGB = x.ActualSizeBytes / 1000d / 1000d / 1000d,
                SizeGiB = x.ActualSizeBytes / 1024d / 1024d / 1024d,
                SourceKld = x.ManifestArtifact.SourceKld,
                SourcePpl = x.ManifestArtifact.SourcePpl,
                SourceSizeBytes = x.ManifestArtifact.SourceSizeBytes
            })
            .ToList();

        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(payload, JsonOptions));
        AnsiConsole.MarkupLine($"[green]Clone benchmark summary generated:[/] {Markup.Escape(path)}");
    }

    private static async Task CopyModelAdjacentFilesAsync(string outputDirectory)
    {
        foreach (var fileName in ModelAdjacentFiles)
        {
            string source = Path.Combine(Cache.ModelDirectory!, fileName);
            string target = Path.Combine(outputDirectory, fileName);

            if (!File.Exists(source))
                continue;

            File.Copy(source, target, overwrite: true);
            await Task.Yield();
            AnsiConsole.MarkupLine($"[green]Copied model-adjacent file:[/] {Markup.Escape(fileName)}");
        }
    }

    private static Task CopyImatrixArtifactsAsync(string outputDirectory)
    {
        if (!Cache.IsImatrixAvailable || string.IsNullOrWhiteSpace(Cache.ActiveImatrixPath))
            return Task.CompletedTask;

        string target = Path.Combine(outputDirectory, "imatrix.dat");
        File.Copy(Cache.ActiveImatrixPath!, target, overwrite: true);
        AnsiConsole.MarkupLine($"[green]Copied imatrix artifact:[/] {Markup.Escape(target)}");
        return Task.CompletedTask;
    }

    private static async Task CleanOutputDirectoryAsync(string outputDirectory, bool preserveReusableGgufs)
    {
        Directory.CreateDirectory(outputDirectory);

        foreach (var file in Directory.EnumerateFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly))
        {
            if (preserveReusableGgufs &&
                string.Equals(Path.GetExtension(file), ".gguf", StringComparison.OrdinalIgnoreCase) &&
                new FileInfo(file).Length > 0)
            {
                continue;
            }

            await HardDeleteHelper.DeleteFileIfExistsAsync(file);
        }

        foreach (var directory in Directory.EnumerateDirectories(outputDirectory, "*", SearchOption.TopDirectoryOnly))
            await HardDeleteHelper.DeleteDirectoryIfExistsAsync(directory, CancellationToken.None);

        AnsiConsole.MarkupLine(preserveReusableGgufs
            ? $"[grey]Cleaned clone export metadata/non-GGUF files; preserved existing non-empty GGUFs for reuse validation:[/] {Markup.Escape(outputDirectory)}"
            : $"[grey]Cleaned clone export directory:[/] {Markup.Escape(outputDirectory)}");
    }

    private static async Task CleanCloneExportSidecarsAsync(string outputDirectory)
    {
        string[] patterns =
        [
            "*.success.json",
            "*.quantize.log",
            "*.convert.log",
            "imatrix.success.json",
            "imatrix.metadata.json",
            "imatrix.build.log"
        ];

        foreach (var pattern in patterns)
        {
            foreach (var file in Directory.EnumerateFiles(outputDirectory, pattern, SearchOption.TopDirectoryOnly))
                await HardDeleteHelper.DeleteFileIfExistsAsync(file);
        }

        AnsiConsole.MarkupLine($"[grey]Cleaned clone export sidecar success/log files:[/] {Markup.Escape(outputDirectory)}");
    }

    private static async Task EnsureSqliteReadyAsync()
    {
        await using var db = new MagicQuantContext();
        await db.Database.MigrateAsync();
    }

    private static string ResolveAndValidateOutputDirectory(IReadOnlyCollection<CliArg> args)
    {
        string? explicitOutput = Get(args, "output-dir");
        string outputDir = !string.IsNullOrWhiteSpace(explicitOutput)
            ? explicitOutput!
            : !string.IsNullOrWhiteSpace(Config.OutputDirectory)
                ? Config.OutputDirectory!
                : Path.Combine(Cache.ModelMagicQuantDirectory!, "FinalOutput");

        outputDir = Path.GetFullPath(outputDir);
        Directory.CreateDirectory(outputDir);
        return outputDir;
    }

    private static string? Get(IReadOnlyCollection<CliArg> args, string name)
        => args.FirstOrDefault(a => string.Equals(a.Name, name, StringComparison.OrdinalIgnoreCase))?.Value;

    private static void ShowHelp()
    {
        AnsiConsole.MarkupLine("[bold yellow]Command: clone-repository-quants[/]");
        AnsiConsole.MarkupLine("Rebuilds the final GGUF list from a MagicQuant-compatible tensor config manifest without running the evolution/search pipeline.");
        AnsiConsole.MarkupLine("Usage:");
        AnsiConsole.MarkupLine("  mq clone-repository-quants --model-dir \"<path>\" --architecture-family \"<family>\" --source-repo \"owner/repo\" [--output-dir \"<path>\"] [--reuse-existing-final-artifacts]");
        AnsiConsole.MarkupLine("  mq clone-repository-quants --model-dir \"<path>\" --architecture-family \"<family>\" --source-json \"<path-or-url>\" [--output-dir \"<path>\"] [--reuse-existing-final-artifacts]");
        AnsiConsole.MarkupLine("Options:");
        AnsiConsole.MarkupLine($"  --source-repo       Hugging Face repo containing {MagicQuantManifestPathService.RelativeManifestPath(MagicQuantManifestPathService.CloneConfigsFileName)} or legacy root {MagicQuantManifestPathService.CloneConfigsFileName}");
        AnsiConsole.MarkupLine("  --source-json       Local or http(s) path to magicquant.clone-configs.json");
        AnsiConsole.MarkupLine("  --use-imatrix       Use configured/provided imatrix for the cloned model");
        AnsiConsole.MarkupLine("  --reuse-existing-final-artifacts  Reuse matching existing GGUFs and matching clone benchmark JSON rows");
        AnsiConsole.MarkupLine("  --allow-missing-manifest-tensors  Allow clone manifests that are strict subsets of the current model tensor list; extra source tensors receive no explicit --tensor-type override and fall through to base quantization");
        AnsiConsole.MarkupLine("  --missing-manifest-base-quant <quant>  Allow strict-subset clone manifests and use this llama.cpp base quant for tensors absent from the manifest, e.g. Q8_0");
        AnsiConsole.MarkupLine("  --recheck-hardware-probe / --force-refresh-hardware-probe  Force Q8/native hardware probe and refresh the SQLite execution-plan cache");
    }

    private sealed record CloneNativeBenchmarkEnvironmentStatus(
        bool IsValid,
        IReadOnlyList<string> MissingOrInvalidArtifacts);

    private sealed record CloneArtifactPolicy(
        bool AllowMissingManifestTensors,
        string? MissingManifestBaseQuantName);

    private sealed class CloneBenchmarkCacheRow
    {
        public string FileName { get; set; } = string.Empty;
        public string DisplayName { get; set; } = string.Empty;
        public string Provider { get; set; } = string.Empty;
        public string QuantFamily { get; set; } = string.Empty;
        public string BaseQuant { get; set; } = string.Empty;
        public double? Kld { get; set; }
        public double? Ppl { get; set; }
        public double? PplDeltaPercent { get; set; }
        public ulong SizeBytes { get; set; }
        public double SizeGB { get; set; }
        public double SizeGiB { get; set; }
        public double? SourceKld { get; set; }
        public double? SourcePpl { get; set; }
        public ulong? SourceSizeBytes { get; set; }
    }
}
