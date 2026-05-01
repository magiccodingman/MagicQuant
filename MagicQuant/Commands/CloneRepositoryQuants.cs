using System.Text.Json;
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

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            ShowHelp();
            return;
        }

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
        Cache.ForceRelearnBaselineTensorMappings = false;
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
    
        AnsiConsole.MarkupLine($"Getting safetensors hash. This may take a bit, please wait...");
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

        var benchmarkService = new BenchmarkService(pyManager);
        var quantizationService = new QuantizationService(benchmarkService);
        var imatrixService = new ImatrixService();

        string baseModelGgufPath = await quantizationService.EnsureBaseModelFileAsync(true);

        var sidecarService = new ModelSidecarArtifactService(pyManager);
        await sidecarService.EnsureMmprojArtifactAvailableAsync();

        var architectureFamilyService = new ArchitectureFamilyService(pyManager);
        await architectureFamilyService.EnsureCurrentArchitectureFamilyAsync(baseModelGgufPath);

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

        await CleanOutputDirectoryAsync(Cache.OutputDirectory!);

        // Always place the clone source manifest in the output, but stamp it with this clone source.
        manifest.SourceRepository = sourceRepo;
        manifest.SourceJson = string.IsNullOrWhiteSpace(sourceRepo) ? sourceDescription : manifest.SourceJson;
        await File.WriteAllTextAsync(
            Path.Combine(Cache.OutputDirectory!, CloneConfigManifestGenerationService.FileName),
            JsonSerializer.Serialize(manifest, JsonOptions));

        string q8QuantizationKey = BaselineQuants.Q8_0.Names[0];
        string nativeQuantizationKey = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        string cloneBenchmarkRootDir = Path.Combine(Cache.ModelMagicQuantDirectory!, "CloneBenchmarks");
        string nativeBenchDir = Path.Combine(cloneBenchmarkRootDir, nativeQuantizationKey);
        string nativeLogitsDir = Path.Combine(nativeBenchDir, "logits");
        string pplCorporaDir = Path.Combine(cloneBenchmarkRootDir, "_ppl_corpora");

        await using var q8Lease = await quantizationService.BuildPureQ8ProbeLeaseAsync();
        await benchmarkService.EnsureDynamicExecutionPlanAsync(
            q8ModelPath: q8Lease.GgufPath,
            nativeModelPath: baseModelGgufPath,
            q8QuantizationKey: q8QuantizationKey,
            nativeQuantizationKey: nativeQuantizationKey,
            discoveryTokenTarget: 8192,
            forceRediscovery: Cache.ForceRefreshHardwareProbe);

        await EnsureCloneNativeBenchmarkArtifactsReadyAsync(
            benchmarkService: benchmarkService,
            nativeModelQuant: HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()),
            nativeModelPath: baseModelGgufPath,
            nativeBenchDir: nativeBenchDir,
            nativeLogitsDir: nativeLogitsDir,
            pplCorporaDir: pplCorporaDir);

        var q8Reference = await benchmarkService.RunAllBenchmarksAsync(
            quantConfig: HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0),
            modelPath: q8Lease.GgufPath,
            benchDir: Path.Combine(cloneBenchmarkRootDir, "_reference_q8"),
            klLogitsDir: nativeLogitsDir,
            domainsOverride: CloneBenchmarkDomains);

        double? referencePpl = q8Reference.Perplexity.TryGetValue("general", out var q8Ppl) && q8Ppl.Ppl > 0
            ? q8Ppl.Ppl
            : null;

        var records = new List<CloneArtifactBuildRecord>();

        foreach (var artifact in manifest.Artifacts)
        {
            string outputFile = Path.Combine(Cache.OutputDirectory!, artifact.FileName);
            string baseQuantName = string.IsNullOrWhiteSpace(artifact.BaseQuant)
                ? artifact.QuantFamily
                : artifact.BaseQuant;

            AnsiConsole.Write(new Rule($"[yellow]Clone Artifact: {Markup.Escape(artifact.FileName)}[/]") { Justification = Justify.Left });

            await quantizationService.BuildExportArtifactFromExactTensorMapAsync(
                tensorTypes: artifact.TensorTypes,
                outputPath: outputFile,
                baseQuantName: baseQuantName,
                forceRebuild: true);

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
                PplDeltaPercent = general != null && referencePpl is > 0d
                    ? FinalReleaseMetadataService.CalculatePplDeltaPercent(general.Ppl, referencePpl)
                    : null
            });
        }

        await CopyModelAdjacentFilesAsync(Cache.OutputDirectory!);
        await CopyImatrixArtifactsAsync(Cache.OutputDirectory!);
        await sidecarService.CopyMmprojArtifactsAsync(Cache.OutputDirectory!);

        await new CloneReadmeGenerationService().GenerateAsync(
            Cache.OutputDirectory!,
            new DirectoryInfo(Cache.ModelDirectory!).Name,
            sourceDescription,
            !string.IsNullOrWhiteSpace(sourceRepo),
            records);

        await WriteCloneBenchmarkSummaryAsync(Cache.OutputDirectory!, records);

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
            // Clone/export mode must not contaminate SQLite benchmark truth, but it still
            // needs the same native KLD base-logit artifacts that evolution prepares.
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
        {
            issues.Add("Clone native benchmark directory path is null/empty.");
        }
        else if (!Directory.Exists(nativeBenchDir))
        {
            issues.Add($"Clone native benchmark directory does not exist: {nativeBenchDir}");
        }

        if (string.IsNullOrWhiteSpace(nativeLogitsDir))
        {
            issues.Add("Clone native KLD logits directory path is null/empty.");
        }
        else if (!Directory.Exists(nativeLogitsDir))
        {
            issues.Add($"Clone native KLD logits directory does not exist: {nativeLogitsDir}");
        }

        if (string.IsNullOrWhiteSpace(pplCorporaDir))
        {
            issues.Add("Clone _ppl_corpora directory path is null/empty.");
        }
        else if (!Directory.Exists(pplCorporaDir))
        {
            issues.Add($"Clone _ppl_corpora directory does not exist: {pplCorporaDir}");
        }
        else if (!Directory.EnumerateFiles(pplCorporaDir, "*", SearchOption.AllDirectories).Any())
        {
            issues.Add($"Clone _ppl_corpora directory exists but contains no files: {pplCorporaDir}");
        }

        foreach (var domain in CloneBenchmarkDomains.OrderBy(x => x, StringComparer.Ordinal))
        {
            if (!string.IsNullOrWhiteSpace(nativeBenchDir) && Directory.Exists(nativeBenchDir))
            {
                var pplLog = Path.Combine(nativeBenchDir, $"perplexity_{domain}.log");

                if (!File.Exists(pplLog))
                {
                    issues.Add($"Missing clone native perplexity log for domain '{domain}': {pplLog}");
                }
                else if (new FileInfo(pplLog).Length <= 0)
                {
                    issues.Add($"Clone native perplexity log is empty for domain '{domain}': {pplLog}");
                }
            }

            if (!string.IsNullOrWhiteSpace(nativeLogitsDir) && Directory.Exists(nativeLogitsDir))
            {
                var logitsFile = Path.Combine(nativeLogitsDir, $"kld_logits_{domain}.bin");

                if (!File.Exists(logitsFile))
                {
                    issues.Add($"Missing clone native KLD logits for domain '{domain}': {logitsFile}");
                }
                else if (new FileInfo(logitsFile).Length <= 0)
                {
                    issues.Add($"Clone native KLD logits file is empty for domain '{domain}': {logitsFile}");
                }
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
        return baseline.Names.Any(n => string.Equals(n, name, StringComparison.OrdinalIgnoreCase)) ||
               string.Equals(baseline.QuantizeBaseArgumentName, name, StringComparison.OrdinalIgnoreCase) ||
               string.Equals(baseline.PrimaryTensorWeightScheme.Names[0], name, StringComparison.OrdinalIgnoreCase) ||
               string.Equals(baseline.CanonicalKey, name, StringComparison.OrdinalIgnoreCase);
    }

    private sealed record CloneNativeBenchmarkEnvironmentStatus(
        bool IsValid,
        IReadOnlyList<string> MissingOrInvalidArtifacts);

    private static async Task WriteCloneBenchmarkSummaryAsync(string outputDirectory, IReadOnlyCollection<CloneArtifactBuildRecord> records)
    {
        var payload = records
            .OrderBy(x => x.Kld ?? double.MaxValue)
            .ThenBy(x => x.ActualSizeBytes)
            .Select(x => new
            {
                fileName = x.ManifestArtifact.FileName,
                displayName = x.ManifestArtifact.DisplayName,
                provider = x.ManifestArtifact.Provider,
                quantFamily = x.ManifestArtifact.QuantFamily,
                baseQuant = x.ManifestArtifact.BaseQuant,
                kld = x.Kld,
                ppl = x.Ppl,
                pplDeltaPercent = x.PplDeltaPercent,
                sizeBytes = x.ActualSizeBytes,
                sizeGiB = x.ActualSizeBytes / 1024d / 1024d / 1024d,
                sourceKld = x.ManifestArtifact.SourceKld,
                sourcePpl = x.ManifestArtifact.SourcePpl,
                sourceSizeBytes = x.ManifestArtifact.SourceSizeBytes
            })
            .ToList();

        string path = Path.Combine(outputDirectory, "magicquant.clone-benchmarks.json");
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

    private static async Task CleanOutputDirectoryAsync(string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);

        foreach (var file in Directory.EnumerateFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly))
            await HardDeleteHelper.DeleteFileIfExistsAsync(file);

        foreach (var directory in Directory.EnumerateDirectories(outputDirectory, "*", SearchOption.TopDirectoryOnly))
            Directory.Delete(directory, recursive: true);

        AnsiConsole.MarkupLine($"[grey]Cleaned clone export directory:[/] {Markup.Escape(outputDirectory)}");
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
        AnsiConsole.MarkupLine("  mq clone-repository-quants --model-dir \"<path>\" --architecture-family \"<family>\" --source-repo \"owner/repo\" [--output-dir \"<path>\"]");
        AnsiConsole.MarkupLine("  mq clone-repository-quants --model-dir \"<path>\" --architecture-family \"<family>\" --source-json \"<path-or-url>\" [--output-dir \"<path>\"]");
        AnsiConsole.MarkupLine("Options:");
        AnsiConsole.MarkupLine("  --source-repo       Hugging Face repo containing magicquant.clone-configs.json");
        AnsiConsole.MarkupLine("  --source-json       Local or http(s) path to magicquant.clone-configs.json");
        AnsiConsole.MarkupLine("  --use-imatrix       Use configured/provided imatrix for the cloned model");
    }
}