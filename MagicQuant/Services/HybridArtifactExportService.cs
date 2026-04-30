using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services.Progress;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class HybridArtifactExportService
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

    private readonly QuantizationService _quantizationService;
    private readonly EffectiveCandidateStateResolverService _effectiveResolver;
    private readonly FinalArtifactNamingService _namingService;
    private readonly ModelSidecarArtifactService _sidecarService;

    public HybridArtifactExportService(
        QuantizationService quantizationService,
        EffectiveCandidateStateResolverService effectiveResolver,
        ModelSidecarArtifactService sidecarService)
    {
        _quantizationService = quantizationService;
        _effectiveResolver = effectiveResolver;
        _namingService = new FinalArtifactNamingService();
        _sidecarService = sidecarService;
    }

    public async Task<IReadOnlyList<ExportedArtifactRecord>> ExportAsync(
        IReadOnlyCollection<FinalSelectionRow> selectedRows,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.OutputDirectory))
            throw new InvalidOperationException("Cache.OutputDirectory is not set.");

        Directory.CreateDirectory(Cache.OutputDirectory);
        await CleanOutputDirectoryAsync(Cache.OutputDirectory!, Config.ReuseExistingFinalArtifacts, ct);

        var output = new List<ExportedArtifactRecord>();
        var reservedFileNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);

        var enabledRows = selectedRows
            .Where(x => x.Enabled)
            .OrderBy(x => x.Snapshot.Kld)
            .ThenBy(x => x.Snapshot.SizeBytes)
            .ToList();

        var localBuilds = new List<(ExportedArtifactRecord Record, HybridQuant Quant, string FullPath, ulong ExpectedBytes)>();

        foreach (var row in enabledRows)
        {
            var snap = row.Snapshot;
            bool isHybrid = snap.IsHybrid;
            bool exportLocally = isHybrid || !snap.IsExternalPureBaseline || Config.ExportExternalLearnedBaselines;
            var name = ResolvePlannedOrBuildName(row, snap, namingContext, reservedFileNames);
            string provider = !string.IsNullOrWhiteSpace(row.PlannedProviderName)
                ? row.PlannedProviderName
                : ResolveReadmeProviderName(snap, isHybrid, name);

            if (!exportLocally)
            {
                AnsiConsole.MarkupLine($"[grey]Skipping local export for external learned baseline by default:[/] {Markup.Escape(name.DisplayName)} [grey](enable with --export-external-learned-baselines or output.export_external_learned_baselines: true)[/]");

                output.Add(new ExportedArtifactRecord
                {
                    Snapshot = snap,
                    DisplayName = name.DisplayName,
                    ProviderName = provider,
                    BaselineFamily = name.QuantFamilyOrBaseline,
                    IsExternalReference = true,
                    DownloadTarget = snap.ExternalRepositoryUrl ?? string.Empty,
                    ExpectedSizeBytes = snap.SizeBytes,
                    EffectiveState = await _effectiveResolver.ResolveAsync(snap.Config, ct)
                });

                continue;
            }

            if (snap.IsExternalPureBaseline && !snap.IsHybrid)
                AnsiConsole.MarkupLine($"[yellow]Local export enabled for external learned baseline:[/] {Markup.Escape(name.DisplayName)}");

            string fullPath = Path.Combine(Cache.OutputDirectory!, name.FileName);
            var record = new ExportedArtifactRecord
            {
                Snapshot = snap,
                DisplayName = name.DisplayName,
                ProviderName = provider,
                BaselineFamily = name.QuantFamilyOrBaseline,
                IsExternalReference = false,
                FileName = name.FileName,
                FullPath = fullPath,
                DownloadTarget = $"./../../resolve/main/{name.FileName}?download=true",
                ExpectedSizeBytes = snap.SizeBytes,
                EffectiveState = await _effectiveResolver.ResolveAsync(snap.Config, ct)
            };

            if (Config.ReuseExistingFinalArtifacts &&
                TryReuseExistingFinalArtifact(Cache.OutputDirectory!, row, name.FileName, out var existingFullPath, out var actualSizeBytes))
            {
                record.ActualSizeBytes = actualSizeBytes;
                AnsiConsole.MarkupLine($"[green]Reused existing final GGUF:[/] {Markup.Escape(existingFullPath)} [grey]({actualSizeBytes:N0} bytes matched benchmark truth)[/]");
                output.Add(record);
                continue;
            }

            output.Add(record);
            localBuilds.Add((record, snap.Quant, fullPath, snap.SizeBytes));
        }

        StageProgressTracker? exportProgress = localBuilds.Count > 0
            ? new StageProgressTracker(new StageProgressOptions
            {
                StageName = "Final artifact export",
                Total = localBuilds.Count,
                ShowEta = false,
                MinimumPrintInterval = TimeSpan.FromSeconds(5),
                UnitLabel = "local GGUF outputs built"
            })
            : null;

        // Kick off all exports together. QuantizationService owns the real concurrency gates,
        // so this trusts that service to self-regulate CPU/GPU/process pressure.
        var buildTasks = localBuilds.Select(async item =>
        {
            string fileName = Path.GetFileName(item.FullPath);
            try
            {
                await _quantizationService.BuildExportArtifactAsync(item.Quant, item.FullPath, forceRebuild: true, ct: ct);

                ulong actualBytes = File.Exists(item.FullPath) ? (ulong)new FileInfo(item.FullPath).Length : 0UL;
                item.Record.ActualSizeBytes = actualBytes;

                if (actualBytes != item.ExpectedBytes)
                {
                    AnsiConsole.MarkupLine($"[yellow]Export byte validation warning:[/] expected [cyan]{item.ExpectedBytes:N0}[/] but got [cyan]{actualBytes:N0}[/] for {Markup.Escape(fileName)}");
                }

                exportProgress?.ReportFinished(SampleProcessState.Completed, fileName);
            }
            catch
            {
                exportProgress?.ReportFinished(SampleProcessState.Failed, fileName);
                throw;
            }
        });

        await Task.WhenAll(buildTasks);

        try
        {
            await CopyModelAdjacentFilesAsync(Cache.OutputDirectory!, ct);
            await CopyImatrixArtifactsAsync(Cache.OutputDirectory!, ct);
            await _sidecarService.CopyMmprojArtifactsAsync(Cache.OutputDirectory!, ct);
        }
        finally
        {
            await CleanExportSidecarsAsync(Cache.OutputDirectory!, ct);
        }

        return output;
    }

    private FinalArtifactName ResolvePlannedOrBuildName(
        FinalSelectionRow row,
        BenchmarkSnapshotRecord snapshot,
        FinalArtifactNamingContext namingContext,
        ISet<string> reservedFileNames)
    {
        if (!string.IsNullOrWhiteSpace(row.PlannedFileName) &&
            !string.IsNullOrWhiteSpace(row.PlannedDisplayName))
        {
            reservedFileNames.Add(row.PlannedFileName);
            return new FinalArtifactName
            {
                FileName = row.PlannedFileName,
                DisplayName = row.PlannedDisplayName,
                ShortDisplayName = _namingService.ToShortDisplayName(row.PlannedDisplayName),
                ProviderToken = row.PlannedProviderName,
                QuantFamilyOrBaseline = row.PlannedQuantFamily
            };
        }

        return _namingService.BuildName(snapshot, namingContext, reservedFileNames);
    }

    private static string ResolveReadmeProviderName(BenchmarkSnapshotRecord snapshot, bool isHybrid, FinalArtifactName name)
    {
        if (isHybrid)
            return "MagicQuant";

        if (string.Equals(name.ProviderToken, "MQ", StringComparison.OrdinalIgnoreCase))
            return "MagicQuant";

        return HybridBenchmarkRepository.ResolveProviderName(snapshot.Quant, exportNaming: false);
    }

    private static bool TryReuseExistingFinalArtifact(
        string outputDirectory,
        FinalSelectionRow row,
        string plannedFileName,
        out string fullPath,
        out ulong actualSizeBytes)
    {
        actualSizeBytes = 0UL;
        fullPath = Path.Combine(outputDirectory, plannedFileName);

        if (string.IsNullOrWhiteSpace(plannedFileName))
            return false;

        string expectedFileName = string.IsNullOrWhiteSpace(row.PlannedFileName)
            ? plannedFileName
            : row.PlannedFileName.Trim();

        if (!string.Equals(Path.GetFileName(fullPath), expectedFileName, StringComparison.OrdinalIgnoreCase))
            return false;

        if (!File.Exists(fullPath))
            return false;

        var info = new FileInfo(fullPath);
        if (info.Length <= 0)
            return false;

        actualSizeBytes = (ulong)info.Length;
        return actualSizeBytes == row.Snapshot.SizeBytes;
    }

    private static async Task CleanOutputDirectoryAsync(string outputDirectory, bool preserveReusableGgufs, CancellationToken ct)
    {
        if (!Directory.Exists(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
            return;
        }

        foreach (var file in Directory.EnumerateFiles(outputDirectory, "*", SearchOption.TopDirectoryOnly))
        {
            ct.ThrowIfCancellationRequested();

            if (preserveReusableGgufs &&
                string.Equals(Path.GetExtension(file), ".gguf", StringComparison.OrdinalIgnoreCase) &&
                new FileInfo(file).Length > 0)
            {
                continue;
            }

            await HardDeleteHelper.DeleteFileIfExistsAsync(file);
        }

        foreach (var directory in Directory.EnumerateDirectories(outputDirectory, "*", SearchOption.TopDirectoryOnly))
        {
            ct.ThrowIfCancellationRequested();
            await HardDeleteHelper.DeleteDirectoryIfExistsAsync(directory, ct);
        }

        AnsiConsole.MarkupLine(preserveReusableGgufs
            ? $"[grey]Cleaned final export directory metadata/non-GGUF files; preserved existing non-empty GGUFs for reuse validation:[/] {Markup.Escape(outputDirectory)}"
            : $"[grey]Cleaned final export directory:[/] {Markup.Escape(outputDirectory)}");
    }

    private static async Task CleanExportSidecarsAsync(string outputDirectory, CancellationToken ct)
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
            {
                ct.ThrowIfCancellationRequested();
                await HardDeleteHelper.DeleteFileIfExistsAsync(file);
            }
        }
    }

    private static async Task CopyModelAdjacentFilesAsync(string outputDirectory, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            return;

        foreach (var fileName in ModelAdjacentFiles)
        {
            string source = Path.Combine(Cache.ModelDirectory!, fileName);
            string target = Path.Combine(outputDirectory, fileName);

            if (!File.Exists(source))
            {
                AnsiConsole.MarkupLine($"[grey]Optional model-adjacent file missing:[/] {Markup.Escape(fileName)}");
                continue;
            }

            File.Copy(source, target, overwrite: true);
            await Task.Yield();
            AnsiConsole.MarkupLine($"[green]Copied model-adjacent file:[/] {Markup.Escape(fileName)}");
        }
    }

    private static Task CopyImatrixArtifactsAsync(string outputDirectory, CancellationToken ct)
    {
        if (!Cache.IsImatrixAvailable || string.IsNullOrWhiteSpace(Cache.ActiveImatrixPath))
            return Task.CompletedTask;

        string source = Cache.ActiveImatrixPath!;
        string target = Path.Combine(outputDirectory, "imatrix.dat");
        File.Copy(source, target, overwrite: true);
        AnsiConsole.MarkupLine($"[green]Copied imatrix artifact:[/] {Markup.Escape(target)}");
        return Task.CompletedTask;
    }

}