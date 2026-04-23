using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
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

    public HybridArtifactExportService(
        QuantizationService quantizationService,
        EffectiveCandidateStateResolverService effectiveResolver)
    {
        _quantizationService = quantizationService;
        _effectiveResolver = effectiveResolver;
    }

    public async Task<IReadOnlyList<ExportedArtifactRecord>> ExportAsync(
        IReadOnlyCollection<FinalSelectionRow> selectedRows,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.OutputDirectory))
            throw new InvalidOperationException("Cache.OutputDirectory is not set.");

        Directory.CreateDirectory(Cache.OutputDirectory);

        var output = new List<ExportedArtifactRecord>();
        var hybridOrdinalByFamily = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var row in selectedRows.Where(x => x.Enabled).OrderBy(x => x.Snapshot.Kld).ThenBy(x => x.Snapshot.SizeBytes))
        {
            var snap = row.Snapshot;
            bool isHybrid = snap.IsHybrid;
            bool exportLocally = isHybrid || !snap.IsExternalPureBaseline || Config.ExportExternalLearnedBaselines;
            string provider = HybridBenchmarkRepository.ResolveProviderName(snap.Quant, exportNaming: exportLocally && isHybrid);

            if (!exportLocally)
            {
                AnsiConsole.MarkupLine($"[grey]Skipping local export for external learned baseline by default:[/] {Markup.Escape(snap.DisplayName)} [grey](enable with --export-external-learned-baselines or output.export_external_learned_baselines: true)[/]");

                output.Add(new ExportedArtifactRecord
                {
                    Snapshot = snap,
                    DisplayName = snap.DisplayName,
                    ProviderName = provider,
                    BaselineFamily = snap.BaselineFamily,
                    IsExternalReference = true,
                    DownloadTarget = snap.ExternalRepositoryUrl ?? string.Empty,
                    ExpectedSizeBytes = snap.SizeBytes,
                    EffectiveState = await _effectiveResolver.ResolveAsync(snap.Config, ct)
                });

                continue;
            }

            if (snap.IsExternalPureBaseline && !snap.IsHybrid)
                AnsiConsole.MarkupLine($"[yellow]Local export enabled for external learned baseline:[/] {Markup.Escape(snap.DisplayName)}");

            string fileName = BuildFileName(snap, provider, hybridOrdinalByFamily);
            string fullPath = Path.Combine(Cache.OutputDirectory!, fileName);
            ulong expectedBytes = snap.SizeBytes;

            bool shouldBuild = true;
            if (File.Exists(fullPath))
            {
                ulong actual = (ulong)new FileInfo(fullPath).Length;
                if (actual == expectedBytes)
                {
                    shouldBuild = false;
                    AnsiConsole.MarkupLine($"[grey]Reusing existing exported artifact:[/] {Markup.Escape(fullPath)}");
                }
                else
                {
                    AnsiConsole.MarkupLine($"[yellow]Existing export byte size mismatch, rebuilding:[/] {Markup.Escape(fullPath)}");
                    File.Delete(fullPath);
                }
            }

            if (shouldBuild)
                await _quantizationService.BuildExportArtifactAsync(snap.Quant, fullPath, forceRebuild: false, ct: ct);

            ulong actualBytes = File.Exists(fullPath) ? (ulong)new FileInfo(fullPath).Length : 0UL;
            if (actualBytes != expectedBytes)
            {
                AnsiConsole.MarkupLine($"[yellow]Export byte validation warning:[/] expected [cyan]{expectedBytes:N0}[/] but got [cyan]{actualBytes:N0}[/] for {Markup.Escape(fileName)}");
            }

            output.Add(new ExportedArtifactRecord
            {
                Snapshot = snap,
                DisplayName = snap.DisplayName,
                ProviderName = provider,
                BaselineFamily = snap.BaselineFamily,
                IsExternalReference = false,
                FileName = fileName,
                FullPath = fullPath,
                DownloadTarget = $"./../../resolve/main/{fileName}?download=true",
                ExpectedSizeBytes = expectedBytes,
                ActualSizeBytes = actualBytes,
                EffectiveState = await _effectiveResolver.ResolveAsync(snap.Config, ct)
            });
        }

        await CopyModelAdjacentFilesAsync(Cache.OutputDirectory!, ct);
        await CopyImatrixArtifactsAsync(Cache.OutputDirectory!, ct);
        await CopyMmprojArtifactsAsync(Cache.OutputDirectory!, ct);

        return output;
    }

    private static string BuildFileName(
        BenchmarkSnapshotRecord snapshot,
        string provider,
        Dictionary<string, int> hybridOrdinalByFamily)
    {
        string prefix = Sanitize(Config.OutputNamePrefix);

        if (!snapshot.IsHybrid)
            return $"{prefix}-{Sanitize(provider)}-{Sanitize(snapshot.BaselineFamily)}.gguf";

        hybridOrdinalByFamily.TryGetValue(snapshot.BaselineFamily, out var current);
        current++;
        hybridOrdinalByFamily[snapshot.BaselineFamily] = current;

        string special = $"H{current}";
        return $"{prefix}-{Sanitize(provider)}-{special}-{Sanitize(snapshot.BaselineFamily)}.gguf";
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

    private static async Task CopyImatrixArtifactsAsync(string outputDirectory, CancellationToken ct)
    {
        if (!Cache.IsImatrixAvailable || string.IsNullOrWhiteSpace(Cache.ActiveImatrixPath))
            return;

        string source = Cache.ActiveImatrixPath!;
        string target = Path.Combine(outputDirectory, "imatrix.dat");
        File.Copy(source, target, overwrite: true);
        AnsiConsole.MarkupLine($"[green]Copied imatrix artifact:[/] {Markup.Escape(target)}");

        string imatrixDir = Path.GetDirectoryName(source)!;
        foreach (var optional in new[] { "imatrix.success.json", "imatrix.metadata.json", "imatrix.build.log" })
        {
            string optionalSource = Path.Combine(imatrixDir, optional);
            if (!File.Exists(optionalSource))
                continue;

            File.Copy(optionalSource, Path.Combine(outputDirectory, optional), overwrite: true);
            await Task.Yield();
            AnsiConsole.MarkupLine($"[green]Copied imatrix sidecar:[/] {Markup.Escape(optional)}");
        }
    }

    private static async Task CopyMmprojArtifactsAsync(string outputDirectory, CancellationToken ct)
    {
        var searchRoots = new List<string>();
        if (!string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            searchRoots.Add(Cache.ModelDirectory!);
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            searchRoots.Add(Cache.ModelMagicQuantDirectory!);

        foreach (var root in searchRoots.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var mmproj = Directory.EnumerateFiles(root, "*mmproj*.gguf", SearchOption.AllDirectories).FirstOrDefault();
            if (mmproj == null)
                continue;

            string target = Path.Combine(outputDirectory, Path.GetFileName(mmproj));
            File.Copy(mmproj, target, overwrite: true);
            AnsiConsole.MarkupLine($"[green]Copied mmproj artifact:[/] {Markup.Escape(target)}");
            return;
        }

        if (!LooksVisionCapableModel())
        {
            AnsiConsole.MarkupLine("[grey]No mmproj artifact was present, but no vision capability hints were detected. Continuing.[/]");
            return;
        }

        throw new InvalidOperationException(
            "This model appears to be vision-capable, but no mmproj GGUF could be found in the working/source artifacts.");
    }

    private static bool LooksVisionCapableModel()
    {
        if (string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            return false;

        string configPath = Path.Combine(Cache.ModelDirectory!, "config.json");
        if (!File.Exists(configPath))
            return false;

        string json = File.ReadAllText(configPath);
        return json.Contains("vision_config", StringComparison.OrdinalIgnoreCase) ||
               json.Contains("vision_tower", StringComparison.OrdinalIgnoreCase) ||
               json.Contains("mm_vision_tower", StringComparison.OrdinalIgnoreCase) ||
               json.Contains("projector", StringComparison.OrdinalIgnoreCase);
    }

    private static string Sanitize(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "model";

        var cleaned = value.Trim();
        foreach (char c in Path.GetInvalidFileNameChars())
            cleaned = cleaned.Replace(c, '-');

        return cleaned.Replace(" ", "-");
    }
}
