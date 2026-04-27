using System.Text.Json;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class RepositoryCloneManifestService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    private readonly HuggingFaceBaselineService _huggingFace;

    public RepositoryCloneManifestService(HuggingFaceBaselineService huggingFace)
    {
        _huggingFace = huggingFace;
    }

    public async Task<(MagicQuantCloneManifest Manifest, string LocalPath, string SourceDescription)> ResolveAsync(
        string? sourceRepo,
        string? sourceJson,
        string modelMagicQuantDirectory,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(sourceRepo) && string.IsNullOrWhiteSpace(sourceJson))
            throw new InvalidOperationException("Clone mode requires --source-repo <hf/repo> or --source-json <path-or-url>.");

        string cloneDir = Path.Combine(modelMagicQuantDirectory, "CloneSource");
        Directory.CreateDirectory(cloneDir);

        string localPath;
        string sourceDescription;

        if (!string.IsNullOrWhiteSpace(sourceRepo))
        {
            localPath = Path.Combine(cloneDir, CloneConfigManifestGenerationService.FileName);
            sourceDescription = sourceRepo.Trim();
            await _huggingFace.DownloadRepositoryFileAsync(
                repoId: sourceRepo.Trim(),
                fileName: CloneConfigManifestGenerationService.FileName,
                destinationPath: localPath,
                forceRedownload: true,
                ct: ct);
        }
        else
        {
            string raw = sourceJson!.Trim();

            if (raw.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                raw.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                localPath = Path.Combine(cloneDir, CloneConfigManifestGenerationService.FileName);
                sourceDescription = raw;

                using var http = new HttpClient();
                var json = await http.GetStringAsync(raw, ct);
                await File.WriteAllTextAsync(localPath, json, ct);
            }
            else
            {
                localPath = Path.GetFullPath(raw);
                sourceDescription = localPath;

                if (!File.Exists(localPath))
                    throw new FileNotFoundException($"Clone JSON file does not exist: {localPath}");

                string copied = Path.Combine(cloneDir, CloneConfigManifestGenerationService.FileName);
                File.Copy(localPath, copied, overwrite: true);
                localPath = copied;
            }
        }

        var manifest = JsonSerializer.Deserialize<MagicQuantCloneManifest>(
            await File.ReadAllTextAsync(localPath, ct),
            JsonOptions);

        if (manifest == null)
            throw new InvalidOperationException($"Clone manifest could not be parsed: {localPath}");

        ValidateManifest(manifest, localPath);

        AnsiConsole.MarkupLine($"[green]Clone manifest loaded:[/] {Markup.Escape(localPath)} artifacts={manifest.Artifacts.Count:N0}");
        return (manifest, localPath, sourceDescription);
    }

    private static void ValidateManifest(MagicQuantCloneManifest manifest, string localPath)
    {
        if (manifest.SchemaVersion <= 0)
            throw new InvalidOperationException($"Clone manifest has invalid schemaVersion in {localPath}.");

        if (manifest.Artifacts.Count == 0)
            throw new InvalidOperationException($"Clone manifest contains zero artifacts: {localPath}");

        var duplicateFiles = manifest.Artifacts
            .Where(x => !string.IsNullOrWhiteSpace(x.FileName))
            .GroupBy(x => x.FileName, StringComparer.OrdinalIgnoreCase)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicateFiles.Count > 0)
            throw new InvalidOperationException($"Clone manifest contains duplicate file names: {string.Join(", ", duplicateFiles)}");

        foreach (var artifact in manifest.Artifacts)
        {
            if (string.IsNullOrWhiteSpace(artifact.FileName))
                throw new InvalidOperationException("Clone manifest artifact is missing fileName.");

            if (!artifact.FileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Clone manifest artifact fileName must end with .gguf: {artifact.FileName}");

            if (artifact.TensorTypes.Count == 0)
                throw new InvalidOperationException($"Clone manifest artifact '{artifact.FileName}' has no tensorTypes map.");
        }
    }
}
