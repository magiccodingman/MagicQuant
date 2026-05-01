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
            string repo = sourceRepo.Trim();
            sourceDescription = repo;
            localPath = await DownloadRequiredCloneManifestFromRepoAsync(repo, cloneDir, ct);
        }
        else
        {
            string raw = sourceJson!.Trim();

            if (raw.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                raw.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                localPath = Path.Combine(MagicQuantManifestPathService.EnsureManifestDirectory(cloneDir), MagicQuantManifestPathService.CloneConfigsFileName);
                sourceDescription = raw;

                using var http = new HttpClient();
                var json = await http.GetStringAsync(raw, ct);
                await File.WriteAllTextAsync(localPath, json, ct);
            }
            else
            {
                string resolved = Path.GetFullPath(raw);
                sourceDescription = resolved;

                if (!File.Exists(resolved))
                    throw new FileNotFoundException($"Clone JSON file does not exist: {resolved}");

                string copied = Path.Combine(MagicQuantManifestPathService.EnsureManifestDirectory(cloneDir), MagicQuantManifestPathService.CloneConfigsFileName);
                File.Copy(resolved, copied, overwrite: true);
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

    private async Task<string> DownloadRequiredCloneManifestFromRepoAsync(string repoId, string cloneDir, CancellationToken ct)
    {
        string manifestDir = MagicQuantManifestPathService.EnsureManifestDirectory(cloneDir);
        string localPath = Path.Combine(manifestDir, MagicQuantManifestPathService.CloneConfigsFileName);

        var candidates = new[]
        {
            MagicQuantManifestPathService.RelativeManifestPath(MagicQuantManifestPathService.CloneConfigsFileName),
            MagicQuantManifestPathService.CloneConfigsFileName
        };

        var errors = new List<string>();
        foreach (var candidate in candidates)
        {
            try
            {
                await _huggingFace.DownloadRepositoryFileAsync(
                    repoId: repoId,
                    fileName: candidate,
                    destinationPath: localPath,
                    forceRedownload: true,
                    ct: ct);

                AnsiConsole.MarkupLine($"[green]Downloaded clone manifest:[/] {Markup.Escape(repoId)}/{Markup.Escape(candidate)}");
                return localPath;
            }
            catch (Exception ex)
            {
                errors.Add($"{candidate}: {ex.Message}");
            }
        }

        throw new InvalidOperationException(
            $"Could not download clone manifest from Hugging Face repo '{repoId}'. Tried new manifest folder path and legacy root path." +
            Environment.NewLine + string.Join(Environment.NewLine, errors.Select(x => "- " + x)));
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
