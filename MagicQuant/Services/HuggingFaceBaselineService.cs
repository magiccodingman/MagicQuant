using System.Text.Json;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class HuggingFaceBaselineService
{
    private readonly PythonManager _python;

    public HuggingFaceBaselineService(PythonManager python)
    {
        _python = python ?? throw new ArgumentNullException(nameof(python));
    }

    public async Task<IReadOnlyList<ResolvedCustomBaselineSpec>> PrecheckAndRegisterConfiguredBaselinesAsync(CancellationToken ct = default)
    {
        await EnsureHubSupportAsync();

        var resolved = new List<ResolvedCustomBaselineSpec>();
        BaselineQuants.ResetDynamicCustomBaselines();

        byte nextId = BaselineQuants.GetFirstAvailableDynamicBaselineId();

        foreach (var repo in Config.Current.Baselines.CustomRepositories.Where(x => x.Enabled))
        {
            if (string.IsNullOrWhiteSpace(repo.RepoId))
                throw new InvalidOperationException("Custom baseline repository entry is missing repo_id.");

            var repoFiles = await ListRepoFilesAsync(repo.RepoId, ct);
            if (repoFiles.Count == 0)
                throw new InvalidOperationException($"No files were returned from Hugging Face repo '{repo.RepoId}'.");

            string shortSourceName = string.IsNullOrWhiteSpace(repo.ShortSourceName)
                ? DeriveShortSourceName(repo.RepoId)
                : repo.ShortSourceName!.Trim();

            foreach (var include in repo.Includes)
            {
                if (string.IsNullOrWhiteSpace(include.BaselineFamily))
                    throw new InvalidOperationException($"Repo '{repo.RepoId}' has an include entry missing baseline_family.");

                var standardFamily = BaselineQuants.ResolveBuiltInStandardBaseline(include.BaselineFamily)
                    ?? throw new InvalidOperationException(
                        $"Custom baseline include '{include.BaselineFamily}' in repo '{repo.RepoId}' could not be matched to a built-in baseline family.");

                string resolvedFileName = ResolveRepoFileName(repoFiles, include, standardFamily);
                string displayName = string.IsNullOrWhiteSpace(include.DisplayName)
                    ? $"{shortSourceName}-{standardFamily.Names[0]}"
                    : include.DisplayName!.Trim();

                string canonicalKey = BuildCanonicalKey(repo.RepoId, resolvedFileName, standardFamily.Names[0]);
                bool requiresImatrix = include.RequiresImatrix ?? standardFamily.RequiresImatrix;
                bool allowAsLearning = include.AllowAsLearningBaseline ?? repo.AllowAsLearningBaseline;
                bool allowAsCarrier = include.AllowAsCombinationCarrier ?? repo.AllowAsCombinationCarrier;
                bool allowAsExplicit = include.AllowAsExplicitGroupCandidate ?? repo.AllowAsExplicitGroupCandidate;
                string quantizeBaseName = string.IsNullOrWhiteSpace(include.QuantizeBaseName)
                    ? standardFamily.Names[0]
                    : include.QuantizeBaseName!.Trim();

                var bannedGroups = include.BannedGroupIds.Count > 0
                    ? include.BannedGroupIds.ToArray()
                    : standardFamily.BannedGroupIds.ToArray();

                var dynamicBaseline = BaselineQuants.CreateDynamicCustomBaseline(
                    uniqueId: nextId,
                    displayName: displayName,
                    quantizeBaseArgumentName: quantizeBaseName,
                    sourceRepository: repo.RepoId,
                    sourceFileName: resolvedFileName,
                    shortSourceName: shortSourceName,
                    sourceOwner: DeriveSourceOwner(repo.RepoId),
                    sourceKind: "huggingface_repo",
                    canonicalKey: canonicalKey,
                    primaryTensorWeightScheme: standardFamily.PrimaryTensorWeightScheme,
                    learnedMatchTensorWeightSchemes: standardFamily.LearnedMatchTensorWeightSchemes,
                    bannedGroupIds: bannedGroups,
                    requiresImatrix: requiresImatrix,
                    isLearningBaseline: allowAsLearning,
                    isCombinationCarrierCandidate: allowAsCarrier,
                    isExplicitGroupCombinationCandidate: allowAsExplicit,
                    explicitCandidateSortOrder: standardFamily.ExplicitCandidateSortOrder);

                BaselineQuants.RegisterDynamicCustomBaseline(dynamicBaseline);

                resolved.Add(new ResolvedCustomBaselineSpec
                {
                    DynamicBaselineId = nextId,
                    CanonicalKey = canonicalKey,
                    DisplayName = displayName,
                    RepoId = repo.RepoId,
                    SourceOwner = DeriveSourceOwner(repo.RepoId),
                    SourceFileName = resolvedFileName,
                    ShortSourceName = shortSourceName,
                    BaselineFamily = standardFamily.Names[0],
                    QuantizeBaseName = quantizeBaseName,
                    RequiresImatrix = requiresImatrix,
                    AllowAsLearningBaseline = allowAsLearning,
                    AllowAsCombinationCarrier = allowAsCarrier,
                    AllowAsExplicitGroupCandidate = allowAsExplicit,
                    BannedGroupIds = bannedGroups
                });

                checked { nextId++; }
            }
        }

        Config.SetResolvedCustomBaselines(resolved);
        BaselineQuants.ValidateIntegrityOrThrow();

        if (resolved.Count > 0)
        {
            AnsiConsole.MarkupLine($"[green]Resolved custom baselines:[/] {resolved.Count:N0}");
            foreach (var item in resolved)
            {
                AnsiConsole.MarkupLine(
                    $"  [grey]- {Markup.Escape(item.DisplayName)}[/] => [cyan]{Markup.Escape(item.RepoId)}[/] / [yellow]{Markup.Escape(item.SourceFileName)}[/]");
            }
        }

        return resolved;
    }

    public async Task<string> DownloadBaselineAsync(BaselineQuants baseline, string destinationPath, bool forceRedownload = false, CancellationToken ct = default)
    {
        if (!baseline.IsExternalRepositoryBaseline)
            throw new InvalidOperationException($"Baseline '{baseline.Names[0]}' is not an external repository baseline.");

        await EnsureHubSupportAsync();

        var spec = Config.GetResolvedCustomBaseline(baseline.CanonicalKey)
            ?? throw new InvalidOperationException($"No resolved custom baseline spec exists for canonical key '{baseline.CanonicalKey}'.");

        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);

        if (forceRedownload && File.Exists(destinationPath))
            File.Delete(destinationPath);

        string payloadPath = Path.Combine(Path.GetDirectoryName(destinationPath)!, $"hf_download_{Guid.NewGuid():N}.json");
        string scriptPath = Path.Combine(Path.GetDirectoryName(destinationPath)!, $"hf_download_{Guid.NewGuid():N}.py");
        string resultPath = Path.Combine(Path.GetDirectoryName(destinationPath)!, $"hf_download_result_{Guid.NewGuid():N}.json");

        try
        {
            await File.WriteAllTextAsync(payloadPath, JsonSerializer.Serialize(new
            {
                repo_id = spec.RepoId,
                file_name = spec.SourceFileName,
                target_path = destinationPath,
                force_redownload = forceRedownload,
                result_path = resultPath
            }), ct);

            const string py = """
            import json
            import os
            import shutil
            import sys
            from huggingface_hub import hf_hub_download

            payload_path = sys.argv[1]
            with open(payload_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            target_path = payload['target_path']
            result_path = payload['result_path']
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            try:
                downloaded = hf_hub_download(
                    repo_id=payload['repo_id'],
                    filename=payload['file_name'],
                    local_dir=os.path.dirname(target_path),
                    force_download=payload.get('force_redownload', False),
                )

                if os.path.abspath(downloaded) != os.path.abspath(target_path):
                    shutil.copy2(downloaded, target_path)

                size = os.path.getsize(target_path)
                result = {'success': True, 'path': target_path, 'size': size}
            except Exception as ex:
                result = {'success': False, 'error': str(ex)}

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            """;

            await File.WriteAllTextAsync(scriptPath, py, ct);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            using var doc = JsonDocument.Parse(await File.ReadAllTextAsync(resultPath, ct));
            if (!doc.RootElement.TryGetProperty("success", out var successProp) || !successProp.GetBoolean())
            {
                string error = doc.RootElement.TryGetProperty("error", out var errProp) ? errProp.GetString() ?? "unknown error" : "unknown error";
                throw new InvalidOperationException($"Failed downloading external baseline '{baseline.Names[0]}': {error}");
            }

            if (!File.Exists(destinationPath) || new FileInfo(destinationPath).Length == 0)
                throw new InvalidOperationException($"External baseline download reported success but no valid file exists at '{destinationPath}'.");

            return destinationPath;
        }
        finally
        {
            TryDelete(payloadPath);
            TryDelete(scriptPath);
            TryDelete(resultPath);
        }
    }

    private async Task EnsureHubSupportAsync()
    {
        string? version = await _python.GetInstalledVersionAsync("huggingface_hub");
        if (version != null)
            return;

        AnsiConsole.MarkupLine("[cyan]Installing huggingface_hub (includes Hub download/CLI support)...[/]");
        await _python.RunPipInstallAsync("--upgrade huggingface_hub");
    }

    private async Task<List<string>> ListRepoFilesAsync(string repoId, CancellationToken ct)
    {
        string tempDir = Cache.ExternalBaselineCacheDirectory ?? Cache.MagicQuantDirectory ?? AppContext.BaseDirectory;
        Directory.CreateDirectory(tempDir);

        string payloadPath = Path.Combine(tempDir, $"hf_repo_list_{Guid.NewGuid():N}.json");
        string resultPath = Path.Combine(tempDir, $"hf_repo_list_result_{Guid.NewGuid():N}.json");
        string scriptPath = Path.Combine(tempDir, $"hf_repo_list_{Guid.NewGuid():N}.py");

        try
        {
            await File.WriteAllTextAsync(payloadPath, JsonSerializer.Serialize(new { repo_id = repoId, result_path = resultPath }), ct);

            const string py = """
            import json
            import sys
            from huggingface_hub import HfApi

            payload_path = sys.argv[1]
            with open(payload_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            result_path = payload['result_path']
            try:
                files = HfApi().list_repo_files(repo_id=payload['repo_id'])
                result = {'success': True, 'files': files}
            except Exception as ex:
                result = {'success': False, 'error': str(ex), 'files': []}

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            """;

            await File.WriteAllTextAsync(scriptPath, py, ct);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            using var doc = JsonDocument.Parse(await File.ReadAllTextAsync(resultPath, ct));
            if (!doc.RootElement.TryGetProperty("success", out var successProp) || !successProp.GetBoolean())
            {
                string error = doc.RootElement.TryGetProperty("error", out var errProp) ? errProp.GetString() ?? "unknown error" : "unknown error";
                throw new InvalidOperationException($"Failed listing files for Hugging Face repo '{repoId}': {error}");
            }

            return doc.RootElement.GetProperty("files")
                .EnumerateArray()
                .Select(x => x.GetString())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Cast<string>()
                .ToList();
        }
        finally
        {
            TryDelete(payloadPath);
            TryDelete(scriptPath);
            TryDelete(resultPath);
        }
    }

    private static string ResolveRepoFileName(IReadOnlyList<string> repoFiles, CustomBaselineIncludeConfig include, BaselineQuants standardFamily)
    {
        if (!string.IsNullOrWhiteSpace(include.FileName))
        {
            string explicitName = include.FileName!.Trim();
            var match = repoFiles.FirstOrDefault(x => string.Equals(x, explicitName, StringComparison.OrdinalIgnoreCase));
            if (match == null)
                throw new InvalidOperationException($"Configured file '{explicitName}' was not found in the configured custom baseline repository.");

            if (!match.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Configured file '{explicitName}' is not a GGUF file.");

            return match;
        }

        string family = NormalizeSuffixToken(standardFamily.Names[0]);
        var matches = repoFiles
            .Where(x => x.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .Where(x => NormalizeSuffixToken(Path.GetFileNameWithoutExtension(x)).EndsWith(family, StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (matches.Count == 0)
        {
            throw new InvalidOperationException(
                $"Could not auto-match a GGUF file for baseline family '{standardFamily.Names[0]}'. " +
                "Specify file_name explicitly in the YAML.");
        }

        if (matches.Count > 1)
        {
            throw new InvalidOperationException(
                $"Auto-match for baseline family '{standardFamily.Names[0]}' returned multiple files: {string.Join(", ", matches)}. " +
                "Specify file_name explicitly in the YAML.");
        }

        return matches[0];
    }

    private static string BuildCanonicalKey(string repoId, string fileName, string family)
        => $"hf:{repoId.Trim().ToLowerInvariant()}::{fileName.Trim().ToLowerInvariant()}::{family.Trim().ToLowerInvariant()}";

    private static string DeriveShortSourceName(string repoId)
    {
        var owner = DeriveSourceOwner(repoId);
        if (string.IsNullOrWhiteSpace(owner))
            return "Custom";

        return char.ToUpperInvariant(owner[0]) + owner[1..];
    }

    private static string DeriveSourceOwner(string repoId)
    {
        if (string.IsNullOrWhiteSpace(repoId))
            return string.Empty;

        var parts = repoId.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length > 0 ? parts[0] : repoId;
    }

    private static string NormalizeSuffixToken(string value)
    {
        return value.Trim()
            .Replace("-", "_")
            .Replace(" ", string.Empty)
            .ToUpperInvariant();
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
        }
    }
}
