using Microsoft.EntityFrameworkCore;
using MQ.DB.Data;
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

        var enabledRepos = Config.Current.Baselines.CustomRepositories.Where(x => x.Enabled).ToList();
        var resolved = new List<ResolvedCustomBaselineSpec>();
        BaselineQuants.ResetDynamicCustomBaselines();

        AnsiConsole.Write(new Rule("[yellow]Custom Baseline Precheck[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[grey]Enabled custom repositories:[/] [cyan]{enabledRepos.Count:N0}[/]");

        if (enabledRepos.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No enabled custom repositories were configured for this run.[/]");
            Config.SetResolvedCustomBaselines(Array.Empty<ResolvedCustomBaselineSpec>());
            BaselineQuants.ValidateIntegrityOrThrow();
            return resolved;
        }

        var existingDynamicIdsByCanonicalKey = LoadExistingDynamicBaselineIds();
        var reservedIds = BaselineQuants.GetAllRecognizedBaselines()
            .Select(x => x.UniqueId)
            .ToHashSet();

        foreach (var persistedId in existingDynamicIdsByCanonicalKey.Values)
            reservedIds.Add(persistedId);

        byte nextId = BaselineQuants.GetFirstAvailableDynamicBaselineId();

        foreach (var repo in enabledRepos)
        {
            if (string.IsNullOrWhiteSpace(repo.RepoId))
                throw new InvalidOperationException("Custom baseline repository entry is missing repo_id.");

            if (repo.Includes.Count == 0)
                throw new InvalidOperationException($"Custom baseline repository '{repo.RepoId}' is enabled but has zero include entries.");

            AnsiConsole.MarkupLine($"[cyan]Repo:[/] {Markup.Escape(repo.RepoId)} [grey](includes={repo.Includes.Count})[/]");

            var repoFiles = await ListRepoFilesAsync(repo.RepoId, ct);
            if (repoFiles.Count == 0)
                throw new InvalidOperationException($"No files were returned from Hugging Face repo '{repo.RepoId}'.");

            var ggufRepoFiles = repoFiles.Where(x => x.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).ToList();
            AnsiConsole.MarkupLine($"  [grey]GGUF files discovered:[/] [cyan]{ggufRepoFiles.Count:N0}[/]");

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

                byte dynamicBaselineId = ResolveDynamicBaselineId(
                    canonicalKey,
                    existingDynamicIdsByCanonicalKey,
                    reservedIds,
                    ref nextId);

                var dynamicBaseline = BaselineQuants.CreateDynamicCustomBaseline(
                    uniqueId: dynamicBaselineId,
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
                    bitRange: standardFamily.BitRange,
                    explicitCandidateSortOrder: standardFamily.ExplicitCandidateSortOrder);

                BaselineQuants.RegisterDynamicCustomBaseline(dynamicBaseline);

                var spec = new ResolvedCustomBaselineSpec
                {
                    DynamicBaselineId = dynamicBaseline.UniqueId,
                    CanonicalKey = dynamicBaseline.CanonicalKey,
                    DisplayName = dynamicBaseline.Names[0],
                    RepoId = repo.RepoId,
                    SourceOwner = dynamicBaseline.SourceOwner ?? string.Empty,
                    SourceFileName = dynamicBaseline.SourceFileName ?? string.Empty,
                    ShortSourceName = dynamicBaseline.ShortSourceName ?? shortSourceName,
                    BaselineFamily = standardFamily.Names[0],
                    QuantizeBaseName = dynamicBaseline.QuantizeBaseArgumentName,
                    RequiresImatrix = dynamicBaseline.RequiresImatrix,
                    AllowAsLearningBaseline = dynamicBaseline.IsLearningBaseline,
                    AllowAsCombinationCarrier = dynamicBaseline.IsCombinationCarrierCandidate,
                    AllowAsExplicitGroupCandidate = dynamicBaseline.IsExplicitGroupCombinationCandidate,
                    BannedGroupIds = dynamicBaseline.BannedGroupIds
                };

                resolved.Add(spec);
                AnsiConsole.MarkupLine(
                    $"  [green]Resolved:[/] id=[cyan]{dynamicBaseline.UniqueId}[/] family=[yellow]{Markup.Escape(standardFamily.Names[0])}[/] file=[blue]{Markup.Escape(resolvedFileName)}[/] learning={allowAsLearning} carrier={allowAsCarrier} explicit={allowAsExplicit}");
            }
        }

        if (enabledRepos.Count > 0 && resolved.Count == 0)
            throw new InvalidOperationException("Custom baseline repositories were enabled, but zero custom baselines resolved into the runtime registry. Check YAML property names and include entries.");

        Config.SetResolvedCustomBaselines(resolved);
        BaselineQuants.ValidateIntegrityOrThrow();

        AnsiConsole.MarkupLine($"[green]Custom baseline precheck complete:[/] [cyan]{resolved.Count:N0}[/] resolved custom baseline(s).");
        return resolved;
    }

    private static Dictionary<string, byte> LoadExistingDynamicBaselineIds()
    {
        try
        {
            using var db = new MagicQuantContext();

            return db.BaselineQuantDefinitions
                .AsNoTracking()
                .Where(x => x.IsCustomBaseline && !string.IsNullOrWhiteSpace(x.CanonicalKey))
                .OrderBy(x => x.BaselineQuantId)
                .ToDictionary(x => x.CanonicalKey, x => x.BaselineQuantId, StringComparer.Ordinal);
        }
        catch
        {
            return new Dictionary<string, byte>(StringComparer.Ordinal);
        }
    }

    private static byte ResolveDynamicBaselineId(
        string canonicalKey,
        IReadOnlyDictionary<string, byte> existingDynamicIdsByCanonicalKey,
        HashSet<byte> reservedIds,
        ref byte nextId)
    {
        if (!string.IsNullOrWhiteSpace(canonicalKey) &&
            existingDynamicIdsByCanonicalKey.TryGetValue(canonicalKey, out var existingId))
        {
            reservedIds.Add(existingId);
            return existingId;
        }

        while (reservedIds.Contains(nextId))
        {
            if (nextId >= 199)
                throw new InvalidOperationException("No free dynamic baseline ids remain in the configured range.");

            nextId++;
        }

        var allocated = nextId;
        reservedIds.Add(allocated);

        if (nextId < 199)
            nextId++;

        return allocated;
    }

    public async Task<string> DownloadBaselineAsync(BaselineQuants baseline, string destinationPath, bool forceRedownload = false, CancellationToken ct = default)
    {
        if (!baseline.IsExternalRepositoryBaseline)
            throw new InvalidOperationException($"Baseline '{baseline.Names[0]}' is not an external repository baseline.");

        await EnsureHubSupportAsync();

        var spec = Config.GetResolvedCustomBaseline(baseline.CanonicalKey)
            ?? throw new InvalidOperationException($"No resolved custom baseline spec exists for canonical key '{baseline.CanonicalKey}'.");

        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);

        if (File.Exists(destinationPath) && new FileInfo(destinationPath).Length == 0)
            File.Delete(destinationPath);

        if (forceRedownload && File.Exists(destinationPath))
            File.Delete(destinationPath);

        if (File.Exists(destinationPath) && new FileInfo(destinationPath).Length > 0)
        {
            AnsiConsole.MarkupLine($"[grey]Reusing cached external baseline:[/] {Markup.Escape(destinationPath)}");
            return destinationPath;
        }

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
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    shutil.copy2(downloaded, target_path)

                result = {
                    'ok': True,
                    'downloaded_path': target_path,
                    'size_bytes': os.path.getsize(target_path) if os.path.exists(target_path) else 0,
                }
            except Exception as ex:
                result = {
                    'ok': False,
                    'error': str(ex),
                }

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            """;

            await File.WriteAllTextAsync(scriptPath, py, ct);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            var json = JsonDocument.Parse(await File.ReadAllTextAsync(resultPath, ct)).RootElement;
            if (!json.GetProperty("ok").GetBoolean())
                throw new InvalidOperationException($"External baseline download failed: {json.GetProperty("error").GetString()}");

            if (!File.Exists(destinationPath) || new FileInfo(destinationPath).Length == 0)
                throw new InvalidOperationException($"External baseline download completed but produced no file: {destinationPath}");

            AnsiConsole.MarkupLine($"[green]Downloaded external baseline:[/] {Markup.Escape(destinationPath)}");
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