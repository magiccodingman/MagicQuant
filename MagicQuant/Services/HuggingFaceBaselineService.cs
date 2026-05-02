using Microsoft.EntityFrameworkCore;
using MQ.DB.Data;
using System.Text.Json;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
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

    int architectureFamilyId = Cache.CurrentArchitectureFamilyId
        ?? throw new InvalidOperationException("Custom baseline sync requires the architecture family to be resolved first.");

    var enabledRepos = Config.Current.Baselines.CustomRepositories.Where(x => x.Enabled).ToList();
    var resolved = new List<ResolvedCustomBaselineSpec>();
    BaselineQuants.ResetDynamicCustomBaselines();

    AnsiConsole.Write(new Rule("[yellow]Custom Baseline DB Sync[/]") { Justification = Justify.Left });
    AnsiConsole.MarkupLine($"[grey]Enabled custom repositories:[/] [cyan]{enabledRepos.Count:N0}[/]");

    await using var db = new MagicQuantContext();
    var existingDefinitions = await db.BaselineQuantDefinitions
        .Where(x => x.ArchitectureFamilyId == architectureFamilyId && x.IsCustomBaseline)
        .ToListAsync(ct);

    foreach (var definition in existingDefinitions)
    {
        definition.IsActiveInCurrentConfig = false;
        definition.LastUpdatedUtc = DateTime.UtcNow;
    }

    RegisterHistoricalDefinitions(existingDefinitions);

    var existingDynamicIdsByCanonicalKey = existingDefinitions
        .Where(x => !string.IsNullOrWhiteSpace(x.NormalizedCanonicalKey))
        .GroupBy(x => x.NormalizedCanonicalKey, StringComparer.Ordinal)
        .ToDictionary(g => g.Key, g => g.OrderBy(x => x.RuntimeBaselineId).First().RuntimeBaselineId, StringComparer.Ordinal);

    var reservedIds = BaselineQuants.GetAllRecognizedBaselines().Select(x => x.UniqueId).ToHashSet();
    foreach (var persistedId in existingDefinitions.Select(x => x.RuntimeBaselineId))
        reservedIds.Add(persistedId);

    byte nextId = BaselineQuants.GetFirstAvailableDynamicBaselineId();
    var now = DateTime.UtcNow;

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
            string normalizedRepo = BaselineDefinitionResolver.NormalizeRepoId(repo.RepoId);
            string normalizedFile = BaselineDefinitionResolver.NormalizeFileName(resolvedFileName);
            string canonicalKey = BuildCanonicalKey(Cache.CurrentArchitectureFamilyName, repo.RepoId, resolvedFileName);
            string normalizedCanonicalKey = BaselineDefinitionResolver.NormalizeCanonicalKey(canonicalKey);

            string displayName = string.IsNullOrWhiteSpace(include.DisplayName)
                ? $"{shortSourceName}-{standardFamily.Names[0]}"
                : include.DisplayName!.Trim();

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

            var definition = existingDefinitions.FirstOrDefault(x =>
                string.Equals(x.NormalizedSourceRepository, normalizedRepo, StringComparison.Ordinal) &&
                string.Equals(x.NormalizedSourceFileName, normalizedFile, StringComparison.Ordinal));

            if (definition != null && !string.Equals(definition.BaselineFamily, standardFamily.Names[0], StringComparison.Ordinal) && !include.ForceRelearn)
            {
                throw new InvalidOperationException(
                    $"Custom baseline semantic family changed for {repo.RepoId}/{resolvedFileName}: " +
                    $"DB has '{definition.BaselineFamily}', YAML now says '{standardFamily.Names[0]}'. " +
                    "This is destructive. Set this include's force_relearn: true so MagicQuant can plan and confirm targeted invalidation before resyncing the definition.");
            }

            byte dynamicBaselineId = definition?.RuntimeBaselineId ?? ResolveDynamicBaselineId(
                normalizedCanonicalKey,
                existingDynamicIdsByCanonicalKey,
                reservedIds,
                ref nextId);

            var nextDefinition = new BaselineQuantDefinition
            {
                ArchitectureFamilyId = architectureFamilyId,
                RuntimeBaselineId = dynamicBaselineId,
                CanonicalKey = canonicalKey,
                NormalizedCanonicalKey = normalizedCanonicalKey,
                BaselineName = displayName,
                DisplayName = displayName,
                QuantizeBaseArgumentName = quantizeBaseName,
                DefaultTensorSchemeId = standardFamily.PrimaryTensorWeightScheme.UniqueId,
                DefaultTensorSchemeName = standardFamily.PrimaryTensorWeightScheme.Names[0],
                SourceKind = repo.SourceKind,
                SourceOwner = DeriveSourceOwner(repo.RepoId),
                SourceRepository = repo.RepoId,
                NormalizedSourceRepository = normalizedRepo,
                SourceFileName = resolvedFileName,
                NormalizedSourceFileName = normalizedFile,
                ShortSourceName = shortSourceName,
                BaselineFamily = standardFamily.Names[0],
                IsCustomBaseline = true,
                IsLearningBaseline = allowAsLearning,
                IsCombinationCarrierCandidate = allowAsCarrier,
                IsExplicitGroupCombinationCandidate = allowAsExplicit,
                RequiresImatrix = requiresImatrix,
                BitRange = standardFamily.BitRange,
                ExplicitCandidateSortOrder = standardFamily.ExplicitCandidateSortOrder,
                IsActiveInCurrentConfig = true,
                FirstSeenUtc = definition?.FirstSeenUtc ?? now,
                LastSeenUtc = now,
                LastUpdatedUtc = now
            };

            if (definition == null)
            {
                definition = nextDefinition;
                db.BaselineQuantDefinitions.Add(definition);
                existingDefinitions.Add(definition);
            }
            else
            {
                MagicQuantContext.ApplyBaselineDefinitionUpdate(definition, nextDefinition, preserveFirstSeen: true);
            }

            var dynamicBaseline = BaselineQuants.CreateDynamicCustomBaseline(
                uniqueId: dynamicBaselineId,
                displayName: displayName,
                quantizeBaseArgumentName: quantizeBaseName,
                sourceRepository: repo.RepoId,
                sourceFileName: resolvedFileName,
                shortSourceName: shortSourceName,
                sourceOwner: DeriveSourceOwner(repo.RepoId),
                sourceKind: repo.SourceKind,
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
                BaselineQuantDefinitionId = definition.Id == 0 ? null : definition.Id,
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
                ForceRelearn = include.ForceRelearn,
                IsActiveInCurrentConfig = true,
                BannedGroupIds = dynamicBaseline.BannedGroupIds
            };

            resolved.Add(spec);
            AnsiConsole.MarkupLine(
                $"  [green]Resolved:[/] id=[cyan]{dynamicBaseline.UniqueId}[/] family=[yellow]{Markup.Escape(standardFamily.Names[0])}[/] file=[blue]{Markup.Escape(resolvedFileName)}[/] learning={allowAsLearning} carrier={allowAsCarrier} explicit={allowAsExplicit} relearn={include.ForceRelearn}");
        }
    }

    await db.SaveChangesAsync(ct);

    foreach (var spec in resolved.Where(x => x.BaselineQuantDefinitionId == null))
    {
        var definition = await db.BaselineQuantDefinitions.AsNoTracking().FirstAsync(x =>
            x.ArchitectureFamilyId == architectureFamilyId &&
            x.RuntimeBaselineId == spec.DynamicBaselineId, ct);
        spec.BaselineQuantDefinitionId = definition.Id;
    }

    RegisterHistoricalDefinitions(existingDefinitions.Where(x => !x.IsActiveInCurrentConfig));

    Config.SetResolvedCustomBaselines(resolved);
    BaselineQuants.ValidateIntegrityOrThrow();

    AnsiConsole.MarkupLine($"[green]Custom baseline sync complete:[/] [cyan]{resolved.Count:N0}[/] active custom baseline(s); [cyan]{existingDefinitions.Count(x => !x.IsActiveInCurrentConfig):N0}[/] inactive historical definition(s) retained.");
    return resolved;
}

private static void RegisterHistoricalDefinitions(IEnumerable<BaselineQuantDefinition> definitions)
{
    foreach (var definition in definitions.Where(x => x.IsCustomBaseline))
    {
        try
        {
            var runtime = BaselineDefinitionResolver.ToRuntimeBaseline(definition, forceInactiveRegistration: true);
            BaselineQuants.RegisterDynamicCustomBaseline(runtime);
        }
        catch
        {
            // A bad historical row should not prevent active YAML from being resolved.
            // It simply will not be available for runtime TensorConfig hydration until fixed.
        }
    }
}

private static byte ResolveDynamicBaselineId(
    string normalizedCanonicalKey,
    IReadOnlyDictionary<string, byte> existingDynamicIdsByCanonicalKey,
    HashSet<byte> reservedIds,
    ref byte nextId)
{
    if (!string.IsNullOrWhiteSpace(normalizedCanonicalKey) &&
        existingDynamicIdsByCanonicalKey.TryGetValue(normalizedCanonicalKey, out var existingId))
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


    public async Task<string> DownloadRepositoryFileAsync(
        string repoId,
        string fileName,
        string destinationPath,
        bool forceRedownload = true,
        CancellationToken ct = default)
    {
        await EnsureHubSupportAsync();

        if (string.IsNullOrWhiteSpace(repoId))
            throw new ArgumentException("Hugging Face repo id is required.", nameof(repoId));
        if (string.IsNullOrWhiteSpace(fileName))
            throw new ArgumentException("Hugging Face file name is required.", nameof(fileName));
        if (string.IsNullOrWhiteSpace(destinationPath))
            throw new ArgumentException("Destination path is required.", nameof(destinationPath));

        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);

        string tempDir = Cache.ExternalBaselineCacheDirectory ?? Cache.MagicQuantDirectory ?? AppContext.BaseDirectory;
        Directory.CreateDirectory(tempDir);

        string payloadPath = Path.Combine(tempDir, $"hf_download_file_{Guid.NewGuid():N}.json");
        string resultPath = Path.Combine(tempDir, $"hf_download_file_result_{Guid.NewGuid():N}.json");
        string scriptPath = Path.Combine(tempDir, $"hf_download_file_{Guid.NewGuid():N}.py");

        try
        {
            await File.WriteAllTextAsync(payloadPath, JsonSerializer.Serialize(new
            {
                repo_id = repoId,
                file_name = fileName,
                destination_path = destinationPath,
                force_redownload = forceRedownload
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

            target_path = payload['destination_path']
            result_path = target_path + '.download_result.json'
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            try:
                downloaded = hf_hub_download(
                    repo_id=payload['repo_id'],
                    filename=payload['file_name'],
                    local_dir=os.path.dirname(target_path),
                    force_download=payload.get('force_redownload', True),
                )

                if os.path.abspath(downloaded) != os.path.abspath(target_path):
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    shutil.copy2(downloaded, target_path)

                result = {'ok': True, 'downloaded_path': target_path, 'size_bytes': os.path.getsize(target_path)}
            except Exception as ex:
                result = {'ok': False, 'error': str(ex)}

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
            """;

            await File.WriteAllTextAsync(scriptPath, py, ct);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            string pythonResultPath = destinationPath + ".download_result.json";
            File.Move(pythonResultPath, resultPath, overwrite: true);

            var json = JsonDocument.Parse(await File.ReadAllTextAsync(resultPath, ct)).RootElement;
            if (!json.GetProperty("ok").GetBoolean())
                throw new InvalidOperationException($"Hugging Face file download failed: {json.GetProperty("error").GetString()}");

            if (!File.Exists(destinationPath) || new FileInfo(destinationPath).Length == 0)
                throw new InvalidOperationException($"Hugging Face file download completed but produced no file: {destinationPath}");

            AnsiConsole.MarkupLine($"[green]Downloaded repository file:[/] {Markup.Escape(repoId)}/{Markup.Escape(fileName)} -> {Markup.Escape(destinationPath)}");
            return destinationPath;
        }
        finally
        {
            TryDelete(payloadPath);
            TryDelete(scriptPath);
            TryDelete(resultPath);
            TryDelete(destinationPath + ".download_result.json");
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

    private static string BuildCanonicalKey(string architectureFamilyName, string repoId, string fileName)
        => BaselineDefinitionResolver.BuildCustomCanonicalKey(architectureFamilyName, repoId, fileName);

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
