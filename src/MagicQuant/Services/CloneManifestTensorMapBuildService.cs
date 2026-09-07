using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed record CloneManifestTensorMapBuildResult(
    string OutputPath,
    bool UsedManifestSubset,
    string EffectiveBaseQuantName,
    IReadOnlyList<string> MissingInManifest,
    IReadOnlyDictionary<string, string> ResolvedTensorTypes);

/// <summary>
/// Clone-mode exact tensor-map builder.
///
/// Normal exact-map builds remain strict. Clone mode can optionally allow a source model
/// to contain extra tensors that are absent from an older manifest. In that case, only the
/// manifest tensors receive explicit --tensor-type overrides; the extra source tensors are
/// intentionally left to llama.cpp's normal base-quant behavior unless a clone-specific
/// missing-manifest base-quant override is provided.
///
/// After a clone artifact is produced, the service re-reads the output GGUF and returns the
/// actual emitted tensor qtypes. That lets the generated clone manifest become the next run's
/// exact recipe without inventing missing tensor overrides before llama.cpp has decided how
/// to store norms and other special tensors.
/// </summary>
public sealed class CloneManifestTensorMapBuildService
{
    public const string AllowMissingManifestTensorsFlag = "allow-missing-manifest-tensors";
    public const string AllowMissingManifestTensorsCliSwitch = "--" + AllowMissingManifestTensorsFlag;
    public const string MissingManifestBaseQuantFlag = "missing-manifest-base-quant";
    public const string MissingManifestBaseQuantCliSwitch = "--" + MissingManifestBaseQuantFlag;

    private static readonly JsonSerializerOptions ManifestJsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    private readonly QuantizationService _quantizationService;
    private readonly ImatrixService _imatrixService;

    public CloneManifestTensorMapBuildService(
        QuantizationService quantizationService,
        ImatrixService imatrixService)
    {
        _quantizationService = quantizationService ?? throw new ArgumentNullException(nameof(quantizationService));
        _imatrixService = imatrixService ?? throw new ArgumentNullException(nameof(imatrixService));
    }

    public async Task<CloneManifestTensorMapBuildResult> BuildAsync(
        IReadOnlyDictionary<string, string> tensorTypes,
        string outputPath,
        string baseQuantName,
        bool allowMissingManifestTensors,
        string? missingManifestBaseQuantName = null,
        bool forceRebuild = false,
        CancellationToken ct = default)
    {
        if (tensorTypes == null || tensorTypes.Count == 0)
            throw new ArgumentException("A clone tensor map must contain at least one tensor entry.", nameof(tensorTypes));

        if (string.IsNullOrWhiteSpace(outputPath))
            throw new InvalidOperationException("Export output path is required.");

        bool hasMissingManifestBaseQuantOverride = !string.IsNullOrWhiteSpace(missingManifestBaseQuantName);
        bool allowManifestSubset = allowMissingManifestTensors || hasMissingManifestBaseQuantOverride;

        string nativeBasePath = await _quantizationService.EnsureBaseModelFileAsync();
        var sourceTensorTypes = await _quantizationService.ReadExactTensorTypesAsync(nativeBasePath, ct);
        var sourceTensorNames = sourceTensorTypes.Keys.ToList();

        var missingInManifest = sourceTensorNames
            .Except(tensorTypes.Keys, StringComparer.Ordinal)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        var unexpectedInManifest = tensorTypes.Keys
            .Except(sourceTensorNames, StringComparer.Ordinal)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        bool exactMatch = missingInManifest.Count == 0 &&
                          unexpectedInManifest.Count == 0 &&
                          sourceTensorNames.Count == tensorTypes.Count;

        if (exactMatch)
        {
            var exactOutputPath = await _quantizationService.BuildExportArtifactFromExactTensorMapAsync(
                tensorTypes: tensorTypes,
                outputPath: outputPath,
                baseQuantName: baseQuantName,
                forceRebuild: forceRebuild,
                ct: ct);

            var resolvedTensorTypes = await CaptureResolvedTensorTypesAsync(exactOutputPath, tensorTypes, ct);
            await PersistResolvedTensorTypesToOutputManifestAsync(exactOutputPath, resolvedTensorTypes, ct);

            return new CloneManifestTensorMapBuildResult(
                OutputPath: exactOutputPath,
                UsedManifestSubset: false,
                EffectiveBaseQuantName: baseQuantName,
                MissingInManifest: Array.Empty<string>(),
                ResolvedTensorTypes: resolvedTensorTypes);
        }

        bool sourceModelIsManifestSuperset = missingInManifest.Count > 0 && unexpectedInManifest.Count == 0;
        if (!allowManifestSubset || !sourceModelIsManifestSuperset)
        {
            throw new InvalidOperationException(BuildManifestMismatchError(
                missingInManifest,
                unexpectedInManifest,
                modelTensorCount: sourceTensorNames.Count,
                manifestTensorCount: tensorTypes.Count,
                includeSubsetHint: sourceModelIsManifestSuperset));
        }

        return await BuildSubsetOverrideCloneAsync(
            inputFile: nativeBasePath,
            outputFile: outputPath,
            tensorTypes: tensorTypes,
            baseQuantName: baseQuantName,
            missingManifestBaseQuantName: missingManifestBaseQuantName,
            missingInManifest: missingInManifest,
            forceRebuild: forceRebuild,
            ct: ct);
    }

    private async Task<CloneManifestTensorMapBuildResult> BuildSubsetOverrideCloneAsync(
        string inputFile,
        string outputFile,
        IReadOnlyDictionary<string, string> tensorTypes,
        string baseQuantName,
        string? missingManifestBaseQuantName,
        IReadOnlyList<string> missingInManifest,
        bool forceRebuild,
        CancellationToken ct)
    {
        var baseQuant = ResolveCloneBaseQuantOrThrow(baseQuantName, missingManifestBaseQuantName);
        bool hasMissingManifestBaseQuantOverride = !string.IsNullOrWhiteSpace(missingManifestBaseQuantName);

        Directory.CreateDirectory(Path.GetDirectoryName(outputFile)!);

        if (!forceRebuild && File.Exists(outputFile) && new FileInfo(outputFile).Length > 0)
        {
            var reusedResolvedTensorTypes = await CaptureResolvedTensorTypesAsync(outputFile, tensorTypes, ct);
            await PersistResolvedTensorTypesToOutputManifestAsync(outputFile, reusedResolvedTensorTypes, ct);

            return new CloneManifestTensorMapBuildResult(
                OutputPath: outputFile,
                UsedManifestSubset: true,
                EffectiveBaseQuantName: baseQuant.Names[0],
                MissingInManifest: missingInManifest.ToArray(),
                ResolvedTensorTypes: reusedResolvedTensorTypes);
        }

        if (forceRebuild)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile + ".success.json");
        }

        AnsiConsole.MarkupLine(
            $"[yellow]Clone manifest subset allowed:[/] [cyan]{missingInManifest.Count:N0}[/] source tensor(s) are absent from the manifest and will receive no explicit --tensor-type override.");

        AnsiConsole.MarkupLine(hasMissingManifestBaseQuantOverride
            ? $"[yellow]Missing-manifest base quant override:[/] [cyan]{Markup.Escape(baseQuant.Names[0])}[/] will be used for source tensors absent from the manifest."
            : $"[grey]Missing-manifest tensors will use artifact base quant:[/] {Markup.Escape(baseQuant.Names[0])}");

        foreach (var tensorName in missingInManifest.Take(15))
            AnsiConsole.MarkupLine($"[grey]  basequant fallback tensor:[/] {Markup.Escape(tensorName)}");

        if (missingInManifest.Count > 15)
            AnsiConsole.MarkupLine($"[grey]  ...and {missingInManifest.Count - 15:N0} more tensor(s).[/]");

        var args = new List<string>(capacity: tensorTypes.Count * 2 + 8);
        foreach (var kv in tensorTypes.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            args.Add("--tensor-type");
            args.Add($"{kv.Key}={NormalizeCloneQuantName(kv.Value)}");
        }

        if (_imatrixService.ShouldUseImatrixForQuant(HybridQuant.CreatePureBaseline(baseQuant)))
        {
            string imatrixPath = _imatrixService.GetCanonicalImatrixPath();
            if (!File.Exists(imatrixPath))
                throw new InvalidOperationException($"Imatrix was marked active but canonical artifact is missing: {imatrixPath}");

            args.Add("--imatrix");
            args.Add(imatrixPath);
        }

        args.Add(inputFile);
        args.Add(outputFile);
        args.Add(baseQuant.QuantizeBaseArgumentName);
        args.Add(ResolveCloneQuantizeThreadCount().ToString());

        string bin = Path.Combine(
            Cache.LlamaBin!,
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llama-quantize.exe" : "llama-quantize");

        string quantizeLogPath = outputFile + ".quantize.log";
        Directory.CreateDirectory(Path.GetDirectoryName(quantizeLogPath)!);

        var psi = new ProcessStartInfo
        {
            FileName = bin
        };

        foreach (var arg in args)
            psi.ArgumentList.Add(arg);

        AnsiConsole.MarkupLine(
            $"[cyan]Quantizing clone artifact from manifest subset:[/] {Markup.Escape(Path.GetFileName(outputFile))} [grey](base quant: {Markup.Escape(baseQuant.Names[0])}; log: {Markup.Escape(quantizeLogPath)})[/]");

        var result = await RunLoggedProcessAsync(psi, quantizeLogPath, ct);
        if (result.ExitCode != 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);
            throw new InvalidOperationException(
                $"Clone quantization failed for '{outputFile}'. ExitCode={result.ExitCode}. See '{quantizeLogPath}'.");
        }

        if (!File.Exists(outputFile) || new FileInfo(outputFile).Length == 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);
            throw new InvalidOperationException(
                $"Clone quantization exited successfully but produced no valid GGUF output: {outputFile}");
        }

        await File.WriteAllTextAsync(outputFile + ".success.json", "{\"status\":\"success\"}", ct);
        AnsiConsole.MarkupLine($"[green]Clone quantized model ready:[/] {Markup.Escape(outputFile)}");

        var resolvedTensorTypes = await CaptureResolvedTensorTypesAsync(outputFile, tensorTypes, ct);
        await PersistResolvedTensorTypesToOutputManifestAsync(outputFile, resolvedTensorTypes, ct);

        return new CloneManifestTensorMapBuildResult(
            OutputPath: outputFile,
            UsedManifestSubset: true,
            EffectiveBaseQuantName: baseQuant.Names[0],
            MissingInManifest: missingInManifest.ToArray(),
            ResolvedTensorTypes: resolvedTensorTypes);
    }

    private async Task<IReadOnlyDictionary<string, string>> CaptureResolvedTensorTypesAsync(
        string outputFile,
        IReadOnlyDictionary<string, string> fallbackTensorTypes,
        CancellationToken ct)
    {
        try
        {
            var resolved = await _quantizationService.ReadExactTensorTypesAsync(outputFile, ct);
            if (resolved.Count > 0)
            {
                AnsiConsole.MarkupLine($"[green]Captured resolved clone tensor map from GGUF:[/] {resolved.Count:N0} tensor(s)");
                return resolved
                    .OrderBy(x => x.Key, StringComparer.Ordinal)
                    .ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal);
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]Could not capture resolved tensor map from clone GGUF; preserving manifest tensor map:[/] {Markup.Escape(ex.Message)}");
        }

        return fallbackTensorTypes
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal);
    }

    private static async Task PersistResolvedTensorTypesToOutputManifestAsync(
        string outputFile,
        IReadOnlyDictionary<string, string> resolvedTensorTypes,
        CancellationToken ct)
    {
        if (resolvedTensorTypes.Count == 0)
            return;

        string? outputDirectory = Path.GetDirectoryName(outputFile);
        if (string.IsNullOrWhiteSpace(outputDirectory))
            return;

        string manifestPath = MagicQuantManifestPathService.GetManifestFilePath(outputDirectory, MagicQuantManifestPathService.CloneConfigsFileName);
        if (!File.Exists(manifestPath))
            return;

        try
        {
            var root = JsonNode.Parse(await File.ReadAllTextAsync(manifestPath, ct)) as JsonObject;
            var artifacts = TryGetProperty(root, "artifacts") as JsonArray;
            if (root == null || artifacts == null)
                return;

            string fileName = Path.GetFileName(outputFile);
            JsonObject? matchingArtifact = null;
            foreach (var node in artifacts.OfType<JsonObject>())
            {
                string? artifactFileName = TryGetString(node, "fileName");
                if (string.Equals(artifactFileName, fileName, StringComparison.OrdinalIgnoreCase))
                {
                    matchingArtifact = node;
                    break;
                }
            }

            if (matchingArtifact == null)
                return;

            var tensorTypesNode = new JsonObject();
            foreach (var kv in resolvedTensorTypes.OrderBy(x => x.Key, StringComparer.Ordinal))
                tensorTypesNode[kv.Key] = kv.Value;

            matchingArtifact["tensorTypes"] = tensorTypesNode;
            matchingArtifact["resolvedTensorTypeCount"] = resolvedTensorTypes.Count;
            matchingArtifact["resolvedTensorTypesGeneratedAtUtc"] = DateTimeOffset.UtcNow.ToString("O");

            await File.WriteAllTextAsync(manifestPath, root.ToJsonString(ManifestJsonOptions), ct);
            AnsiConsole.MarkupLine(
                $"[green]Resolved tensorTypes persisted to clone manifest:[/] {Markup.Escape(fileName)} [grey]({resolvedTensorTypes.Count:N0} tensor(s))[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Could not persist resolved tensorTypes to clone manifest; build output is still valid:[/] {Markup.Escape(ex.Message)}");
        }
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

    private static BaselineQuants ResolveCloneBaseQuantOrThrow(string baseQuantName, string? missingManifestBaseQuantName)
    {
        bool hasOverride = !string.IsNullOrWhiteSpace(missingManifestBaseQuantName);
        string quantName = hasOverride ? missingManifestBaseQuantName!.Trim() : baseQuantName;

        var resolved = BaselineQuants.ResolveBuiltInStandardBaseline(quantName);
        if (resolved != null)
            return resolved;

        if (hasOverride)
        {
            throw new InvalidOperationException(
                $"Unknown {MissingManifestBaseQuantCliSwitch} value '{missingManifestBaseQuantName}'. " +
                "Use a built-in llama.cpp base quant name such as Q8_0, Q6_K, Q5_K_M, or Q4_K_M.");
        }

        return BaselineQuants.Q8_0;
    }

    private static string BuildManifestMismatchError(
        IReadOnlyList<string> missingInManifest,
        IReadOnlyList<string> unexpectedInManifest,
        int modelTensorCount,
        int manifestTensorCount,
        bool includeSubsetHint)
    {
        var builder = new StringBuilder();
        builder.Append("Clone tensor manifest does not exactly match this model architecture. ");
        builder.Append($"MissingInManifest=[{string.Join(", ", missingInManifest.Take(20))}] ");
        builder.Append($"UnexpectedInManifest=[{string.Join(", ", unexpectedInManifest.Take(20))}] ");
        builder.Append($"ModelTensorCount={modelTensorCount} ManifestTensorCount={manifestTensorCount}.");

        if (includeSubsetHint)
        {
            builder.AppendLine();
            builder.Append("This looks like a clone manifest subset: every manifest tensor exists in the current model, ");
            builder.Append("but the current model has extra tensors. To let those extra tensors fall through to the artifact base quant, rerun clone mode with ");
            builder.Append(AllowMissingManifestTensorsCliSwitch);
            builder.Append(". To force those extra tensors to a specific base quant, use ");
            builder.Append(MissingManifestBaseQuantCliSwitch);
            builder.Append(" Q8_0.");
        }

        return builder.ToString();
    }

    private static string NormalizeCloneQuantName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "UNKNOWN";

        string token = value.Trim().Replace("-", "_").Replace(" ", string.Empty).ToUpperInvariant();

        foreach (var scheme in TensorWeightScheme.All)
        {
            if (scheme.Names.IsDefaultOrEmpty)
                continue;

            if (scheme.Names.Any(name => string.Equals(
                    name.Trim().Replace("-", "_").Replace(" ", string.Empty).ToUpperInvariant(),
                    token,
                    StringComparison.Ordinal)))
            {
                return scheme.Names[0];
            }
        }

        return token;
    }

    private static int ResolveCloneQuantizeThreadCount()
    {
        int threadCount = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        int reservedThreads = threadCount switch
        {
            >= 16 => 2,
            >= 8 => 2,
            >= 4 => 1,
            _ => 0
        };

        return Math.Max(1, threadCount - reservedThreads);
    }

    private static async Task<LoggedProcessResult> RunLoggedProcessAsync(
        ProcessStartInfo psi,
        string logPath,
        CancellationToken ct)
    {
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;

        using var process = new Process
        {
            StartInfo = psi,
            EnableRaisingEvents = true
        };

        var stdoutBuilder = new StringBuilder();
        var stderrBuilder = new StringBuilder();
        object sync = new();

        var stdoutClosed = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var stderrClosed = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        await using var logStream = new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read);
        await using var logWriter = new StreamWriter(logStream) { AutoFlush = true };

        void HandleLine(string? line, bool isError)
        {
            if (line == null)
            {
                if (isError)
                    stderrClosed.TrySetResult(true);
                else
                    stdoutClosed.TrySetResult(true);

                return;
            }

            lock (sync)
            {
                if (isError)
                    stderrBuilder.AppendLine(line);
                else
                    stdoutBuilder.AppendLine(line);

                logWriter.WriteLine(line);
            }

            if (Cache.VerboseProcessOutput)
                AnsiConsole.WriteLine(line);
        }

        process.OutputDataReceived += (_, e) => HandleLine(e.Data, isError: false);
        process.ErrorDataReceived += (_, e) => HandleLine(e.Data, isError: true);

        if (!process.Start())
            throw new InvalidOperationException($"Failed to start process: {psi.FileName}");

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using var ctr = ct.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                    process.Kill(entireProcessTree: true);
            }
            catch
            {
            }
        });

        await process.WaitForExitAsync(ct);
        await Task.WhenAll(stdoutClosed.Task, stderrClosed.Task);

        return new LoggedProcessResult
        {
            ExitCode = process.ExitCode,
            StdOut = stdoutBuilder.ToString(),
            StdErr = stderrBuilder.ToString()
        };
    }

    private sealed class LoggedProcessResult
    {
        public int ExitCode { get; init; }
        public string StdOut { get; init; } = string.Empty;
        public string StdErr { get; init; } = string.Empty;
    }
}
