using System.Diagnostics;
using System.Text.Json;
using MagicQuant.Helpers;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed record VisionCapabilityDetection(
    bool IsLikelyVisionCapable,
    bool IsStrongSignal,
    List<string> Reasons,
    List<string> Warnings);

public sealed record MmprojArtifactResult
{
    public bool IsVisionCapable { get; init; }
    public bool ExistingFound { get; init; }
    public bool Built { get; init; }
    public bool Copied { get; init; }
    public string? SourcePath { get; init; }
    public string? OutputPath { get; init; }
    public List<string> Warnings { get; init; } = new();
}

public sealed class ModelSidecarArtifactService
{
    private static readonly string[] MultimodalTokens =
    [
        "llava", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "gemma3", "internvl", "minicpm", "phi4mm", "glmv", "mllama", "idefics", "florence", "paligemma"
    ];

    private readonly PythonManager _pythonManager;

    public ModelSidecarArtifactService(PythonManager pythonManager)
    {
        _pythonManager = pythonManager;
    }

    public async Task<MmprojArtifactResult> EnsureMmprojArtifactAvailableAsync(CancellationToken ct = default)
    {
        var detection = DetectVisionCapability();
        var warnings = new List<string>(detection.Warnings);
        string? existing = FindExistingMmprojArtifact();
        if (existing != null)
        {
            AnsiConsole.MarkupLine($"[green]mmproj artifact ready:[/] {Markup.Escape(existing)}");
            return new MmprojArtifactResult { IsVisionCapable = detection.IsLikelyVisionCapable, ExistingFound = true, SourcePath = existing, Warnings = warnings };
        }

        if (!detection.IsLikelyVisionCapable)
        {
            AnsiConsole.MarkupLine("[grey]No mmproj artifact present and no strong vision capability hints detected. Continuing.[/]");
            return new MmprojArtifactResult { IsVisionCapable = false, Warnings = warnings };
        }

        if (!Config.AttemptMmprojBuild || !detection.IsStrongSignal)
        {
            string warning = "Model has vision-capability hints, but no mmproj GGUF was found or generated. Continuing without mmproj sidecar. Vision inference may require a separate --mmproj file.";
            warnings.Add(warning);
            AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(warning)}[/]");
            return HandleStrictRequirement(new MmprojArtifactResult { IsVisionCapable = true, Warnings = warnings }, warning);
        }

        var built = await BuildMmprojArtifactAsync(warnings, ct);
        if (!built.Built)
            return HandleStrictRequirement(built, warnings.LastOrDefault() ?? "mmproj build failed.");

        return built;
    }

    public async Task<MmprojArtifactResult> CopyMmprojArtifactsAsync(string outputDirectory, CancellationToken ct = default)
    {
        var detection = DetectVisionCapability();
        var warnings = new List<string>(detection.Warnings);
        string? source = FindExistingMmprojArtifact();
        if (source == null)
        {
            if (detection.IsLikelyVisionCapable)
            {
                string warning = "Model has vision-capability hints, but no mmproj GGUF was found or generated. Continuing without mmproj sidecar. Vision inference may require a separate --mmproj file.";
                warnings.Add(warning);
                AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(warning)}[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[grey]No mmproj artifact present and no strong vision capability hints detected. Continuing.[/]");
            }

            return new MmprojArtifactResult { IsVisionCapable = detection.IsLikelyVisionCapable, Warnings = warnings };
        }

        Directory.CreateDirectory(outputDirectory);
        string target = Path.Combine(outputDirectory, Path.GetFileName(source));
        File.Copy(source, target, overwrite: true);
        await Task.Yield();
        AnsiConsole.MarkupLine($"[green]Copied mmproj artifact:[/] {Markup.Escape(target)}");

        return new MmprojArtifactResult { IsVisionCapable = detection.IsLikelyVisionCapable, ExistingFound = true, Copied = true, SourcePath = source, OutputPath = target, Warnings = warnings };
    }

    public string? FindExistingMmprojArtifact()
    {
        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory) || string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            return null;

        var roots = new[]
        {
            Cache.ModelDirectory!,
            Cache.ModelMagicQuantDirectory!,
            Path.Combine(Cache.ModelMagicQuantDirectory!, "Sidecars"),
            Path.Combine(Cache.ModelMagicQuantDirectory!, "GGUF")
        };

        foreach (var root in roots.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!Directory.Exists(root))
                continue;

            var found = Directory.EnumerateFiles(root, "*mmproj*.gguf", SearchOption.TopDirectoryOnly)
                .FirstOrDefault(path => new FileInfo(path).Length > 0);
            if (found != null)
                return found;
        }

        return null;
    }

    public VisionCapabilityDetection DetectVisionCapability()
    {
        var reasons = new List<string>();
        var warnings = new List<string>();
        if (string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            return new VisionCapabilityDetection(false, false, reasons, warnings);

        string configPath = Path.Combine(Cache.ModelDirectory!, "config.json");
        if (!File.Exists(configPath))
        {
            warnings.Add("config.json not found; unable to evaluate vision capability hints.");
            return new VisionCapabilityDetection(false, false, reasons, warnings);
        }

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
            var root = doc.RootElement;
            string[] strongKeys = ["vision_config", "vision_tower", "mm_vision_tower", "visual", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"];
            foreach (var key in strongKeys)
            {
                if (root.TryGetProperty(key, out _))
                    reasons.Add($"config has '{key}'");
            }

            if (root.TryGetProperty("model_type", out var modelType) && modelType.ValueKind == JsonValueKind.String)
            {
                string value = modelType.GetString() ?? string.Empty;
                if (MultimodalTokens.Any(t => value.Contains(t, StringComparison.OrdinalIgnoreCase)))
                    reasons.Add($"model_type suggests multimodal: {value}");
            }

            if (root.TryGetProperty("architectures", out var archs) && archs.ValueKind == JsonValueKind.Array)
            {
                foreach (var arch in archs.EnumerateArray())
                {
                    var value = arch.GetString() ?? string.Empty;
                    if (MultimodalTokens.Any(t => value.Contains(t, StringComparison.OrdinalIgnoreCase)))
                        reasons.Add($"architecture suggests multimodal: {value}");
                }
            }
        }
        catch (Exception ex)
        {
            warnings.Add($"Failed to parse config.json for vision capability hints: {ex.Message}");
            return new VisionCapabilityDetection(false, false, reasons, warnings);
        }

        bool likely = reasons.Count > 0;
        bool strong = reasons.Any(r => r.Contains("config has", StringComparison.OrdinalIgnoreCase));
        return new VisionCapabilityDetection(likely, strong, reasons, warnings);
    }

    private async Task<MmprojArtifactResult> BuildMmprojArtifactAsync(List<string> warnings, CancellationToken ct)
    {
        string sidecarDir = Path.Combine(Cache.ModelMagicQuantDirectory!, "Sidecars");
        Directory.CreateDirectory(sidecarDir);
        string safeName = new DirectoryInfo(Cache.ModelDirectory!).Name.Replace(' ', '-');
        string targetPath = Path.Combine(sidecarDir, $"mmproj-{safeName}-f16.gguf");
        string successPath = targetPath + ".success.json";
        string logPath = targetPath + ".convert.log";

        if (File.Exists(targetPath) && new FileInfo(targetPath).Length > 0 && File.Exists(successPath))
        {
            AnsiConsole.MarkupLine($"[green]mmproj artifact ready:[/] {Markup.Escape(targetPath)}");
            return new MmprojArtifactResult { IsVisionCapable = true, ExistingFound = true, SourcePath = targetPath, Warnings = warnings };
        }

        var psi = new ProcessStartInfo
        {
            FileName = _pythonManager.GetPythonExecutable(),
            WorkingDirectory = Cache.LlamaRoot,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        psi.ArgumentList.Add("convert_hf_to_gguf.py");
        psi.ArgumentList.Add(Cache.ModelDirectory!);
        psi.ArgumentList.Add("--mmproj");
        psi.ArgumentList.Add("--outtype");
        psi.ArgumentList.Add("f16");
        psi.ArgumentList.Add("--outfile");
        psi.ArgumentList.Add(targetPath);

        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start mmproj conversion process.");
        string stdout = await proc.StandardOutput.ReadToEndAsync();
        string stderr = await proc.StandardError.ReadToEndAsync();
        await proc.WaitForExitAsync(ct);
        await File.WriteAllTextAsync(logPath, stdout + Environment.NewLine + stderr, ct);

        if (proc.ExitCode != 0 || !File.Exists(targetPath) || new FileInfo(targetPath).Length == 0)
        {
            if (File.Exists(targetPath))
                await HardDeleteHelper.DeleteFileIfExistsAsync(targetPath);

            string warning = $"Model has vision-capability hints, but no mmproj GGUF was found or generated. Continuing without mmproj sidecar. Vision inference may require a separate --mmproj file. Log: {logPath}";
            warnings.Add(warning);
            AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(warning)}[/]");
            return new MmprojArtifactResult { IsVisionCapable = true, Warnings = warnings };
        }

        await File.WriteAllTextAsync(successPath, "{\"status\":\"success\"}", ct);
        AnsiConsole.MarkupLine($"[green]mmproj artifact ready:[/] {Markup.Escape(targetPath)}");
        return new MmprojArtifactResult { IsVisionCapable = true, Built = true, SourcePath = targetPath, Warnings = warnings };
    }

    private static MmprojArtifactResult HandleStrictRequirement(MmprojArtifactResult result, string message)
    {
        if (Config.RequireMmprojForVisionModels && result.IsVisionCapable && !result.ExistingFound && !result.Built)
            throw new InvalidOperationException(message);

        return result;
    }
}
