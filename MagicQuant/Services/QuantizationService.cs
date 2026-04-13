using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Services;

public enum SampleProcessState
{
    Completed = 1,
    Skipped = 2,
    Failed = 3
}

public sealed class SampleProcessingSummary
{
    public int Requested { get; set; }
    public int Completed { get; set; }
    public int Skipped { get; set; }
    public int Failed { get; set; }
}

public class QuantizationService
{
    private readonly BenchmarkService _benchmarker;
    private readonly string _ggufDir;
    private readonly string _benchDir;
    private readonly PythonManager _python;
    private readonly SemaphoreSlim _cpuQuantLock;
    private readonly int _maxConcurrentQuantizations;

    private static readonly SemaphoreSlim BaseModelLock = new(1, 1);

    public QuantizationService(BenchmarkService benchmarker)
    {
        _benchmarker = benchmarker ?? throw new ArgumentNullException(nameof(benchmarker));
        _python = _benchmarker._pyManager;

        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new Exception(
                "Cache.ModelMagicQuantDirectory not set. Evolution must set this before quantization starts.");

        if (string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            throw new Exception("Cache.ModelDirectory not set. Evolution must set this before quantization starts.");

        if (string.IsNullOrWhiteSpace(Cache.LlamaBin))
            throw new Exception("Cache.LlamaBin not set. Initialization must complete before quantization starts.");

        _ggufDir = Path.Combine(Cache.ModelMagicQuantDirectory, "GGUF");
        _benchDir = Path.Combine(Cache.ModelMagicQuantDirectory, "Benchmarks");

        Directory.CreateDirectory(_ggufDir);
        Directory.CreateDirectory(_benchDir);

        int threadCount = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        _maxConcurrentQuantizations = Math.Max(1, threadCount / 8);
        _cpuQuantLock = new SemaphoreSlim(_maxConcurrentQuantizations, _maxConcurrentQuantizations);
    }

    // ----------------------------------------------------------------
    // Batch processing
    // ----------------------------------------------------------------

    public async Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<HybridQuant> quants,
        CancellationToken ct = default)
    {
        if (quants == null)
            throw new ArgumentNullException(nameof(quants));

        int completed = 0;
        int skipped = 0;
        int failed = 0;

        // Warm the base model file once so workers don't all race into conversion.
        await EnsureBaseModelFileAsync(false);

        await Parallel.ForEachAsync(
            quants,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = _maxConcurrentQuantizations,
                CancellationToken = ct
            },
            async (quant, token) =>
            {
                try
                {
                    var state = await ProcessHybridQuantAsync(quant, token);

                    switch (state)
                    {
                        case SampleProcessState.Completed:
                            Interlocked.Increment(ref completed);
                            break;
                        case SampleProcessState.Skipped:
                            Interlocked.Increment(ref skipped);
                            break;
                        default:
                            Interlocked.Increment(ref failed);
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref failed);
                    AnsiConsole.MarkupLine($"[red]Sample failed:[/] {Markup.Escape(GenerateHybridName(quant))}");
                    AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.Message)}[/]");
                }
            });

        return new SampleProcessingSummary
        {
            Requested = quants.Count,
            Completed = completed,
            Skipped = skipped,
            Failed = failed
        };
    }

    public async Task<SampleProcessState> ProcessHybridQuantAsync(
        HybridQuant quant,
        CancellationToken ct = default)
    {
        string modelName = GenerateHybridName(quant);
        string quantPath = Path.Combine(_ggufDir, $"{modelName}.gguf");
        string modelBenchDir = Path.Combine(_benchDir, modelName);
        string baseLogitsDir = GetBaseLogitsDirectory();

        // 1. Fast path: valid artifacts already exist on disk and can be synced/reused
        if (await _benchmarker.TryReuseExistingBenchmarksAsync(
                quantConfig: quant,
                modelPath: quantPath,
                benchDir: modelBenchDir,
                klLogitsDir: baseLogitsDir,
                domainsOverride: new[] { "general" }))
        {
            AnsiConsole.MarkupLine($"[grey]Reused existing benchmark artifacts:[/] {Markup.Escape(modelName)}");

            if (!IsProtectedModel(modelName))
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

            return SampleProcessState.Skipped;
        }

        // 2. DB truth still matters too
        if (await BenchmarkExistsAsync(quant, ct))
        {
            AnsiConsole.MarkupLine($"[grey]Skipping already completed sample:[/] {Markup.Escape(modelName)}");

            if (!IsProtectedModel(modelName))
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

            return SampleProcessState.Skipped;
        }

        try
        {
            string basePath = await EnsureBaseModelFileAsync();

            await _cpuQuantLock.WaitAsync(ct);
            try
            {
                if (!File.Exists(quantPath))
                {
                    AnsiConsole.MarkupLine($"[cyan]Building sample:[/] {Markup.Escape(modelName)}");
                    await RunLlamaQuantizeAsync(basePath, quantPath, quant);
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            // Re-check after build in case another worker finished the DB sync while we were quantizing
            if (await BenchmarkExistsAsync(quant, ct))
            {
                if (!IsProtectedModel(modelName))
                    await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

                return SampleProcessState.Skipped;
            }

            AnsiConsole.MarkupLine($"[yellow]Benchmarking:[/] {Markup.Escape(modelName)}");

            await _benchmarker.RunAllBenchmarksAsync(
                quantConfig: quant,
                modelPath: quantPath,
                benchDir: modelBenchDir,
                klLogitsDir: baseLogitsDir,
                saveLogits: false,
                domainsOverride: new[] { "general" });

            return SampleProcessState.Completed;
        }
        finally
        {
            if (!IsProtectedModel(modelName))
            {
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);
            }
        }
    }

    // ----------------------------------------------------------------
    // Benchmark/logit helpers
    // ----------------------------------------------------------------

    private string GetBaseLogitsDirectory()
    {
        string typeStr = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        return Path.Combine(_benchDir, typeStr, "logits");
    }

    private async Task<bool> BenchmarkExistsAsync(HybridQuant quant, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var lookup = BuildTensorLookup(quant);

        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null)
            return false;

        var bench = await db.AiBenchmarks
            .AsNoTracking()
            .Where(x => x.AiModelHashId == model.Id)
            .Join(
                db.TensorCombos.AsNoTracking(),
                benchmark => benchmark.TensorComboId,
                combo => combo.Id,
                (benchmark, combo) => new { benchmark, combo })
            .Where(x =>
                x.combo.BaseQuant == lookup.BaseQuant &&
                x.combo.Embeddings == lookup.Embeddings &&
                x.combo.LmHead == lookup.LmHead &&
                x.combo.AttnQ == lookup.AttnQ &&
                x.combo.AttnKV == lookup.AttnKV &&
                x.combo.AttnOutput == lookup.AttnOutput &&
                x.combo.FfnUpGate == lookup.FfnUpGate &&
                x.combo.FfnDown == lookup.FfnDown &&
                x.combo.MoeExperts == lookup.MoeExperts &&
                x.combo.MoeRouter == lookup.MoeRouter)
            .Select(x => x.benchmark.Id)
            .FirstOrDefaultAsync(ct);

        if (bench == 0)
            return false;

        // Require at least one category row too, so a half-baked parent row doesn't count as complete.
        bool hasCategory = await db.Set<MQ.DB.Models.DbModels.CategoryBenchmark>()
            .AsNoTracking()
            .AnyAsync(x => x.AiBenchmarkId == bench, ct);

        return hasCategory;
    }

    private static (
        byte BaseQuant,
        byte Embeddings,
        byte LmHead,
        byte AttnQ,
        byte AttnKV,
        byte AttnOutput,
        byte FfnUpGate,
        byte FfnDown,
        byte MoeExperts,
        byte MoeRouter) BuildTensorLookup(HybridQuant quant)
    {
        byte embeddings = 0;
        byte lmHead = 0;
        byte attnQ = 0;
        byte attnKV = 0;
        byte attnOutput = 0;
        byte ffnUpGate = 0;
        byte ffnDown = 0;
        byte moeExperts = 0;
        byte moeRouter = 0;

        if (quant.Tensors != null)
        {
            foreach (var tensor in quant.Tensors)
            {
                if (tensor?.TGroup == null)
                    continue;

                if (tensor.TGroup.UniqueId == TReg.Embeddings.UniqueId) embeddings = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.LmHead.UniqueId) lmHead = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.AttnQ.UniqueId) attnQ = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.AttnKV.UniqueId) attnKV = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.AttnOutput.UniqueId) attnOutput = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.FfnUpGate.UniqueId) ffnUpGate = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.FfnDown.UniqueId) ffnDown = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.MoeExperts.UniqueId) moeExperts = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.MoeRouter.UniqueId) moeRouter = tensor.TensorType.UniqueId;
            }
        }

        return (
            quant.BaseQuant.UniqueId,
            embeddings,
            lmHead,
            attnQ,
            attnKV,
            attnOutput,
            ffnUpGate,
            ffnDown,
            moeExperts,
            moeRouter
        );
    }

    private bool IsProtectedModel(string name)
    {
        return name.EndsWith("BF16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F32", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("Q8_0", StringComparison.OrdinalIgnoreCase);
    }

    // ----------------------------------------------------------------
    // Base/native model helpers
    // ----------------------------------------------------------------

    public async Task<string> EnsureBaseModelAsync(bool deleteProcess = false)
    {
        string outputPath = await EnsureBaseModelFileAsync(deleteProcess);

        string typeStr = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        string benchPath = Path.Combine(_benchDir, typeStr);
        string logitsDir = Path.Combine(benchPath, "logits");

        AnsiConsole.MarkupLine($"[bold yellow]Benchmarking Base {typeStr} (Saving Logits)...[/]");

        var baseModelQuant = new HybridQuant
        {
            BaseQuant = BaselineQuants.GetBF16Quant(),
            Tensors = new List<HybridTensor>()
        };

        await _benchmarker.RunAllBenchmarksAsync(
            quantConfig: baseModelQuant,
            modelPath: outputPath,
            benchDir: benchPath,
            klLogitsDir: logitsDir,
            saveLogits: true,
            domainsOverride: new[] { "general", "code", "math" }
        );

        return outputPath;
    }

    public async Task<string> EnsureBaseModelFileAsync(bool deleteProcess = false)
    {
        await BaseModelLock.WaitAsync();
        try
        {
            string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
            var torchType = Cache.TorchType ?? Cache.MainTorchType.BF16;
            string typeStr = torchType.ToString();

            string fileName = $"{modelName}-{typeStr}.gguf";
            string outputPath = Path.Combine(_ggufDir, fileName);
            string successFile = Path.Combine(_ggufDir, $"{fileName}.success.json");
            string convertLogPath = outputPath + ".convert.log";

            if (deleteProcess)
            {
                if (!Directory.Exists(_ggufDir))
                    Directory.CreateDirectory(_ggufDir);

                var normalizedFileName = Path.GetFileName(fileName);
                var successFileName = normalizedFileName + ".success.json";
                var successFilePath = Path.Combine(_ggufDir, successFileName);
                bool isImmune = File.Exists(successFilePath);

                foreach (var filePath in Directory.EnumerateFiles(_ggufDir, "*.gguf", SearchOption.TopDirectoryOnly))
                {
                    var currentFileName = Path.GetFileName(filePath);

                    if (isImmune &&
                        string.Equals(currentFileName, normalizedFileName, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    await HardDeleteHelper.DeleteFileIfExistsAsync(filePath);
                }
            }

            if (!File.Exists(outputPath) || !File.Exists(successFile))
            {
                AnsiConsole.MarkupLine($"[bold cyan]Converting to {typeStr}...[/]");

                await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                string convertScript = Cache.ConvertScript
                                       ?? throw new Exception("ConvertScript path missing in Cache");

                string outTypeArg = typeStr.ToLowerInvariant();

                string arguments =
                    $"\"{convertScript}\" \"{Cache.ModelDirectory}\" " +
                    $"--outtype {outTypeArg} " +
                    $"--outfile \"{outputPath}\"";

                string python = _python.GetPythonExecutable();

                var psi = new ProcessStartInfo
                {
                    FileName = python,
                    Arguments = arguments,
                    WorkingDirectory = Cache.LlamaRoot
                };

                var result = await RunLoggedProcessAsync(psi, convertLogPath);

                if (result.ExitCode != 0)
                {
                    await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                    throw new Exception(
                        $"{typeStr} conversion failed. ExitCode={result.ExitCode}. See '{convertLogPath}'.");
                }

                if (!File.Exists(outputPath) || new FileInfo(outputPath).Length == 0)
                {
                    await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                    throw new InvalidOperationException(
                        $"Conversion exited successfully but produced no valid GGUF output: {outputPath}");
                }

                await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");
            }

            return outputPath;
        }
        finally
        {
            BaseModelLock.Release();
        }
    }

    public async Task<string> EnsurePureQ8ModelAsync()
    {
        string basePath = await EnsureBaseModelFileAsync();

        var pureQ8 = new HybridQuant
        {
            BaseQuant = BaselineQuants.Q8_0,
            Tensors = new List<HybridTensor>()
        };

        string modelName = GenerateHybridName(pureQ8);
        string q8Path = Path.Combine(_ggufDir, $"{modelName}.gguf");
        string successFile = Path.Combine(_ggufDir, $"{Path.GetFileName(q8Path)}.success.json");

        if (!File.Exists(q8Path) || !File.Exists(successFile))
        {
            await _cpuQuantLock.WaitAsync();
            try
            {
                if (!File.Exists(q8Path))
                {
                    AnsiConsole.MarkupLine($"[cyan]Building pure Q8 baseline:[/] {Markup.Escape(modelName)}");
                    await RunLlamaQuantizeAsync(basePath, q8Path, pureQ8);
                    AnsiConsole.MarkupLine(
                        $"[green]Pure Q8 baseline quantization finished:[/] {Markup.Escape(q8Path)}");
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");
        }
        else
        {
            AnsiConsole.MarkupLine($"[grey]Pure Q8 baseline already exists:[/] {Markup.Escape(q8Path)}");
        }

        return q8Path;
    }

    // ----------------------------------------------------------------
    // Quantization
    // ----------------------------------------------------------------

    public async Task RunLlamaQuantizeAsync(string inputFile, string outputFile, HybridQuant quant)
    {
        if (string.IsNullOrWhiteSpace(inputFile) || !File.Exists(inputFile))
            throw new FileNotFoundException($"Input GGUF not found: {inputFile}");

        Directory.CreateDirectory(Path.GetDirectoryName(outputFile)!);

        var requestedOverrides = BuildRequestedTensorOverrides(quant);

        // Keep this resolution step:
        // it is not output validation; it is how logical group rules become real tensor names.
        var concreteOverrides = await ResolveConcreteTensorOverridesAsync(
            inputGgufPath: inputFile,
            outputFilePath: outputFile,
            requestedOverrides: requestedOverrides);

        if (requestedOverrides.Count > 0 && concreteOverrides.Count == 0)
        {
            throw new InvalidOperationException(
                $"No concrete tensors were resolved for requested overrides when quantizing '{outputFile}'. " +
                "This means the requested tensor selectors did not match the input GGUF.");
        }

        var args = new List<string>(capacity: 256);

        foreach (var overrideItem in concreteOverrides)
        {
            args.Add($"--tensor-type \"{overrideItem.TensorName}={overrideItem.SchemeName}\"");
        }

        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");
        args.Add(ResolveQuantizeBaseArgument(quant, concreteOverrides));
        args.Add("8");

        string arguments = string.Join(" ", args);

        string bin = Path.Combine(
            Cache.LlamaBin!,
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llama-quantize.exe" : "llama-quantize");

        string quantizeLogPath = outputFile + ".quantize.log";

        var psi = new ProcessStartInfo
        {
            FileName = bin,
            Arguments = arguments
        };

        var result = await RunLoggedProcessAsync(psi, quantizeLogPath);

        if (result.ExitCode != 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);

            throw new InvalidOperationException(
                $"Quantization failed for '{outputFile}'. ExitCode={result.ExitCode}. See '{quantizeLogPath}'.");
        }

        if (!File.Exists(outputFile) || new FileInfo(outputFile).Length == 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);

            throw new InvalidOperationException(
                $"Quantization process exited successfully but produced no valid GGUF output: {outputFile}");
        }

        AnsiConsole.MarkupLine($"[green]Quantized model ready:[/] {Markup.Escape(outputFile)}");
    }

    private static string ResolveQuantizeBaseArgument(
        HybridQuant quant,
        List<ConcreteTensorOverride> concreteOverrides)
    {
        if (quant.BaseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId &&
            concreteOverrides.Count > 0)
        {
            throw new InvalidOperationException(
                "Selective tensor overrides cannot use a native-source base quant (BF16/F16/F32). " +
                "A real base quant such as Q8_0, Q6_K, Q5_K, Q4_K_M, or IQ4_XS must be provided.");
        }

        return ResolveBaseName(quant.BaseQuant);
    }
    
    private static TensorWeightScheme? TryResolveBaseTensorScheme(BaselineQuants baseQuant)
    {
        if (baseQuant.Names.IsDefaultOrEmpty)
            return null;

        return TensorWeightScheme.All.FirstOrDefault(s =>
            !s.Names.IsDefaultOrEmpty &&
            s.Names.Any(sn => baseQuant.Names.Contains(sn, StringComparer.OrdinalIgnoreCase)));
    }

    private List<RequestedTensorOverride> BuildRequestedTensorOverrides(HybridQuant quant)
    {
        var result = new List<RequestedTensorOverride>();

        if (quant.Tensors == null || quant.Tensors.Count == 0)
            return result;

        var baseScheme = TryResolveBaseTensorScheme(quant.BaseQuant);

        foreach (var hybrid in quant.Tensors)
        {
            if (hybrid?.TGroup == null)
                continue;

            if (hybrid.TensorType.UniqueId == TensorWeightScheme.NULL.UniqueId)
                continue;

            // Do not emit a redundant override if this tensor type is already the same
            // as the blanket base quant.
            if (baseScheme != null && hybrid.TensorType.UniqueId == baseScheme.UniqueId)
                continue;

            string schemeName = ResolveSchemeName(hybrid.TensorType);

            result.Add(new RequestedTensorOverride
            {
                GroupName = hybrid.TGroup.Name,
                SchemeName = schemeName,
                Patterns = hybrid.TGroup.Tensors.ToList()
            });
        }

        return result;
    }

    private async Task<List<ConcreteTensorOverride>> ResolveConcreteTensorOverridesAsync(
        string inputGgufPath,
        string outputFilePath,
        List<RequestedTensorOverride> requestedOverrides)
    {
        if (requestedOverrides.Count == 0)
            return new List<ConcreteTensorOverride>();

        string workingDir = Path.GetDirectoryName(outputFilePath)!;
        string unique = Guid.NewGuid().ToString("N");

        string payloadPath = Path.Combine(workingDir, $"resolve_tensor_overrides_{unique}.json");
        string resultPath = Path.Combine(workingDir, $"resolve_tensor_overrides_result_{unique}.json");
        string scriptPath = Path.Combine(workingDir, $"resolve_tensor_overrides_{unique}.py");

        try
        {
            var payload = new
            {
                gguf_path = inputGgufPath,
                output_path = resultPath,
                requests = requestedOverrides
            };

            await File.WriteAllTextAsync(
                payloadPath,
                JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));

            string py = """
                        import json
                        import re
                        import sys

                        payload_path = sys.argv[1]

                        def write_result(obj, output_path):
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(obj, f, indent=2)

                        with open(payload_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)

                        output_path = payload["output_path"]

                        try:
                            import gguf
                        except Exception as e:
                            write_result({"Error": f"Failed to import gguf: {e}"}, output_path)
                            sys.exit(0)

                        try:
                            reader = gguf.GGUFReader(payload["gguf_path"])
                        except Exception as e:
                            write_result({"Error": f"Failed to read GGUF: {e}"}, output_path)
                            sys.exit(0)

                        tensor_names = [t.name for t in reader.tensors]

                        resolved = []
                        group_counts = {}
                        unmatched = []
                        duplicates = []
                        seen = {}

                        for req in payload["requests"]:
                            group = req["GroupName"]
                            scheme = req["SchemeName"]
                            patterns = req["Patterns"]

                            compiled = [re.compile(p) for p in patterns]
                            matches = []

                            for name in tensor_names:
                                if any(r.fullmatch(name) for r in compiled):
                                    matches.append(name)

                            group_counts[group] = len(matches)

                            if len(matches) == 0:
                                unmatched.append(group)

                            for name in matches:
                                if name in seen and seen[name] != group:
                                    duplicates.append(name)
                                else:
                                    seen[name] = group

                                resolved.append({
                                    "TensorName": name,
                                    "SchemeName": scheme,
                                    "GroupName": group
                                })

                        write_result({
                            "Resolved": resolved,
                            "GroupMatchCounts": group_counts,
                            "UnmatchedGroups": unmatched,
                            "DuplicateTensors": sorted(set(duplicates))
                        }, output_path)
                        """;

            await File.WriteAllTextAsync(scriptPath, py);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            if (!File.Exists(resultPath))
                throw new InvalidOperationException("Tensor override resolution produced no result file.");

            var result = JsonSerializer.Deserialize<TensorResolutionResult>(
                await File.ReadAllTextAsync(resultPath));

            if (result == null)
                throw new InvalidOperationException("Tensor override resolution returned null.");

            if (!string.IsNullOrWhiteSpace(result.Error))
                throw new InvalidOperationException(result.Error);

            if (result.UnmatchedGroups.Count > 0)
            {
                throw new InvalidOperationException(
                    $"The following requested override groups matched zero tensors in the input GGUF: " +
                    $"{string.Join(", ", result.UnmatchedGroups)}");
            }

            if (result.DuplicateTensors.Count > 0)
            {
                throw new InvalidOperationException(
                    $"A tensor matched more than one override group, which is ambiguous: " +
                    $"{string.Join(", ", result.DuplicateTensors.Take(20))}");
            }

            return result.Resolved;
        }
        finally
        {
            if (File.Exists(payloadPath)) File.Delete(payloadPath);
            if (File.Exists(resultPath)) File.Delete(resultPath);
            if (File.Exists(scriptPath)) File.Delete(scriptPath);
        }
    }

    // ----------------------------------------------------------------
    // Internal DTOs
    // ----------------------------------------------------------------

    private sealed class RequestedTensorOverride
    {
        public string GroupName { get; set; } = string.Empty;
        public string SchemeName { get; set; } = string.Empty;
        public List<string> Patterns { get; set; } = new();
    }

    private sealed class ConcreteTensorOverride
    {
        public string TensorName { get; set; } = string.Empty;
        public string SchemeName { get; set; } = string.Empty;
        public string GroupName { get; set; } = string.Empty;
    }

    private sealed class TensorResolutionResult
    {
        public string? Error { get; set; }
        public List<ConcreteTensorOverride> Resolved { get; set; } = new();
        public Dictionary<string, int> GroupMatchCounts { get; set; } = new();
        public List<string> UnmatchedGroups { get; set; } = new();
        public List<string> DuplicateTensors { get; set; } = new();
    }

    // ----------------------------------------------------------------
    // Naming helpers
    // ----------------------------------------------------------------

    private static string ResolveBaseName(BaselineQuants b)
    {
        if (b.Names.IsDefaultOrEmpty)
            throw new InvalidOperationException($"BaselineQuants '{b.UniqueId}' has no Names.");

        return b.Names[0];
    }

    private static string ResolveSchemeName(TensorWeightScheme s)
    {
        if (s.Names.IsDefaultOrEmpty)
            throw new InvalidOperationException($"TensorWeightScheme '{s.UniqueId}' has no Names.");

        if (s.UniqueId == TensorWeightScheme.BF16_F16.UniqueId && s.Names.Length >= 2)
        {
            if (Cache.TorchType == Cache.MainTorchType.F16)
                return "F16";

            if (Cache.TorchType == Cache.MainTorchType.F32)
                return "F32";

            return "BF16";
        }

        return s.Names[0];
    }

    public string GenerateHybridName(HybridQuant quant)
    {
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        string baseName = ResolveBaseName(quant.BaseQuant);

        var effectiveTensors = quant.Tensors?
            .Where(t => t?.TGroup != null && t.TensorType.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .ToList();

        if (effectiveTensors == null || effectiveTensors.Count == 0)
            return $"{modelName}-{baseName}";

        var grouped = effectiveTensors
            .GroupBy(t => ResolveSchemeName(t.TensorType))
            .Select(g => new
            {
                Type = g.Key,
                Codes = g.Select(x => x.TGroup.ShortCode)
                    .OrderBy(c => GetOrder(c))
                    .ToArray()
            })
            .OrderBy(x => GetOrder(x.Codes.FirstOrDefault()))
            .ToList();

        var nameParts = new List<string>(grouped.Count);

        foreach (var group in grouped)
        {
            string codeStr = new string(group.Codes);
            string quantStr = SimplifyQuant(group.Type);
            nameParts.Add($"{codeStr}-{quantStr}");
        }

        string suffix = string.Join("-", nameParts);
        return $"{modelName}-{baseName}-{suffix}";
    }

    private int GetOrder(char c)
    {
        return "EHQKOUDXR".IndexOf(c);
    }

    private string SimplifyQuant(string quant)
    {
        return quant.Replace("_", "")
            .Replace("BF16", "B16")
            .Replace("F16", "F16")
            .Replace("F32", "F32");
    }

    private sealed class LoggedProcessResult
    {
        public int ExitCode { get; init; }
        public string StdOut { get; init; } = string.Empty;
        public string StdErr { get; init; } = string.Empty;
    }

    private async Task<LoggedProcessResult> RunLoggedProcessAsync(
        ProcessStartInfo psi,
        string? logPath,
        CancellationToken ct = default)
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

        StreamWriter? logWriter = null;
        FileStream? logStream = null;

        if (!string.IsNullOrWhiteSpace(logPath))
        {
            logStream = new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            logWriter = new StreamWriter(logStream) { AutoFlush = true };
        }

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

                logWriter?.WriteLine(line);
            }

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
                // ignored
            }
        });

        await process.WaitForExitAsync(ct);
        await Task.WhenAll(stdoutClosed.Task, stderrClosed.Task);

        logWriter?.Dispose();
        logStream?.Dispose();

        return new LoggedProcessResult
        {
            ExitCode = process.ExitCode,
            StdOut = stdoutBuilder.ToString(),
            StdErr = stderrBuilder.ToString()
        };
    }
}