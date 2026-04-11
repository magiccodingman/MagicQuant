using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using MagicQuant.Helpers;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels; // Required for BenchmarkCategory and TensorCombo
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public class BenchmarkService
{
    private readonly LlamaBinaries _bins;
    public readonly PythonManager _pyManager;

    private static readonly string[] BaseDomains = { "general", "code", "math" };
    private static readonly string[] SampleDomains = { "general" };

    private static bool IsNativeBaseModel(HybridQuant quantConfig)
    {
        return quantConfig.BaseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId;
    }

    private static IReadOnlyCollection<string> ResolveRequestedDomains(
        HybridQuant quantConfig,
        IReadOnlyCollection<string>? domainsOverride)
    {
        if (domainsOverride != null && domainsOverride.Count > 0)
        {
            return domainsOverride
                .Select(x => x.Trim().ToLowerInvariant())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        return IsNativeBaseModel(quantConfig) ? BaseDomains : SampleDomains;
    }

    private static bool RequiresKld(HybridQuant quantConfig)
    {
        return !IsNativeBaseModel(quantConfig);
    }

    private static ulong TryGetModelSize(string modelPath)
    {
        return File.Exists(modelPath) ? (ulong)new FileInfo(modelPath).Length : 0UL;
    }

    private static bool IsPositiveKld(double? kld)
    {
        return kld.HasValue && kld.Value > 0d;
    }

    public async Task<bool> TryReuseExistingBenchmarksAsync(
        HybridQuant quantConfig,
        string modelPath,
        string benchDir,
        string? klLogitsDir,
        IReadOnlyCollection<string>? domainsOverride = null)
    {
        var requestedDomains = ResolveRequestedDomains(quantConfig, domainsOverride);
        bool requireKld = RequiresKld(quantConfig);

        if (!TryReadExistingBenchmarkArtifacts(
                benchDir: benchDir,
                requestedDomains: requestedDomains,
                requireKld: requireKld,
                result: out var reused))
        {
            return false;
        }

        reused.ModelSizeBytes ??= TryGetModelSize(modelPath);

        using var db = new MagicQuantContext();

        var currentHashStr = Cache.CurrentModelId;
        var aiModelHash = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == currentHashStr);

        if (aiModelHash == null)
        {
            aiModelHash = new AiModelHash { UniqueHash = currentHashStr };
            db.AiModelHashes.Add(aiModelHash);
            await db.SaveChangesAsync();
        }

        var tensorCombo = await GetOrCreateTensorComboAsync(db, quantConfig);

        await SaveBenchmarkToDbAsync(db, aiModelHash, tensorCombo, reused, modelPath);
        await WriteMetricsJsonAsync(benchDir, reused);

        return true;
    }

    // Constants
    private static readonly string[] OomMarkers =
    {
        "out of memory", "cudaMalloc failed", "unable to allocate cuda", "try reducing --n-gpu-layers"
    };

    private static readonly int[] NglCandidates = { 35, 30, 24, 20, 16, 12, 8, 4, 0 };

    // ----------------------------------------------------------------
    // Concurrency Controls
    // ----------------------------------------------------------------

    public static readonly SemaphoreSlim ExclusiveBenchLock = new(1, 1);
    public static readonly SemaphoreSlim VramLock = new(1, 1);

    public BenchmarkService(PythonManager pyManager)
    {
        _bins = new LlamaBinaries(Cache.LlamaRoot);
        _bins.Validate();
        _pyManager = pyManager;
    }

    // ----------------------------------------------------------------
    // Public Entry Point
    // ----------------------------------------------------------------

    public async Task<BenchmarkResult> RunAllBenchmarksAsync(
        HybridQuant quantConfig,
        string modelPath,
        string benchDir,
        int tokenTarget = 32768,
        int? startNgl = null,
        string? klLogitsDir = null,
        bool saveLogits = false,
        IReadOnlyCollection<string>? domainsOverride = null)
    {
        Directory.CreateDirectory(benchDir);

        var requestedDomains = ResolveRequestedDomains(quantConfig, domainsOverride);
        bool requireKld = RequiresKld(quantConfig);

        using var db = new MagicQuantContext();

        var currentHashStr = Cache.CurrentModelId;
        var aiModelHash = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == currentHashStr);

        if (aiModelHash == null)
        {
            aiModelHash = new AiModelHash { UniqueHash = currentHashStr };
            db.AiModelHashes.Add(aiModelHash);
            await db.SaveChangesAsync();
        }

        var tensorCombo = await GetOrCreateTensorComboAsync(db, quantConfig);

        var existingBench = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .AsNoTracking()
            .FirstOrDefaultAsync(b => b.AiModelHashId == aiModelHash.Id && b.TensorComboId == tensorCombo.Id);

        // 1. If DB already has everything required, trust DB first and don't rerun.
        if (existingBench != null && HasRequiredCategories(existingBench, requestedDomains, requireKld))
        {
            if (TryReadExistingBenchmarkArtifacts(benchDir, requestedDomains, requireKld, out var diskResult))
            {
                diskResult.ModelSizeBytes ??= existingBench.SizeBytes;
                return diskResult;
            }

            return BuildResultFromDb(existingBench, requestedDomains);
        }

        // 2. If disk already has reusable artifacts, sync DB and return.
        if (TryReadExistingBenchmarkArtifacts(benchDir, requestedDomains, requireKld, out var reused))
        {
            reused.ModelSizeBytes ??= TryGetModelSize(modelPath);
            await SaveBenchmarkToDbAsync(db, aiModelHash, tensorCombo, reused, modelPath);
            await WriteMetricsJsonAsync(benchDir, reused);
            return reused;
        }

        // 3. Otherwise run only the pieces that are actually missing/invalid.
        var result = new BenchmarkResult
        {
            ModelSizeBytes = TryGetModelSize(modelPath)
        };

        string llamaBenchPath = Path.Combine(benchDir, "llamabench.md");
        if (TryReadExistingLlamaBenchLog(llamaBenchPath, out var existingLlamaBench))
        {
            result.LlamaBench = existingLlamaBench;
        }
        else
        {
            await ExclusiveBenchLock.WaitAsync();
            try
            {
                AnsiConsole.MarkupLine("[yellow]Running Llama-Bench (Exclusive Mode)...[/]");
                result.LlamaBench = await RunLlamaBenchAsync(modelPath, benchDir, startNgl);
            }
            finally
            {
                ExclusiveBenchLock.Release();
            }
        }

        var corporaRoot = Path.Combine(Path.GetDirectoryName(benchDir)!, "_ppl_corpora");
        Directory.CreateDirectory(corporaRoot);

        if (saveLogits && !string.IsNullOrEmpty(klLogitsDir))
            Directory.CreateDirectory(klLogitsDir);

        foreach (var domain in requestedDomains)
        {
            if (TryReadExistingPplLog(
                    benchDir: benchDir,
                    domain: domain,
                    allowMissingKld: !requireKld,
                    requirePositiveKld: requireKld,
                    metrics: out var existingPpl))
            {
                result.Perplexity[domain] = existingPpl;
                continue;
            }

            string corpusPath = Path.Combine(corporaRoot, $"ppl_corpus_{domain}.txt");
            await PreparePplCorpusAsync(domain, corpusPath, tokenTarget);

            await VramLock.WaitAsync();
            try
            {
                AnsiConsole.MarkupLine($"[yellow]Running Perplexity ({domain})...[/]");
                var metrics = await RunPplBenchmarkAsync(
                    modelPath: modelPath,
                    benchDir: benchDir,
                    domain: domain,
                    corpusPath: corpusPath,
                    startNgl: startNgl,
                    klLogitsDir: klLogitsDir,
                    saveLogits: saveLogits);

                if (requireKld && !IsPositiveKld(metrics.Kld))
                {
                    throw new InvalidOperationException(
                        $"Non-base benchmark produced invalid KLD for domain '{domain}'. " +
                        $"KLD must exist and be > 0. Parsed value: {(metrics.Kld.HasValue ? metrics.Kld.Value.ToString() : "null")}");
                }

                result.Perplexity[domain] = metrics;
            }
            finally
            {
                VramLock.Release();
            }
        }

        await WriteMetricsJsonAsync(benchDir, result);
        await SaveBenchmarkToDbAsync(db, aiModelHash, tensorCombo, result, modelPath);

        return result;
    }

    // ----------------------------------------------------------------
    // Database Helpers (Fixed for Immutability)
    // ----------------------------------------------------------------

    private async Task<TensorCombo> GetOrCreateTensorComboAsync(MagicQuantContext db, HybridQuant quant)
    {
        // 1. Extract values into local variables. 
        //    Default to 0 (NULL scheme) if not present in the mutable list.
        byte baseQuant = quant.BaseQuant.UniqueId;

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
            foreach (var t in quant.Tensors)
            {
                // Compare using UniqueId to be safe
                if (t.TGroup.UniqueId == TReg.Embeddings.UniqueId) embeddings = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.LmHead.UniqueId) lmHead = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.AttnQ.UniqueId) attnQ = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.AttnKV.UniqueId) attnKV = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.AttnOutput.UniqueId) attnOutput = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.FfnUpGate.UniqueId) ffnUpGate = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.FfnDown.UniqueId) ffnDown = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.MoeExperts.UniqueId) moeExperts = t.TensorType.UniqueId;
                else if (t.TGroup.UniqueId == TReg.MoeRouter.UniqueId) moeRouter = t.TensorType.UniqueId;
            }
        }

        // 2. Create the Immutable Config using the Constructor
        var c = new TensorConfig(
            baseQuant,
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

        // 3. Check DB using the extracted values
        //    (We query by the raw bytes because the DB entity fields are readonly and might not map directly in Expression trees depending on EF version)
        var existing = await db.TensorCombos.FirstOrDefaultAsync(x =>
            x.BaseQuant == c.BaseQuant &&
            x.Embeddings == c.Embeddings &&
            x.LmHead == c.LmHead &&
            x.AttnQ == c.AttnQ &&
            x.AttnKV == c.AttnKV &&
            x.AttnOutput == c.AttnOutput &&
            x.FfnUpGate == c.FfnUpGate &&
            x.FfnDown == c.FfnDown &&
            x.MoeExperts == c.MoeExperts &&
            x.MoeRouter == c.MoeRouter
        );

        if (existing != null) return existing;

        // 4. Create New TensorCombo using the Constructor (which accepts TensorConfig)
        var newCombo = new TensorCombo(c);
        db.TensorCombos.Add(newCombo);
        await db.SaveChangesAsync();
        return newCombo;
    }

    private async Task SaveBenchmarkToDbAsync(
        MagicQuantContext db,
        AiModelHash model,
        TensorCombo combo,
        BenchmarkResult res,
        string modelPath)
    {
        using var transaction = await db.Database.BeginTransactionAsync();

        try
        {
            bool isBaseModel =
                combo.BaseQuant == BaselineQuants.NativeSourceUniqueId &&
                combo.Embeddings == 0 &&
                combo.LmHead == 0 &&
                combo.AttnQ == 0 &&
                combo.AttnKV == 0 &&
                combo.AttnOutput == 0 &&
                combo.FfnUpGate == 0 &&
                combo.FfnDown == 0 &&
                combo.MoeExperts == 0 &&
                combo.MoeRouter == 0;

            ulong sizeBytes =
                res.ModelSizeBytes.GetValueOrDefault() > 0
                    ? res.ModelSizeBytes!.Value
                    : (File.Exists(modelPath) ? (ulong)new FileInfo(modelPath).Length : 0UL);

            var bench = await db.AiBenchmarks
                .Include(x => x.CategorBenchmarks)
                .FirstOrDefaultAsync(x =>
                    x.AiModelHashId == model.Id &&
                    x.TensorComboId == combo.Id);

            if (bench == null)
            {
                bench = new AiBenchmark
                {
                    AiModelHashId = model.Id,
                    TensorComboId = combo.Id
                };

                db.AiBenchmarks.Add(bench);
            }

            bench.TokensPerSecond = res.LlamaBench?.Tps ?? 0;
            bench.Ngl = (byte)(res.LlamaBench?.Ngl ?? 0);
            bench.SizeBytes = sizeBytes;

            await db.SaveChangesAsync();

            if (bench.CategorBenchmarks != null && bench.CategorBenchmarks.Count > 0)
            {
                db.Set<CategoryBenchmark>().RemoveRange(bench.CategorBenchmarks);
                await db.SaveChangesAsync();
            }

            var categories = new List<CategoryBenchmark>();

            foreach (var kvp in res.Perplexity)
            {
                string domain = kvp.Key.ToLowerInvariant();
                var m = kvp.Value;

                byte category = domain switch
                {
                    "general" => (byte)BenchmarkCategory.General,
                    "math" => (byte)BenchmarkCategory.Math,
                    "code" => (byte)BenchmarkCategory.Code,
                    _ => throw new InvalidOperationException($"Unknown benchmark domain '{domain}'.")
                };

                double kld;
                if (isBaseModel)
                {
                    kld = 0d;
                }
                else
                {
                    if (!IsPositiveKld(m.Kld))
                    {
                        throw new InvalidOperationException(
                            $"Refusing to save non-base benchmark with invalid KLD. Domain='{domain}', KLD='{m.Kld?.ToString() ?? "null"}'");
                    }

                    kld = m.Kld!.Value;
                }

                categories.Add(new CategoryBenchmark
                {
                    AiBenchmarkId = bench.Id,
                    Category = category,
                    Ppl = m.Ppl,
                    PplError = m.PplError,
                    Kld = kld
                });
            }

            if (categories.Count > 0)
            {
                db.Set<CategoryBenchmark>().AddRange(categories);
                await db.SaveChangesAsync();
            }

            await transaction.CommitAsync();
        }
        catch (Exception ex)
        {
            await transaction.RollbackAsync();

            var inner = ex.InnerException?.Message;
            if (!string.IsNullOrWhiteSpace(inner))
            {
                AnsiConsole.MarkupLine($"[red]Failed to save benchmarks to DB:[/] {Markup.Escape(ex.Message)}");
                AnsiConsole.MarkupLine($"[red]Inner Exception:[/] {Markup.Escape(inner)}");
            }
            else
            {
                AnsiConsole.MarkupLine($"[red]Failed to save benchmarks to DB:[/] {Markup.Escape(ex.Message)}");
            }

            throw;
        }
    }

    private async Task WriteMetricsJsonAsync(string benchDir, BenchmarkResult result)
    {
        Directory.CreateDirectory(benchDir);

        string jsonPath = Path.Combine(benchDir, "bench_metrics.json");
        await File.WriteAllTextAsync(
            jsonPath,
            JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
    }

    private bool TryReadExistingBenchmarkArtifacts(
        string benchDir,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld,
        out BenchmarkResult result)
    {
        result = new BenchmarkResult();

        string jsonPath = Path.Combine(benchDir, "bench_metrics.json");
        if (File.Exists(jsonPath))
        {
            try
            {
                var parsed = JsonSerializer.Deserialize<BenchmarkResult>(File.ReadAllText(jsonPath));
                if (parsed != null && IsReusableBenchmarkResult(parsed, requestedDomains, requireKld))
                {
                    result = parsed;
                    return true;
                }
            }
            catch
            {
                // fall through and try rebuilding from individual logs
            }
        }

        string llamaBenchPath = Path.Combine(benchDir, "llamabench.md");
        if (!TryReadExistingLlamaBenchLog(llamaBenchPath, out var llamaBench))
            return false;

        var rebuilt = new BenchmarkResult
        {
            LlamaBench = llamaBench
        };

        foreach (var domain in requestedDomains)
        {
            if (!TryReadExistingPplLog(
                    benchDir: benchDir,
                    domain: domain,
                    allowMissingKld: !requireKld,
                    requirePositiveKld: requireKld,
                    metrics: out var ppl))
            {
                return false;
            }

            rebuilt.Perplexity[domain] = ppl;
        }

        result = rebuilt;
        return true;
    }

    private bool IsReusableBenchmarkResult(
        BenchmarkResult result,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld)
    {
        if (result.LlamaBench == null || !result.LlamaBench.Tps.HasValue || result.LlamaBench.Tps.Value <= 0)
            return false;

        foreach (var domain in requestedDomains)
        {
            if (!result.Perplexity.TryGetValue(domain, out var ppl))
                return false;

            if (ppl.Ppl <= 0 || ppl.PplError < 0)
                return false;

            if (requireKld && !IsPositiveKld(ppl.Kld))
                return false;
        }

        return true;
    }

    private bool TryReadExistingLlamaBenchLog(string logPath, out LlamaBenchMetrics metrics)
    {
        metrics = null!;

        if (!File.Exists(logPath) || new FileInfo(logPath).Length == 0)
            return false;

        try
        {
            var parsed = ParseLlamaBench(logPath);
            if (parsed.Tps.HasValue && parsed.Tps.Value > 0)
            {
                metrics = parsed;
                return true;
            }
        }
        catch
        {
            // ignore and return false
        }

        return false;
    }

    private bool TryReadExistingPplLog(
        string benchDir,
        string domain,
        bool allowMissingKld,
        bool requirePositiveKld,
        out PplMetrics metrics)
    {
        metrics = null!;

        string logPath = Path.Combine(benchDir, $"perplexity_{domain}.log");
        if (!File.Exists(logPath) || new FileInfo(logPath).Length == 0)
            return false;

        try
        {
            var parsed = ParsePerplexity(logPath, allowMissingKld);

            if (parsed.Ppl <= 0)
                return false;

            if (requirePositiveKld && !IsPositiveKld(parsed.Kld))
                return false;

            metrics = parsed;
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool HasRequiredCategories(
        AiBenchmark bench,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld)
    {
        if (bench.CategorBenchmarks == null || bench.CategorBenchmarks.Count == 0)
            return false;

        foreach (var domain in requestedDomains)
        {
            byte category = domain switch
            {
                "general" => (byte)BenchmarkCategory.General,
                "math" => (byte)BenchmarkCategory.Math,
                "code" => (byte)BenchmarkCategory.Code,
                _ => throw new InvalidOperationException($"Unknown benchmark domain '{domain}'.")
            };

            var existing = bench.CategorBenchmarks.FirstOrDefault(x => x.Category == category);
            if (existing == null)
                return false;

            if (existing.Ppl <= 0)
                return false;

            if (requireKld && existing.Kld <= 0)
                return false;
        }

        return bench.TokensPerSecond > 0;
    }

    private BenchmarkResult BuildResultFromDb(
        AiBenchmark bench,
        IReadOnlyCollection<string> requestedDomains)
    {
        var result = new BenchmarkResult
        {
            ModelSizeBytes = bench.SizeBytes,
            LlamaBench = new LlamaBenchMetrics
            {
                Ngl = bench.Ngl,
                Tps = bench.TokensPerSecond
            }
        };

        foreach (var domain in requestedDomains)
        {
            byte category = domain switch
            {
                "general" => (byte)BenchmarkCategory.General,
                "math" => (byte)BenchmarkCategory.Math,
                "code" => (byte)BenchmarkCategory.Code,
                _ => throw new InvalidOperationException($"Unknown benchmark domain '{domain}'.")
            };

            var existing = bench.CategorBenchmarks.First(x => x.Category == category);

            result.Perplexity[domain] = new PplMetrics
            {
                Ppl = existing.Ppl,
                PplError = existing.PplError,
                Kld = existing.Kld
            };
        }

        return result;
    }

    // ----------------------------------------------------------------
    // 1. Llama-Bench Logic (Unchanged)
    // ----------------------------------------------------------------

    private async Task<LlamaBenchMetrics> RunLlamaBenchAsync(string modelPath, string benchDir, int? startNgl)
    {
        string logFile = Path.Combine(benchDir, "llamabench.md");
        var candidates = startNgl.HasValue
            ? NglCandidates.Where(n => n <= startNgl.Value).ToList()
            : NglCandidates.ToList();

        string BuildCmd(int ngl) =>
            $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {ngl} -o md";

        int? finalNgl = await RunWithRetryAsync(BuildCmd, logFile, candidates, "llama-bench");

        if (finalNgl == null)
        {
            AnsiConsole.MarkupLine("[red]GPU Failed. Fallback to CPU backend...[/]");
            string cpuCmd = $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -backend cpu -o md";
            await RunShellCommandAsync(cpuCmd, logFile);
        }

        return ParseLlamaBench(logFile);
    }

    private LlamaBenchMetrics ParseLlamaBench(string logPath)
    {
        var metrics = new LlamaBenchMetrics { LogPath = GetRelativePath(logPath) };
        if (!File.Exists(logPath)) return metrics;

        var lines = File.ReadAllLines(logPath);
        int headerIdx = -1;
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].Contains("|") && lines[i].Contains("backend"))
            {
                headerIdx = i;
                break;
            }
        }

        if (headerIdx == -1 || lines.Length <= headerIdx + 2) return metrics;

        var headers = lines[headerIdx].Split('|', StringSplitOptions.RemoveEmptyEntries).Select(h => h.Trim()).ToList();
        var dataRow = lines[headerIdx + 2].Split('|', StringSplitOptions.RemoveEmptyEntries).Select(d => d.Trim())
            .ToList();

        if (headers.Count != dataRow.Count) return metrics;
        var row = headers.Zip(dataRow, (h, d) => new { Header = h, Data = d }).ToDictionary(x => x.Header, x => x.Data);

        string tpsStr = row.ContainsKey("t/s") ? row["t/s"] : (row.ContainsKey("tps") ? row["tps"] : "0");
        var match = Regex.Match(tpsStr, @"([0-9.]+)");

        if (match.Success && double.TryParse(match.Groups[1].Value, out double tps))
        {
            metrics.Tps = tps;
            metrics.Backend = row.ContainsKey("backend") ? row["backend"] : "unknown";
            metrics.Test = row.ContainsKey("test") ? row["test"] : "unknown";
            if (row.ContainsKey("ngl") && int.TryParse(row["ngl"], out int ngl)) metrics.Ngl = ngl;
        }

        return metrics;
    }

    // ----------------------------------------------------------------
    // 2. Perplexity Logic (Unchanged)
    // ----------------------------------------------------------------

    private async Task<PplMetrics> RunPplBenchmarkAsync(
        string modelPath,
        string benchDir,
        string domain,
        string corpusPath,
        int? startNgl,
        string? klLogitsDir,
        bool saveLogits)
    {
        string logFile = Path.Combine(benchDir, $"perplexity_{domain}.log");
        var candidates = startNgl.HasValue
            ? NglCandidates.Where(n => n <= startNgl.Value).ToList()
            : NglCandidates.ToList();

        string kldArgs = "";
        bool expectKld = false;

        if (!string.IsNullOrEmpty(klLogitsDir))
        {
            string logitsFile = Path.Combine(klLogitsDir, $"kld_logits_{domain}.bin");

            if (saveLogits)
            {
                // Base/native model path: save logits only, do not expect KLD yet.
                kldArgs = $"--kl-divergence-base \"{logitsFile}\"";
                expectKld = false;
            }
            else if (File.Exists(logitsFile))
            {
                // Sample path: compare against the already saved base logits.
                kldArgs = $"--kl-divergence-base \"{logitsFile}\" --kl-divergence";
                expectKld = true;
            }
        }

        string BuildCmd(int ngl) =>
            $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {ngl} -t 4 -c 2048 --file \"{corpusPath}\" {kldArgs}";

        await RunWithRetryAsync(BuildCmd, logFile, candidates, $"perplexity-{domain}");

        bool allowMissingKld = !expectKld;
        var parsed = ParsePerplexity(logFile, allowMissingKld);

        if (expectKld && !IsPositiveKld(parsed.Kld))
        {
            throw new InvalidOperationException(
                $"Expected a real KLD for domain '{domain}', but parsed '{parsed.Kld?.ToString() ?? "null"}' from {logFile}");
        }

        return parsed;
    }

    private PplMetrics ParsePerplexity(string logPath, bool allowMissingKld)
    {
        var metrics = new PplMetrics { LogPath = GetRelativePath(logPath) };

        if (!File.Exists(logPath))
            throw new FileNotFoundException($"Perplexity log file was not created: {logPath}");

        string text = File.ReadAllText(logPath);
        string cleanText = StripAnsi(text);

        var pplMatch = Regex.Match(
            cleanText,
            @"(?:Mean PPL\(Q\)|PPL)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:±|\+/-)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            RegexOptions.IgnoreCase);

        if (!pplMatch.Success)
        {
            throw new InvalidOperationException(
                $"Failed to parse PPL from log: {logPath}\n\nLast log content:\n{cleanText}");
        }

        metrics.Ppl = double.Parse(pplMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        metrics.PplError = double.Parse(pplMatch.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);

        var kldMatch = Regex.Match(
            cleanText,
            @"(?:Mean\s+KLD|Mean\s+KL|KL[-_\s]*divergence|KLD|kl[-_\s]*div)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            RegexOptions.IgnoreCase);

        if (kldMatch.Success)
        {
            metrics.Kld = double.Parse(kldMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        }
        else if (!allowMissingKld)
        {
            throw new InvalidOperationException(
                $"KLD was expected but could not be parsed from log: {logPath}\n\nLast log content:\n{cleanText}");
        }

        return metrics;
    }

    // ----------------------------------------------------------------
    // 3. Corpus Preparation (Unchanged)
    // ----------------------------------------------------------------

    private async Task PreparePplCorpusAsync(string domain, string outPath, int tokenTarget)
    {
        if (File.Exists(outPath) && new FileInfo(outPath).Length > 0) return;

        AnsiConsole.MarkupLine($"[grey]Generating corpus for domain: {domain}[/]");

        string pyScript = $@"
import sys
from datasets import load_dataset

domain = '{domain}'
out_path = r'{outPath}'
max_chars = {tokenTarget} * 4

def get_sources(d):
    if d == 'general': return [('wikitext', 'wikitext-103-raw-v1', 'test', 'text'), ('wikitext', 'wikitext-2-raw-v1', 'test', 'text')]
    if d == 'code': return [('codeparrot/codeparrot-clean', None, 'train', 'content')]
    if d == 'math': return [('gsm8k', 'main', 'test', 'question')]
    return []

parts = []
total = 0
for ds, conf, split, field in get_sources(domain):
    try:
        d = load_dataset(ds, conf) if conf else load_dataset(ds)
        for text in d[split][field]:
            if not text or not isinstance(text, str): continue
            chunk = text.strip() + '\n'
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars: break
    except Exception as e:
        print(f'Error loading {{ds}}: {{e}}')
    if total >= max_chars: break

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(''.join(parts))
";
        string scriptPath = Path.Combine(Path.GetDirectoryName(outPath)!, $"gen_{domain}.py");
        await File.WriteAllTextAsync(scriptPath, pyScript);

        string pythonExe = _pyManager.GetPythonExecutable();
        string args = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? $"/c \"{pythonExe}\" \"{scriptPath}\""
            : $"\"{scriptPath}\"";

        string runner = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cmd.exe" : pythonExe;
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) args = scriptPath;

        await _pyManager.RunPipInstallAsync("datasets");
        await RunShellCommandAsync(runner + " " + args, null);

        if (File.Exists(scriptPath)) File.Delete(scriptPath);
    }

    // ----------------------------------------------------------------
    // 4. Retry Logic
    // ----------------------------------------------------------------
    private async Task<int?> RunWithRetryAsync(
        Func<int, string> cmdBuilder,
        string logPath,
        List<int> candidates,
        string label)
    {
        string? lastFailureDetails = null;

        foreach (int ngl in candidates)
        {
            string cmd = cmdBuilder(ngl);
            AnsiConsole.WriteLine($"[*] {label}: trying -ngl {ngl}");

            var result = await RunShellCommandAsync(cmd, logPath);

            string logContent = !string.IsNullOrWhiteSpace(result.LogOutput)
                ? result.LogOutput
                : (File.Exists(logPath) ? File.ReadAllText(logPath) : string.Empty);

            bool looksLikeOom = OomMarkers.Any(m =>
                logContent.Contains(m, StringComparison.OrdinalIgnoreCase));

            bool looksLikeLoadFailure =
                logContent.Contains("failed to load model", StringComparison.OrdinalIgnoreCase) ||
                logContent.Contains("error:", StringComparison.OrdinalIgnoreCase);

            if (!result.Success)
            {
                lastFailureDetails =
                    $"ExitCode={result.ExitCode}, ngl={ngl}\nCommand: {cmd}\n\nLog Output:\n{logContent}";

                if (looksLikeOom || looksLikeLoadFailure)
                {
                    AnsiConsole.WriteLine($"[WARN] {label}: failed at -ngl {ngl}, retrying lower setting...");
                    continue;
                }

                // Unknown non-zero exit: still retry lower ngl first,
                // because many llama.cpp GPU/load issues recover that way.
                AnsiConsole.WriteLine($"[WARN] {label}: non-zero exit at -ngl {ngl}, retrying lower setting...");
                continue;
            }

            if (logContent.Length < 50)
            {
                lastFailureDetails =
                    $"Log too short at ngl={ngl}\nCommand: {cmd}\n\nLog Output:\n{logContent}";
                AnsiConsole.WriteLine($"[WARN] {label}: Failed at -ngl {ngl} (log too short), trying next...");
                continue;
            }

            if (label.StartsWith("perplexity", StringComparison.OrdinalIgnoreCase) &&
                !Regex.IsMatch(logContent, @"PPL\s*[:=]\s*[-+]?\d*\.?\d+", RegexOptions.IgnoreCase))
            {
                lastFailureDetails =
                    $"No parsable PPL marker found at ngl={ngl}\nCommand: {cmd}\n\nLog Output:\n{logContent}";
                AnsiConsole.WriteLine($"[WARN] {label}: No parsable PPL marker found at -ngl {ngl}, trying next...");
                continue;
            }

            AnsiConsole.WriteLine($"[OK] {label}: succeeded with -ngl {ngl}");
            return ngl;
        }

        throw new InvalidOperationException(
            $"{label}: all -ngl candidates failed.\n\nLast failure details:\n{lastFailureDetails}");
    }

    // ----------------------------------------------------------------
    // 5. System Utilities
    // ----------------------------------------------------------------

    private sealed class CommandRunResult
    {
        public bool Success { get; init; }
        public int ExitCode { get; init; }
        public string LogOutput { get; init; } = string.Empty;
    }

    private async Task<CommandRunResult> RunShellCommandAsync(string cmd, string? logPath)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cmd.exe" : "/bin/bash",
            Arguments = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? $"/c {cmd}" : $"-c \"{cmd}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = startInfo };
        FileStream? fs = null;
        StreamWriter? sw = null;

        if (logPath != null)
        {
            fs = new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            sw = new StreamWriter(fs) { AutoFlush = true };
        }

        process.Start();

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        if (!string.IsNullOrWhiteSpace(stdout))
            sw?.WriteLine(stdout);

        if (!string.IsNullOrWhiteSpace(stderr))
            sw?.WriteLine(stderr);

        sw?.Dispose();
        fs?.Dispose();

        string combinedLog;
        if (logPath != null && File.Exists(logPath))
            combinedLog = File.ReadAllText(logPath);
        else
            combinedLog = $"{stdout}\n{stderr}";

        return new CommandRunResult
        {
            Success = process.ExitCode == 0,
            ExitCode = process.ExitCode,
            LogOutput = combinedLog
        };
    }

    private string StripAnsi(string text)
    {
        return Regex.Replace(text, @"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "");
    }

    private string GetRelativePath(string fullPath)
    {
        return Path.GetFileName(fullPath);
    }
}