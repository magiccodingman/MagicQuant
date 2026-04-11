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
        bool saveLogits = false)
    {
        Directory.CreateDirectory(benchDir);
        string jsonPath = Path.Combine(benchDir, "bench_metrics.json");

        using var db = new MagicQuantContext();
        
        // 1. Resolve Model Hash
        var currentHashStr = Cache.CurrentModelId;
        var aiModelHash = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == currentHashStr);

        if (aiModelHash == null)
        {
            aiModelHash = new AiModelHash { UniqueHash = currentHashStr };
            db.AiModelHashes.Add(aiModelHash);
            await db.SaveChangesAsync();
        }

        // 2. Resolve Tensor Combo (Fixed to use Constructor)
        var tensorCombo = await GetOrCreateTensorComboAsync(db, quantConfig);

        // 3. Check if Benchmark already exists
        var existingBench = await db.AiBenchmarks
            .FirstOrDefaultAsync(b => b.AiModelHashId == aiModelHash.Id && b.TensorComboId == tensorCombo.Id);

        if (existingBench != null)
        {
            AnsiConsole.MarkupLine($"[green]Benchmark found in database for Combo ID {tensorCombo.Id}. Skipping execution.[/]");
            
            if (File.Exists(jsonPath))
            {
                var cachedJson = await File.ReadAllTextAsync(jsonPath);
                if (!string.IsNullOrWhiteSpace(cachedJson))
                {
                    var deserializedMetrics = JsonSerializer.Deserialize<BenchmarkResult>(cachedJson);
                    if (deserializedMetrics != null) return deserializedMetrics;
                }
            }
            return new BenchmarkResult(); 
        }

        // ----------------------------------------------------------------
        // 4. Execution
        // ----------------------------------------------------------------

        var result = new BenchmarkResult();

        // A. Run Llama-Bench
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

        // B. Run Perplexity
        var domains = new[] { "general", "code", "math" };
        var corporaRoot = Path.Combine(Path.GetDirectoryName(benchDir)!, "_ppl_corpora");
        Directory.CreateDirectory(corporaRoot);

        if (saveLogits && !string.IsNullOrEmpty(klLogitsDir))
            Directory.CreateDirectory(klLogitsDir);

        foreach (var domain in domains)
        {
            string corpusPath = Path.Combine(corporaRoot, $"ppl_corpus_{domain}.txt");
            await PreparePplCorpusAsync(domain, corpusPath, tokenTarget);

            await VramLock.WaitAsync();
            try
            {
                AnsiConsole.MarkupLine($"[yellow]Running Perplexity ({domain})...[/]");
                var metrics = await RunPplBenchmarkAsync(
                    modelPath, benchDir, domain, corpusPath,
                    startNgl, klLogitsDir, saveLogits
                );
                result.Perplexity[domain] = metrics;
            }
            finally
            {
                VramLock.Release();
            }
        }

        // 5. Save Results
        await File.WriteAllTextAsync(jsonPath,
            JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));

        // Pass modelPath so we can calculate SizeBytes
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
            // Calculate File Size
            ulong sizeBytes = 0;
            if (File.Exists(modelPath))
            {
                sizeBytes = (ulong)new FileInfo(modelPath).Length;
            }

            // 1. Create Parent Benchmark
            var bench = new AiBenchmark
            {
                AiModelHashId = model.Id,
                TensorComboId = combo.Id,
                
                // Map LlamaBench fields
                TokensPerSecond = res.LlamaBench?.Tps ?? 0,
                Ngl = (byte)(res.LlamaBench?.Ngl ?? 0),
                SizeBytes = sizeBytes
            };

            db.AiBenchmarks.Add(bench);
            await db.SaveChangesAsync(); // Generates bench.Id

            // 2. Create Child Category Benchmarks
            var categories = new List<CategoryBenchmark>();

            // Helper to determine if we need to force 0.0 KLD for Base Model
            // (Checks if everything is 0 except BaseQuant which is BF16/F16)
            bool isBaseModel = combo.BaseQuant == TensorWeightScheme.BF16_F16.UniqueId &&
                               combo.Embeddings == 0 && combo.LmHead == 0;

            // Map "general" -> BenchmarkCategory.General (1)
            if (res.Perplexity.ContainsKey("general"))
            {
                var m = res.Perplexity["general"];
                var cb = new CategoryBenchmark
                {
                    AiBenchmarkId = bench.Id,
                    Category = (byte)BenchmarkCategory.General,
                    Ppl = m.Ppl,
                    PplError = m.PplError,
                    Kld = m.Kld ?? (isBaseModel ? 0.0 : 0.0) // Defaults to 0 if null
                };
                categories.Add(cb);
            }

            // Map "code" -> BenchmarkCategory.Code (3)
            if (res.Perplexity.ContainsKey("code"))
            {
                var m = res.Perplexity["code"];
                var cb = new CategoryBenchmark
                {
                    AiBenchmarkId = bench.Id,
                    Category = (byte)BenchmarkCategory.Code,
                    Ppl = m.Ppl,
                    PplError = m.PplError,
                    Kld = m.Kld ?? (isBaseModel ? 0.0 : 0.0)
                };
                categories.Add(cb);
            }

            // Map "math" -> BenchmarkCategory.Math (2)
            if (res.Perplexity.ContainsKey("math"))
            {
                var m = res.Perplexity["math"];
                var cb = new CategoryBenchmark
                {
                    AiBenchmarkId = bench.Id,
                    Category = (byte)BenchmarkCategory.Math,
                    Ppl = m.Ppl,
                    PplError = m.PplError,
                    Kld = m.Kld ?? (isBaseModel ? 0.0 : 0.0)
                };
                categories.Add(cb);
            }

            // Batch Insert Categories
            if (categories.Count > 0)
            {
                db.Set<CategoryBenchmark>().AddRange(categories);
                await db.SaveChangesAsync();
            }

            await transaction.CommitAsync();
            AnsiConsole.MarkupLine("[green]Benchmarks saved to Database successfully.[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Failed to save benchmarks to DB: {ex.Message}[/]");
            await transaction.RollbackAsync();
        }
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
        var dataRow = lines[headerIdx + 2].Split('|', StringSplitOptions.RemoveEmptyEntries).Select(d => d.Trim()).ToList();

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
        string modelPath, string benchDir, string domain, string corpusPath,
        int? startNgl, string? klLogitsDir, bool saveLogits)
    {
        string logFile = Path.Combine(benchDir, $"perplexity_{domain}.log");
        var candidates = startNgl.HasValue
            ? NglCandidates.Where(n => n <= startNgl.Value).ToList()
            : NglCandidates.ToList();

        string kldArgs = "";
        if (!string.IsNullOrEmpty(klLogitsDir))
        {
            string logitsFile = Path.Combine(klLogitsDir, $"kld_logits_{domain}.bin");
            if (saveLogits)
                kldArgs = $"--kl-divergence-base \"{logitsFile}\"";
            else if (File.Exists(logitsFile))
                kldArgs = $"--kl-divergence-base \"{logitsFile}\" --kl-divergence";
        }

        string BuildCmd(int ngl) =>
            $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {ngl} -t 4 -c 2048 --file \"{corpusPath}\" {kldArgs}";

        await RunWithRetryAsync(BuildCmd, logFile, candidates, $"perplexity-{domain}");

        bool expectKld = (!saveLogits && !string.IsNullOrEmpty(klLogitsDir));
        return ParsePerplexity(logFile, expectKld);
    }

    private PplMetrics ParsePerplexity(string logPath, bool allowMissingKld)
    {
        var metrics = new PplMetrics { LogPath = GetRelativePath(logPath) };
        if (!File.Exists(logPath)) return metrics;

        string text = File.ReadAllText(logPath);
        string cleanText = StripAnsi(text);

        var pplMatch = Regex.Match(cleanText, @"(?:Mean PPL\(Q\)|PPL)\s*[:=]\s*([0-9.]+)\s*(?:±|\+/-)\s*([0-9.]+)", RegexOptions.IgnoreCase);

        if (pplMatch.Success)
        {
            metrics.Ppl = double.Parse(pplMatch.Groups[1].Value);
            metrics.PplError = double.Parse(pplMatch.Groups[2].Value);
        }
        else
        {
            AnsiConsole.MarkupLine($"[red]Error parsing PPL from {logPath}[/]");
        }

        var kldMatch = Regex.Match(cleanText, @"(?:Mean\s+KLD|KL[-_\s]*divergence|kl[-_\s]*div)\s*[:=]\s*([0-9.]+)", RegexOptions.IgnoreCase);

        if (kldMatch.Success)
        {
            metrics.Kld = double.Parse(kldMatch.Groups[1].Value);
        }
        else if (!allowMissingKld && cleanText.Contains("KL", StringComparison.OrdinalIgnoreCase))
        {
            AnsiConsole.MarkupLine("[yellow]Warning: 'KL' found in log but regex failed to parse value.[/]");
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
        foreach (int ngl in candidates)
        {
            string cmd = cmdBuilder(ngl);
            AnsiConsole.WriteLine($"[*] {label}: trying -ngl {ngl}");

            await RunShellCommandAsync(cmd, logPath);

            string logContent = File.Exists(logPath)
                ? File.ReadAllText(logPath)
                : string.Empty;

            if (OomMarkers.Any(m => logContent.Contains(m, StringComparison.OrdinalIgnoreCase)))
            {
                AnsiConsole.WriteLine($"[WARN] {label}: OOM at -ngl {ngl}, retrying...");
                continue;
            }

            if (logContent.Length < 50)
            {
                AnsiConsole.WriteLine($"[WARN] {label}: Failed at -ngl {ngl} (Unknown Error), trying next...");
                continue;
            }

            AnsiConsole.WriteLine($"[OK] {label}: succeeded with -ngl {ngl}");
            return ngl;
        }

        AnsiConsole.WriteLine($"[ERROR] {label}: All -ngl candidates failed.");
        return null;
    }

    // ----------------------------------------------------------------
    // 5. System Utilities
    // ----------------------------------------------------------------

    private async Task RunShellCommandAsync(string cmd, string? logPath)
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
            sw = new StreamWriter(fs);
        }

        process.OutputDataReceived += (s, e) => { if (e.Data != null) sw?.WriteLine(e.Data); };
        process.ErrorDataReceived += (s, e) => { if (e.Data != null) sw?.WriteLine(e.Data); };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        await process.WaitForExitAsync();

        sw?.Dispose();
        fs?.Dispose();
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