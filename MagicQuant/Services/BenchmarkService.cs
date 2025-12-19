using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public class BenchmarkService
{
    private readonly LlamaBinaries _bins;
    private readonly PythonManager _pyManager;

    // Constants
    private static readonly string[] OomMarkers = 
    {
        "out of memory", "cudaMalloc failed", "unable to allocate cuda", "try reducing --n-gpu-layers"
    };

    private static readonly int[] NglCandidates = { 35, 30, 24, 20, 16, 12, 8, 4, 0 };

    // ----------------------------------------------------------------
    // Concurrency Controls
    // ----------------------------------------------------------------
    
    // 1. Exclusive Lock: When LlamaBench runs, it must be the ONLY thing running.
    // Higher-level logic should acquire this before calling RunLlamaBenchAsync.
    public static readonly SemaphoreSlim ExclusiveBenchLock = new(1, 1);

    // 2. VRAM Lock: Only one VRAM-heavy task (Perplexity) can run at a time.
    // However, it CAN run alongside CPU tasks (like quantization if VRAM allows).
    public static readonly SemaphoreSlim VramLock = new(1, 1);

    public BenchmarkService(string llamaRoot, PythonManager pyManager)
    {
        _bins = new LlamaBinaries(llamaRoot);
        _bins.Validate();
        _pyManager = pyManager;
    }

    // ----------------------------------------------------------------
    // Public Entry Point
    // ----------------------------------------------------------------

    public async Task<BenchmarkResult> RunAllBenchmarksAsync(
        string modelPath,
        string benchDir,
        int tokenTarget = 32768,
        int? startNgl = null,
        string? klLogitsDir = null,
        bool saveLogits = false)
    {
        Directory.CreateDirectory(benchDir);
        var result = new BenchmarkResult();

        // 1. Run Llama-Bench (Exclusive Mode)
        // We acquire the exclusive lock to ensure stability
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

        // 2. Run Perplexity (General, Code, Math)
        // We prepare folders first so we don't block locks unnecessarily
        var domains = new[] { "general", "code", "math" };
        var corporaRoot = Path.Combine(Path.GetDirectoryName(benchDir)!, "_ppl_corpora");
        Directory.CreateDirectory(corporaRoot);

        if (saveLogits && !string.IsNullOrEmpty(klLogitsDir))
            Directory.CreateDirectory(klLogitsDir);

        foreach (var domain in domains)
        {
            // A. Prepare Corpus (CPU bound, low risk)
            string corpusPath = Path.Combine(corporaRoot, $"ppl_corpus_{domain}.txt");
            await PreparePplCorpusAsync(domain, corpusPath, tokenTarget);

            // B. Run Benchmark (VRAM Intensive)
            // We acquire VRAM lock so we don't run 2 perplexities at once
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

        // Save Results JSON
        string jsonPath = Path.Combine(benchDir, "bench_metrics.json");
        await File.WriteAllTextAsync(jsonPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
        
        return result;
    }

    // ----------------------------------------------------------------
    // 1. Llama-Bench Logic
    // ----------------------------------------------------------------

    private async Task<LlamaBenchMetrics> RunLlamaBenchAsync(string modelPath, string benchDir, int? startNgl)
    {
        string logFile = Path.Combine(benchDir, "llamabench.md");
        
        // Filter candidates
        var candidates = startNgl.HasValue 
            ? NglCandidates.Where(n => n <= startNgl.Value).ToList() 
            : NglCandidates.ToList();

        // Command Builder
        // Note: Keeping -p 8 -t 16 as requested ("just like we're now")
        string BuildCmd(int ngl) => 
            $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {ngl} -o md";

        // Retry Loop
        int? finalNgl = await RunWithRetryAsync(BuildCmd, logFile, candidates, "llama-bench");

        // Fallback to CPU if GPU failed completely
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
    // 2. Perplexity Logic
    // ----------------------------------------------------------------

    private async Task<PplMetrics> RunPplBenchmarkAsync(
        string modelPath, string benchDir, string domain, string corpusPath, 
        int? startNgl, string? klLogitsDir, bool saveLogits)
    {
        string logFile = Path.Combine(benchDir, $"perplexity_{domain}.log");
        var candidates = startNgl.HasValue 
            ? NglCandidates.Where(n => n <= startNgl.Value).ToList() 
            : NglCandidates.ToList();

        // KL Divergence Logic
        string kldArgs = "";
        if (!string.IsNullOrEmpty(klLogitsDir))
        {
            string logitsFile = Path.Combine(klLogitsDir, $"kld_logits_{domain}.bin");
            if (saveLogits)
                kldArgs = $"--kl-divergence-base \"{logitsFile}\""; 
            else if (File.Exists(logitsFile))
                kldArgs = $"--kl-divergence-base \"{logitsFile}\" --kl-divergence"; 
        }

        // Command Builder
        // Added "-t 4" to limit thread usage as requested
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
    // 3. Corpus Preparation
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
        
        if(File.Exists(scriptPath)) File.Delete(scriptPath);
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
            AnsiConsole.MarkupLine($"[grey][*] {label}: trying -ngl {ngl}[/]");

            await RunShellCommandAsync(cmd, logPath);

            string logContent = File.Exists(logPath) ? File.ReadAllText(logPath) : "";
            
            if (OomMarkers.Any(m => logContent.Contains(m, StringComparison.OrdinalIgnoreCase)))
            {
                AnsiConsole.MarkupLine($"[yellow][WARN] {label}: OOM at -ngl {ngl}, retrying...[/]");
                continue;
            }

            if (logContent.Length < 50) 
            {
                AnsiConsole.MarkupLine($"[yellow][WARN] {label}: Failed at -ngl {ngl} (Unknown Error), trying next...[/]");
                continue;
            }

            AnsiConsole.MarkupLine($"[green][OK] {label}: succeeded with -ngl {ngl}[/]");
            return ngl;
        }

        AnsiConsole.MarkupLine($"[red][!] {label}: All -ngl candidates failed.[/]");
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