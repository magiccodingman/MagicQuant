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

        // 1. Run Llama-Bench
        AnsiConsole.MarkupLine("[yellow]Running Llama-Bench...[/]");
        result.LlamaBench = await RunLlamaBenchAsync(modelPath, benchDir, startNgl);

        // 2. Run Perplexity (General, Code, Math)
        var domains = new[] { "general", "code", "math" };
        var corporaRoot = Path.Combine(Path.GetDirectoryName(benchDir)!, "_ppl_corpora");
        Directory.CreateDirectory(corporaRoot);

        if (saveLogits && !string.IsNullOrEmpty(klLogitsDir))
            Directory.CreateDirectory(klLogitsDir);

        foreach (var domain in domains)
        {
            AnsiConsole.MarkupLine($"[yellow]Running Perplexity ({domain})...[/]");
            
            // A. Prepare Corpus
            string corpusPath = Path.Combine(corporaRoot, $"ppl_corpus_{domain}.txt");
            await PreparePplCorpusAsync(domain, corpusPath, tokenTarget);

            // B. Run Benchmark
            var metrics = await RunPplBenchmarkAsync(
                modelPath, benchDir, domain, corpusPath, 
                startNgl, klLogitsDir, saveLogits
            );

            result.Perplexity[domain] = metrics;
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
        string BuildCmd(int ngl) => 
            $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {ngl} -o md";

        // Retry Loop
        int? finalNgl = await RunWithRetryAsync(BuildCmd, logFile, candidates, "llama-bench");

        // Fallback to CPU if GPU failed completely
        if (finalNgl == null)
        {
            AnsiConsole.MarkupLine("[red]GPU Failed. Fallback to CPU backend...[/]");
            string cpuCmd = $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -backend cpu -o md";
            await RunShellCommandAsync(cpuCmd, logFile); // Just run, no parsing check here usually
        }

        return ParseLlamaBench(logFile);
    }

    private LlamaBenchMetrics ParseLlamaBench(string logPath)
    {
        var metrics = new LlamaBenchMetrics { LogPath = GetRelativePath(logPath) };
        if (!File.Exists(logPath)) return metrics;

        var lines = File.ReadAllLines(logPath);
        
        // Find header row starting with | model | ... | backend |
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

        // Parse Header and Data Row
        var headers = lines[headerIdx].Split('|', StringSplitOptions.RemoveEmptyEntries).Select(h => h.Trim()).ToList();
        var dataRow = lines[headerIdx + 2].Split('|', StringSplitOptions.RemoveEmptyEntries).Select(d => d.Trim()).ToList();

        if (headers.Count != dataRow.Count) return metrics;

        var row = headers.Zip(dataRow, (h, d) => new { Header = h, Data = d }).ToDictionary(x => x.Header, x => x.Data);

        // Extract TPS
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
                kldArgs = $"--kl-divergence-base \"{logitsFile}\""; // Save to this file
            else if (File.Exists(logitsFile))
                kldArgs = $"--kl-divergence-base \"{logitsFile}\" --kl-divergence"; // Load from file
        }

        string BuildCmd(int ngl) => 
            $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {ngl} -c 2048 --file \"{corpusPath}\" {kldArgs}";

        await RunWithRetryAsync(BuildCmd, logFile, candidates, $"perplexity-{domain}");

        // Parsing
        bool expectKld = (!saveLogits && !string.IsNullOrEmpty(klLogitsDir));
        return ParsePerplexity(logFile, expectKld);
    }

    private PplMetrics ParsePerplexity(string logPath, bool allowMissingKld)
    {
        var metrics = new PplMetrics { LogPath = GetRelativePath(logPath) };
        if (!File.Exists(logPath)) return metrics;

        string text = File.ReadAllText(logPath);
        string cleanText = StripAnsi(text);

        // Regex for PPL: "Mean PPL(Q) : 8.88 +/- 0.20" OR "PPL = 8.88 +/- 0.20"
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

        // Regex for KLD: "Mean KLD : 0.0008" OR "KL divergence: 0.1234"
        var kldMatch = Regex.Match(cleanText, @"(?:Mean\s+KLD|KL[-_\s]*divergence|kl[-_\s]*div)\s*[:=]\s*([0-9.]+)", RegexOptions.IgnoreCase);
        
        if (kldMatch.Success)
        {
            metrics.Kld = double.Parse(kldMatch.Groups[1].Value);
        }
        else if (!allowMissingKld && cleanText.Contains("KL", StringComparison.OrdinalIgnoreCase))
        {
             // Warn if expected but not found
             AnsiConsole.MarkupLine("[yellow]Warning: 'KL' found in log but regex failed to parse value.[/]");
        }

        return metrics;
    }

    // ----------------------------------------------------------------
    // 3. Corpus Preparation (Using Python Interop)
    // ----------------------------------------------------------------

    private async Task PreparePplCorpusAsync(string domain, string outPath, int tokenTarget)
    {
        if (File.Exists(outPath) && new FileInfo(outPath).Length > 0) return;

        AnsiConsole.MarkupLine($"[grey]Generating corpus for domain: {domain}[/]");

        // We use the PythonManager to run a script that uses 'datasets' library
        // This mirrors your 'prepare_ppl_corpus' python function
        // We pass the python code as a string to the python environment
        
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
        // Create a temp python file to run this script safely
        string scriptPath = Path.Combine(Path.GetDirectoryName(outPath)!, $"gen_{domain}.py");
        await File.WriteAllTextAsync(scriptPath, pyScript);

        // Run it via PythonManager
        string pythonExe = _pyManager.GetPythonExecutable();
        string args = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) 
            ? $"/c \"{pythonExe}\" \"{scriptPath}\"" 
            : $"\"{scriptPath}\"";
        
        string runner = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cmd.exe" : pythonExe;
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) args = scriptPath; // Correct linux arg

        // We need 'datasets' installed
        await _pyManager.RunPipInstallAsync("datasets");
        
        await RunShellCommandAsync(runner + " " + args, null); // Run the generation script
        
        // Cleanup script
        if(File.Exists(scriptPath)) File.Delete(scriptPath);
    }

    // ----------------------------------------------------------------
    // 4. Helper: Retry Logic (OOM Handling)
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
            
            // Check for OOM
            if (OomMarkers.Any(m => logContent.Contains(m, StringComparison.OrdinalIgnoreCase)))
            {
                AnsiConsole.MarkupLine($"[yellow][WARN] {label}: OOM at -ngl {ngl}, retrying...[/]");
                continue;
            }

            // Simple check: if log is empty or super short, it crashed non-OOM
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
        
        // If logPath is provided, we stream output to it
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
        // Ideally make this relative to the project root, but for now returning filename is safer for display
        return Path.GetFileName(fullPath); 
    }
}