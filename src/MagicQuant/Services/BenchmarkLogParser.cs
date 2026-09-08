using System.Globalization;
using System.Text.RegularExpressions;
using MQ.DB.Models;

namespace MagicQuant.Services;

/// <summary>Parses llama.cpp logs without scheduling work or opening the benchmark database.</summary>
public static class BenchmarkLogParser
{
    public static LlamaBenchMetrics ParseLlamaBench(string logPath)
    {
        var metrics = new LlamaBenchMetrics { LogPath = Path.GetFileName(logPath) };
        if (!File.Exists(logPath))
            return metrics;

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

        if (headerIdx == -1 || lines.Length <= headerIdx + 2)
            return metrics;

        var headers = lines[headerIdx]
            .Split('|', StringSplitOptions.RemoveEmptyEntries)
            .Select(h => h.Trim())
            .ToList();

        var dataRow = lines[headerIdx + 2]
            .Split('|', StringSplitOptions.RemoveEmptyEntries)
            .Select(d => d.Trim())
            .ToList();

        if (headers.Count != dataRow.Count)
            return metrics;

        var row = headers
            .Zip(dataRow, (h, d) => new { Header = h, Data = d })
            .ToDictionary(x => x.Header, x => x.Data, StringComparer.OrdinalIgnoreCase);

        string tpsStr = row.ContainsKey("t/s")
            ? row["t/s"]
            : (row.ContainsKey("tps") ? row["tps"] : "0");

        var match = Regex.Match(tpsStr, @"([0-9.]+)");
        if (match.Success &&
            double.TryParse(match.Groups[1].Value, NumberStyles.Any, CultureInfo.InvariantCulture, out double tps))
        {
            metrics.Tps = tps;
            metrics.Backend = row.ContainsKey("backend") ? row["backend"] : "unknown";
            metrics.Test = row.ContainsKey("test") ? row["test"] : "unknown";

            if (row.ContainsKey("ngl") &&
                int.TryParse(row["ngl"], NumberStyles.Any, CultureInfo.InvariantCulture, out int ngl))
            {
                metrics.Ngl = ngl;
            }
        }

        return metrics;
    }

    public static PplMetrics ParsePerplexity(string logPath, bool allowMissingKld)
    {
        var metrics = new PplMetrics { LogPath = Path.GetFileName(logPath) };

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

        metrics.Ppl = double.Parse(pplMatch.Groups[1].Value, CultureInfo.InvariantCulture);
        metrics.PplError = double.Parse(pplMatch.Groups[2].Value, CultureInfo.InvariantCulture);

        var kldMatch = Regex.Match(
            cleanText,
            @"(?:Mean\s+KLD|Mean\s+KL|KL[-_\s]*divergence|KLD|kl[-_\s]*div)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            RegexOptions.IgnoreCase);

        if (kldMatch.Success)
        {
            metrics.Kld = double.Parse(kldMatch.Groups[1].Value, CultureInfo.InvariantCulture);
        }
        else if (!allowMissingKld)
        {
            throw new InvalidOperationException(
                $"KLD was expected but could not be parsed from log: {logPath}\n\nLast log content:\n{cleanText}");
        }

        return metrics;
    }

    public static string StripAnsi(string text) => Regex.Replace(text, @"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "");
}
