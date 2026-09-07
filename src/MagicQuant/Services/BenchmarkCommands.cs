using System.Globalization;
using MagicQuant.Runtime;

namespace MagicQuant.Services;

/// <summary>Native benchmark arguments, separated from scheduling and measured-truth persistence.</summary>
internal static class BenchmarkCommands
{
    public static NativeCommand Bench(string executable, string model, bool gpu, int ngl, string tensorSplit)
    {
        List<string> args = ["-m", model, "-p", "8", "-t", "16"];
        if (gpu) args.AddRange(["-ngl", ngl.ToString(CultureInfo.InvariantCulture), .. SplitArgs(tensorSplit)]);
        else args.AddRange(["-ngl", "0"]);
        args.AddRange(["-o", "md"]);
        return new NativeCommand(executable, args);
    }

    public static NativeCommand Perplexity(string executable, string model, string corpus, bool gpu,
        int ngl, string tensorSplit, string? logitsFile = null, bool compareLogits = false)
    {
        List<string> args = ["-m", model, "-ngl", (gpu ? ngl : 0).ToString(CultureInfo.InvariantCulture)];
        if (gpu) args.AddRange(SplitArgs(tensorSplit));
        args.AddRange(["-t", "4", "-c", "2048", "--file", corpus]);
        if (logitsFile != null)
        {
            args.AddRange(["--kl-divergence-base", logitsFile]);
            if (compareLogits) args.Add("--kl-divergence");
        }
        return new NativeCommand(executable, args);
    }

    // This fragment is emitted only by LlamaGpuArgumentBuilder (flag + numeric vector).
    private static string[] SplitArgs(string tensorSplit) => tensorSplit.Split(' ', StringSplitOptions.RemoveEmptyEntries);
}
