using System.Globalization;

namespace MagicQuant.Services;

internal enum LlamaGpuTool
{
    CommonCli = 1,
    LlamaBench = 2
}

internal static class LlamaGpuArgumentBuilder
{
    public static string BuildTensorSplitArgs(
        IReadOnlyList<int> deviceIndices,
        IReadOnlyDictionary<int, double> gpuMemoryLimitsGb,
        LlamaGpuTool tool)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        ArgumentNullException.ThrowIfNull(gpuMemoryLimitsGb);

        if (deviceIndices.Count <= 1 || gpuMemoryLimitsGb.Count == 0)
            return string.Empty;

        var missing = deviceIndices
            .Where(i => !gpuMemoryLimitsGb.ContainsKey(i))
            .ToArray();

        if (missing.Length > 0)
        {
            throw new InvalidOperationException(
                $"GPU memory limits were configured, but limits are missing for GPU(s): {string.Join(", ", missing)}.");
        }

        // llama-bench uses commas to generate a Cartesian product of benchmark cases;
        // members of one multi-GPU split are slash-separated. The common llama.cpp
        // argument parser used by llama-perplexity expects comma-separated members.
        string separator = tool == LlamaGpuTool.LlamaBench ? "/" : ",";
        string split = string.Join(separator, deviceIndices.Select(i =>
            gpuMemoryLimitsGb[i].ToString("0.###", CultureInfo.InvariantCulture)));

        return $" --tensor-split {split}";
    }
}
