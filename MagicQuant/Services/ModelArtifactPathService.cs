using System.Text;
using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Services;

public sealed class ModelArtifactPathService
{
    public string ModelDirectory => Cache.ModelDirectory
        ?? throw new InvalidOperationException("Cache.ModelDirectory is not set.");

    public string ModelMagicQuantDirectory => Cache.ModelMagicQuantDirectory
        ?? throw new InvalidOperationException("Cache.ModelMagicQuantDirectory is not set.");

    public string GgufDir => Path.Combine(ModelMagicQuantDirectory, "GGUF");
    public string BenchDir => Path.Combine(ModelMagicQuantDirectory, "Benchmarks");
    public string LogsDir => Path.Combine(ModelMagicQuantDirectory, "Logs");
    public string QuantizationLogsDir => Path.Combine(LogsDir, "Quantization");
    public string ExternalBaselinesDir => Cache.ExternalBaselineCacheDirectory
        ?? Path.Combine(ModelMagicQuantDirectory, "ExternalBaselines");

    public string ScratchModelNamespace
    {
        get
        {
            var modelName = new DirectoryInfo(ModelDirectory).Name;
            if (string.IsNullOrWhiteSpace(modelName))
                modelName = "model";

            var modelId = string.IsNullOrWhiteSpace(Cache.CurrentModelId) ? "unknown" : Cache.CurrentModelId;
            return MakeSafeFileComponent($"{modelName}_{modelId}");
        }
    }

    public string GetBenchmarkDir(string modelName) => Path.Combine(BenchDir, modelName);

    public string GetBaseLogitsDirectory()
    {
        string typeStr = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        return Path.Combine(BenchDir, typeStr, "logits");
    }

    public string GetNativeBaseGgufPath()
    {
        string modelName = new DirectoryInfo(ModelDirectory).Name;
        var torchType = Cache.TorchType ?? Cache.MainTorchType.BF16;
        string typeStr = torchType.ToString();
        return Path.Combine(GgufDir, $"{modelName}-{typeStr}.gguf");
    }

    public string GetExternalBaselineDurablePath(BaselineQuants baseline)
    {
        string safe = MakeSafeFileComponent(baseline.CanonicalKey);
        string extension = Path.GetExtension(baseline.SourceFileName ?? string.Empty);
        if (string.IsNullOrWhiteSpace(extension))
            extension = ".gguf";

        return Path.Combine(ExternalBaselinesDir, safe + extension);
    }

    public string GetQuantizationLogPath(string artifactName, Guid leaseId)
    {
        string safeName = MakeSafeFileComponent(artifactName);
        return Path.Combine(QuantizationLogsDir, $"{safeName}-{leaseId:N}.quantize.log");
    }

    public static string MakeSafeFileComponent(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "artifact";

        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var sb = new StringBuilder(value.Length);
        foreach (var ch in value)
            sb.Append(invalid.Contains(ch) ? '_' : ch);

        return sb.ToString();
    }
}
