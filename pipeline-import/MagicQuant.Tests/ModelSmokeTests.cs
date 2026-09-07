using System.Diagnostics;
using System.Text.Json;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MagicQuant.Runtime;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public sealed class ModelSmokeFactAttribute : FactAttribute
{
    public ModelSmokeFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("MQ_RUN_MODEL_SMOKE") != "1")
            Skip = "Opt in with MQ_RUN_MODEL_SMOKE=1 and the documented model/toolchain paths.";
    }
}

/// <summary>Small-model integration, isolated from ordinary PR checks and existing campaign state.</summary>
public sealed class ModelSmokeTests
{
    [ModelSmokeFact]
    [Trait("Category", "ModelSmoke")]
    public async Task Convert_quantize_read_metadata_benchmark_and_reuse_native_artifact()
    {
        string Required(string key) => Environment.GetEnvironmentVariable(key)
            ?? throw new InvalidOperationException($"Set {key}; see docs/testing.md.");
        string source = Path.GetFullPath(Required("MQ_SMOKE_MODEL"));
        string llama = Path.GetFullPath(Required("MQ_SMOKE_LLAMA_ROOT"));
        string runtime = Path.GetFullPath(Required("MQ_SMOKE_RUNTIME_ROOT"));
        string output = Path.GetFullPath(Required("MQ_SMOKE_OUTPUT"));
        string root = Path.Combine(output, $"smoke-{Guid.NewGuid():N}");
        string model = Path.Combine(root, "model with spaces");
        Assert.True(Directory.Exists(source));
        Directory.CreateDirectory(model);
        // Inputs are copied/linked into a new directory; no source-model files are modified.
        foreach (string file in Directory.EnumerateFiles(source))
        {
            string destination = Path.Combine(model, Path.GetFileName(file));
            if (file.EndsWith(".safetensors", StringComparison.Ordinal)) File.CreateSymbolicLink(destination, file);
            else File.Copy(file, destination);
        }

        var oldConfig = Config.Current;
        var oldPaths = (Cache.ModelDirectory, Cache.ModelMagicQuantDirectory, Cache.MagicQuantDirectory,
            Cache.LlamaRoot, Cache.LlamaBin, Cache.ConvertScript, Cache.CurrentModelId);
        var oldPrecision = Cache.TorchType;
        var oldScratch = Cache.ScratchRoots;
        var oldImatrix = (Cache.UseImatrix, Cache.IsImatrixAvailable);
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(20));
            using var scope = RunCancellation.Use(timeout.Token);
            Config.Load(MagicQuantYamlConfig.CreateDefault());
            Cache.ModelDirectory = model;
            Cache.ModelMagicQuantDirectory = Path.Combine(model, "MagicQuant");
            Cache.MagicQuantDirectory = runtime;
            Cache.LlamaRoot = llama;
            Cache.LlamaBin = Path.Combine(llama, "build", "bin");
            Cache.ConvertScript = Path.Combine(llama, "convert_hf_to_gguf.py");
            Cache.CurrentModelId = "isolated-smoke";
            Cache.ScratchRoots = [root];
            Cache.UseImatrix = false;
            Cache.IsImatrixAvailable = false;
            JsonHelper.DetectAndSetTorchType(model);
            var python = new PythonManager(runtime);
            var quantizer = new QuantizationService(new BenchmarkService(python));
            string native = await quantizer.EnsureBaseModelFileAsync();
            DateTime nativeTimestamp = File.GetLastWriteTimeUtc(native);
            Assert.Equal(native, await quantizer.EnsureBaseModelFileAsync());
            Assert.Equal(nativeTimestamp, File.GetLastWriteTimeUtc(native));

            string export = Path.Combine(root, "export");
            Directory.CreateDirectory(export);
            string q8 = Path.Combine(export, "smoke Q8_0.gguf");
            string scratch;
            await using (var lease = await quantizer.BuildPureQ8ProbeLeaseAsync(timeout.Token))
            {
                scratch = lease.GgufPath;
                Assert.True(new FileInfo(scratch).Length > 0);

            }
            Assert.False(File.Exists(scratch));
            await quantizer.BuildExportArtifactAsync(HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0), q8, forceRebuild: true, ct: timeout.Token);
            DateTime exportTimestamp = File.GetLastWriteTimeUtc(q8);
            await quantizer.BuildExportArtifactAsync(HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0), q8, ct: timeout.Token);
            Assert.Equal(exportTimestamp, File.GetLastWriteTimeUtc(q8));
            var reader = new GgufMetadataReader(python);
            var nativeMetadata = await reader.ReadAsync(native, root, timeout.Token);
            var q8Metadata = await reader.ReadAsync(q8, root, timeout.Token);
            Assert.NotEmpty(nativeMetadata.TensorNames);
            Assert.Equal(nativeMetadata.TensorNames.OrderBy(x => x), q8Metadata.TensorNames.OrderBy(x => x));
            string benchLog = Path.Combine(export, "llamabench.md");
            var command = BenchmarkCommands.Bench(Path.Combine(Cache.LlamaBin, OperatingSystem.IsWindows() ? "llama-bench.exe" : "llama-bench"), q8, false, 0, "");
            var result = await new ProcessRunner().RunAsync(command.CreateStartInfo(), benchLog, ct: timeout.Token);
            Assert.True(result.Success, result.CombinedOutput);
            var metrics = BenchmarkLogParser.ParseLlamaBench(benchLog);
            Assert.True(metrics.Tps > 0);
            string manifest = MagicQuantManifestPathService.GetManifestFilePath(export, "smoke.tensor-map.json");
            await File.WriteAllTextAsync(manifest, JsonSerializer.Serialize(q8Metadata.TensorTypes));
            await File.WriteAllTextAsync(Path.Combine(root, "smoke-result.json"), JsonSerializer.Serialize(new
            {
                SourceModel = source,
                LlamaRoot = llama,
                RuntimeRoot = runtime,
                NativeBytes = new FileInfo(native).Length,
                Q8Bytes = new FileInfo(q8).Length,
                TensorCount = q8Metadata.TensorNames.Count,
                metrics.Tps,
                NativeReuseVerified = true,
                ScratchCleanupVerified = true,
                CompletedUtc = DateTimeOffset.UtcNow
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        finally
        {
            Config.Load(oldConfig);
            (Cache.ModelDirectory, Cache.ModelMagicQuantDirectory, Cache.MagicQuantDirectory,
                Cache.LlamaRoot, Cache.LlamaBin, Cache.ConvertScript, Cache.CurrentModelId) = oldPaths;
            Cache.TorchType = oldPrecision;
            Cache.ScratchRoots = oldScratch;
            (Cache.UseImatrix, Cache.IsImatrixAvailable) = oldImatrix;
            // Keep only logs, metadata and the result report. Always remove heavy test weights.
            foreach (string file in Directory.EnumerateFiles(root, "*.gguf", SearchOption.AllDirectories)) File.Delete(file);
            foreach (string file in Directory.EnumerateFiles(model, "*.safetensors")) File.Delete(file);
        }
    }
}
