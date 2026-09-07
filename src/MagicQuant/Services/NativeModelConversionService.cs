using MagicQuant.Helpers;
using MagicQuant.Runtime;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

/// <summary>
/// Owns native GGUF conversion and its success-marker lifecycle. It has no benchmark
/// or learned-truth dependency, so conversion can be exercised independently.
/// </summary>
public sealed class NativeModelConversionService(ModelArtifactPathService paths, PythonManager python, IProcessRunner? runner = null)
{
    private readonly ModelArtifactPathService _paths = paths;
    private readonly PythonManager _python = python;
    private readonly IProcessRunner _runner = runner ?? new ProcessRunner();
    private static readonly SemaphoreSlim BaseModelLock = new(1, 1);

    public async Task<string> EnsureAsync(bool deleteProcess = false)
    {
        await BaseModelLock.WaitAsync(MagicQuant.Runtime.RunCancellation.Token);
        try
        {
            string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
            var torchType = Cache.TorchType ?? Cache.MainTorchType.BF16;
            string typeStr = torchType.ToString();

            string fileName = $"{modelName}-{typeStr}.gguf";
            string outputPath = Path.Combine(_paths.GgufDir, fileName);
            string successFile = Path.Combine(_paths.GgufDir, $"{fileName}.success.json");
            string convertLogPath = outputPath + ".convert.log";

            if (deleteProcess)
            {
                if (!Directory.Exists(_paths.GgufDir))
                    Directory.CreateDirectory(_paths.GgufDir);

                var normalizedFileName = Path.GetFileName(fileName);
                var successFileName = normalizedFileName + ".success.json";
                var successFilePath = Path.Combine(_paths.GgufDir, successFileName);
                bool isImmune = File.Exists(successFilePath);

                foreach (var filePath in Directory.EnumerateFiles(_paths.GgufDir, "*.gguf", SearchOption.TopDirectoryOnly))
                {
                    var currentFileName = Path.GetFileName(filePath);
                    var currentModelName = Path.GetFileNameWithoutExtension(currentFileName);

                    if (isImmune &&
                        string.Equals(currentFileName, normalizedFileName, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(currentModelName) && IsProtectedModel(currentModelName))
                    {
                        continue;
                    }

                    await HardDeleteHelper.DeleteFileIfExistsAsync(filePath);
                }
            }

            if (!File.Exists(outputPath) || new FileInfo(outputPath).Length == 0 || !File.Exists(successFile))
            {
                AnsiConsole.MarkupLine($"[bold cyan]Converting to {Markup.Escape(typeStr)}...[/]");

                await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                string convertScript = Cache.ConvertScript
                                       ?? throw new Exception("ConvertScript path missing in Cache");

                string outTypeArg = typeStr.ToLowerInvariant();

                var psi = new MagicQuant.Runtime.NativeCommand(_python.GetPythonExecutable(),
                    [convertScript, Cache.ModelDirectory!, "--outtype", outTypeArg, "--outfile", outputPath]).CreateStartInfo();
                psi.WorkingDirectory = Cache.LlamaRoot;

                MagicQuant.Runtime.ProcessResult result;
                try
                {
                    result = await _runner.RunAsync(psi, convertLogPath, (line, _) => { if (Cache.VerboseProcessOutput) AnsiConsole.WriteLine(line); }, RunCancellation.Token);
                }
                catch
                {
                    // No success marker may survive an interrupted/failed conversion.
                    await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);
                    await HardDeleteHelper.DeleteFileIfExistsAsync(successFile);
                    throw;
                }

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


    private static bool IsProtectedModel(string name)
    {
        return name.EndsWith("BF16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F32", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("Q8_0", StringComparison.OrdinalIgnoreCase);
    }

}
