using System.Text.Json;
using MagicQuant.Helpers;

namespace MagicQuant.Services;

internal sealed class GgufMetadataReader
{
    private readonly PythonManager _python;

    public GgufMetadataReader(PythonManager python)
    {
        _python = python ?? throw new ArgumentNullException(nameof(python));
    }

    public async Task<GgufTensorReadResult> ReadAsync(
        string ggufPath,
        string workingDirectory,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(ggufPath) || !File.Exists(ggufPath))
            throw new FileNotFoundException("GGUF metadata source was not found.", ggufPath);

        Directory.CreateDirectory(workingDirectory);

        string unique = Guid.NewGuid().ToString("N");
        string payloadPath = Path.Combine(workingDirectory, $"read_gguf_tensors_{unique}.json");
        string resultPath = Path.Combine(workingDirectory, $"read_gguf_tensors_result_{unique}.json");
        string scriptPath = Path.Combine(workingDirectory, $"read_gguf_tensors_{unique}.py");

        try
        {
            await File.WriteAllTextAsync(
                payloadPath,
                JsonSerializer.Serialize(new { gguf_path = ggufPath, output_path = resultPath }),
                ct);

            const string py = """
                              import json
                              import sys

                              payload_path = sys.argv[1]
                              with open(payload_path, "r", encoding="utf-8") as f:
                                  payload = json.load(f)

                              output_path = payload["output_path"]

                              def resolve_type_name(t):
                                  for attr in ["type_name", "tensor_type", "type"]:
                                      v = getattr(t, attr, None)
                                      if v is None:
                                          continue
                                      if hasattr(v, "name"):
                                          return str(v.name)
                                      return str(v)
                                  return "UNKNOWN"

                              try:
                                  import gguf
                                  reader = gguf.GGUFReader(payload["gguf_path"])
                                  tensor_names = [t.name for t in reader.tensors]
                                  tensor_types = {t.name: resolve_type_name(t) for t in reader.tensors}

                                  def read_scalar(key):
                                      field = reader.fields.get(key)
                                      if field is None:
                                          return None
                                      value = field.contents()
                                      return value.item() if hasattr(value, "item") else value

                                  architecture = read_scalar("general.architecture")
                                  architecture_key = str(architecture) if architecture is not None else None
                                  block_count = read_scalar(f"{architecture_key}.block_count") if architecture_key else None
                                  nextn_layers = read_scalar(f"{architecture_key}.nextn_predict_layers") if architecture_key else None
                                  result = {
                                      "Architecture": architecture_key,
                                      "BlockCount": int(block_count) if block_count is not None else None,
                                      "NextnPredictLayers": int(nextn_layers) if nextn_layers is not None else None,
                                      "TensorNames": tensor_names,
                                      "TensorTypes": tensor_types
                                  }
                              except Exception as e:
                                  result = {"Error": str(e), "TensorNames": [], "TensorTypes": {}}

                              with open(output_path, "w", encoding="utf-8") as f:
                                  json.dump(result, f, indent=2)
                              """;

            await File.WriteAllTextAsync(scriptPath, py, ct);
            await _python.RunPythonScriptAsync(scriptPath, [payloadPath], ct: ct);

            var result = JsonSerializer.Deserialize<GgufTensorReadResult>(
                await File.ReadAllTextAsync(resultPath, ct));

            if (result == null)
                throw new InvalidOperationException("Failed to parse GGUF metadata result.");
            if (!string.IsNullOrWhiteSpace(result.Error))
                throw new InvalidOperationException($"Failed to read GGUF metadata: {result.Error}");

            return result;
        }
        finally
        {
            TryDelete(payloadPath);
            TryDelete(resultPath);
            TryDelete(scriptPath);
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
            // Best-effort cleanup; metadata success/failure is the authoritative result.
        }
    }
}
