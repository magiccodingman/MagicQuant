using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Blake3;


namespace MagicQuant.Helpers;

public static class MagicQuantModelId
{
    private const string IdFileName = "MagicQuant.id.json";
    private const string IdPrefix = "mq-blake3:";

    private sealed class ModelIdWrapper
    {
        public string Id { get; set; } = default!;
    }

    public static string GetOrCreateModelId(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
            throw new ArgumentException("Model directory path is null or empty.", nameof(modelDirectory));

        if (!Directory.Exists(modelDirectory))
            throw new DirectoryNotFoundException($"Directory does not exist: {modelDirectory}");

        string idFilePath = Path.Combine(modelDirectory, IdFileName);

        // 🚀 Fast path: cached ID exists
        if (File.Exists(idFilePath))
        {
            var wrapper = JsonSerializer.Deserialize<ModelIdWrapper>(File.ReadAllText(idFilePath));
            if (string.IsNullOrWhiteSpace(wrapper?.Id))
                throw new InvalidDataException($"{IdFileName} exists but is invalid (missing Id).");

            return wrapper.Id;
        }

        // 🔍 Find safetensors
        var safetensors = Directory
            .EnumerateFiles(modelDirectory, "*.safetensors", SearchOption.TopDirectoryOnly)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (safetensors.Length == 0)
            throw new InvalidOperationException($"No .safetensors files found in directory: {modelDirectory}");

        // 🧠 Hash tensor payloads deterministically
        using var hasher = Hasher.New();

        foreach (var file in safetensors)
            HashSafetensorPayload(file, hasher);

        // ✅ Blake3.NET: Finalize() returns Blake3.Hash which stringifies to hex
        var hash = hasher.Finalize();
        string finalHashHex = hash.ToString(); // hex digest (lowercase)
        string finalId = IdPrefix + finalHashHex;

        // 💾 Persist ID
        var output = new ModelIdWrapper { Id = finalId };

        File.WriteAllText(
            idFilePath,
            JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }),
            Encoding.UTF8
        );

        return finalId;
    }

    /// <summary>
    /// Hashes ONLY the tensor payload of a safetensors file (skips header + JSON metadata).
    /// Safetensors format: [u64 header_len][header_json_bytes][tensor_bytes...]
    /// </summary>
    private static void HashSafetensorPayload(string filePath, Hasher hasher)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // UInt64 metadata length (little endian)
        ulong metadataLength = reader.ReadUInt64();

        long tensorDataOffset = 8L + checked((long)metadataLength);

        if (tensorDataOffset >= stream.Length)
            throw new InvalidDataException($"Safetensors file is malformed (bad header length): {filePath}");

        stream.Position = tensorDataOffset;

        byte[] buffer = new byte[1024 * 1024]; // 1MB
        int bytesRead;

        while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
            hasher.Update(buffer.AsSpan(0, bytesRead));
    }
}