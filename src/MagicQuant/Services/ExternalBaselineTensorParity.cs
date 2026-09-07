namespace MagicQuant.Services;

internal sealed class GgufTensorReadResult
{
    public string? Error { get; set; }
    public string? Architecture { get; set; }
    public int? BlockCount { get; set; }
    public int? NextnPredictLayers { get; set; }
    public List<string> TensorNames { get; set; } = new();
    public Dictionary<string, string> TensorTypes { get; set; } = new(StringComparer.Ordinal);
}

internal sealed class ExternalBaselineTensorParityResult
{
    public required GgufTensorReadResult NativeMetadata { get; init; }
    public required GgufTensorReadResult ExternalMetadata { get; init; }
    public IReadOnlyList<string> InheritedOptionalTensorNames { get; init; } = [];
    public int OmittedNextnLayerCount { get; init; }
}

internal static class ExternalBaselineTensorParity
{
    public static ExternalBaselineTensorParityResult ValidateOrThrow(
        GgufTensorReadResult nativeMetadata,
        GgufTensorReadResult externalMetadata)
    {
        ArgumentNullException.ThrowIfNull(nativeMetadata);
        ArgumentNullException.ThrowIfNull(externalMetadata);

        var nativeNames = nativeMetadata.TensorNames
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();
        var externalNames = externalMetadata.TensorNames
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        var missing = nativeNames
            .Except(externalNames, StringComparer.Ordinal)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();
        var unexpected = externalNames
            .Except(nativeNames, StringComparer.Ordinal)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        if (missing.Count == 0 && unexpected.Count == 0 && nativeNames.Count == externalNames.Count)
        {
            return new ExternalBaselineTensorParityResult
            {
                NativeMetadata = nativeMetadata,
                ExternalMetadata = externalMetadata
            };
        }

        if (unexpected.Count == 0 &&
            TryValidateDeclaredOptionalNextnOmission(
                nativeMetadata,
                externalMetadata,
                missing,
                out int omittedLayerCount))
        {
            return new ExternalBaselineTensorParityResult
            {
                NativeMetadata = nativeMetadata,
                ExternalMetadata = externalMetadata,
                InheritedOptionalTensorNames = missing,
                OmittedNextnLayerCount = omittedLayerCount
            };
        }

        throw new InvalidOperationException(
            $"External/custom baseline tensor mismatch detected. " +
            $"Missing=[{string.Join(", ", missing.Take(20))}] " +
            $"Unexpected=[{string.Join(", ", unexpected.Take(20))}]. " +
            "MagicQuant only permits missing tensors when GGUF metadata explicitly declares fewer trailing NextN/MTP layers; " +
            "all model-trunk tensors must exactly match the source model.");
    }

    private static bool TryValidateDeclaredOptionalNextnOmission(
        GgufTensorReadResult nativeMetadata,
        GgufTensorReadResult externalMetadata,
        IReadOnlyCollection<string> missing,
        out int omittedLayerCount)
    {
        omittedLayerCount = 0;
        if (missing.Count == 0 ||
            string.IsNullOrWhiteSpace(nativeMetadata.Architecture) ||
            !string.Equals(nativeMetadata.Architecture, externalMetadata.Architecture,
                StringComparison.Ordinal) ||
            nativeMetadata.BlockCount is not > 0 ||
            externalMetadata.BlockCount is not > 0 ||
            nativeMetadata.NextnPredictLayers is not >= 0 ||
            externalMetadata.NextnPredictLayers is not >= 0)
        {
            return false;
        }

        int nativeBlockCount = nativeMetadata.BlockCount.Value;
        int externalBlockCount = externalMetadata.BlockCount.Value;
        int nativeNextnLayers = nativeMetadata.NextnPredictLayers.Value;
        int externalNextnLayers = externalMetadata.NextnPredictLayers.Value;
        int nativeTrunkBlockCount = nativeBlockCount - nativeNextnLayers;
        int externalTrunkBlockCount = externalBlockCount - externalNextnLayers;
        int omittedNextnLayers = nativeNextnLayers - externalNextnLayers;

        // Qwen GGUF block_count includes its trailing NextN layers. Therefore 65/1 and
        // 64/0 describe the same 64-block model trunk. Both the trunk size and the exact
        // block-count reduction must agree with the declared NextN reduction.
        if (omittedNextnLayers <= 0 ||
            nativeTrunkBlockCount <= 0 ||
            externalTrunkBlockCount != nativeTrunkBlockCount ||
            nativeBlockCount - externalBlockCount != omittedNextnLayers)
        {
            return false;
        }

        int firstOmittedBlock = externalBlockCount;
        int endExclusive = nativeBlockCount;

        bool IsInOmittedRange(string tensorName) =>
            TryGetBlockIndex(tensorName, out int blockIndex) &&
            blockIndex >= firstOmittedBlock &&
            blockIndex < endExclusive;

        // A file declaring fewer NextN layers must omit those layers completely. A partial
        // block is still malformed and must not be accepted as an optional-layer omission.
        if (externalMetadata.TensorNames.Any(IsInOmittedRange) || missing.Any(x => !IsInOmittedRange(x)))
            return false;

        omittedLayerCount = omittedNextnLayers;
        return true;
    }

    private static bool TryGetBlockIndex(string tensorName, out int blockIndex)
    {
        blockIndex = default;
        const string prefix = "blk.";
        if (!tensorName.StartsWith(prefix, StringComparison.Ordinal))
            return false;

        int separator = tensorName.IndexOf('.', prefix.Length);
        if (separator <= prefix.Length)
            return false;

        return int.TryParse(tensorName.AsSpan(prefix.Length, separator - prefix.Length), out blockIndex);
    }
}
