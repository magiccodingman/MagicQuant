using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public class ExternalBaselineTensorParityTests
{
    [Fact]
    public void ExactTensorSet_IsAccepted()
    {
        var native = Metadata(nextnLayers: 1, "token_embd.weight", "blk.0.attn_q.weight", "blk.64.nextn.eh_proj.weight");
        var external = Metadata(nextnLayers: 1, "token_embd.weight", "blk.0.attn_q.weight", "blk.64.nextn.eh_proj.weight");

        var result = ExternalBaselineTensorParity.ValidateOrThrow(native, external);

        Assert.Empty(result.InheritedOptionalTensorNames);
        Assert.Equal(0, result.OmittedNextnLayerCount);
    }

    [Fact]
    public void MetadataDeclaredTrailingMtpOmission_IsAccepted()
    {
        var native = Metadata(
            nextnLayers: 1,
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.63.ffn_down.weight",
            "blk.64.attn_q.weight",
            "blk.64.nextn.eh_proj.weight");
        var external = Metadata(
            nextnLayers: 0,
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.63.ffn_down.weight");

        var result = ExternalBaselineTensorParity.ValidateOrThrow(native, external);

        Assert.Equal(1, result.OmittedNextnLayerCount);
        Assert.Equal(
            ["blk.64.attn_q.weight", "blk.64.nextn.eh_proj.weight"],
            result.InheritedOptionalTensorNames);
    }

    [Fact]
    public void MissingModelTrunkTensor_IsRejected()
    {
        var native = Metadata(
            nextnLayers: 1,
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.64.nextn.eh_proj.weight");
        var external = Metadata(nextnLayers: 0, "token_embd.weight");

        var ex = Assert.Throws<InvalidOperationException>(
            () => ExternalBaselineTensorParity.ValidateOrThrow(native, external));

        Assert.Contains("blk.0.attn_q.weight", ex.Message, StringComparison.Ordinal);
        Assert.Contains("model-trunk tensors", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingMtpTensorWithoutReducedMetadata_IsRejected()
    {
        var native = Metadata(
            nextnLayers: 1,
            "token_embd.weight",
            "blk.64.nextn.eh_proj.weight");
        var external = Metadata(nextnLayers: 1, "token_embd.weight");

        Assert.Throws<InvalidOperationException>(
            () => ExternalBaselineTensorParity.ValidateOrThrow(native, external));
    }

    [Fact]
    public void PartialOmittedMtpBlock_IsRejected()
    {
        var native = Metadata(
            nextnLayers: 1,
            "token_embd.weight",
            "blk.64.attn_q.weight",
            "blk.64.nextn.eh_proj.weight");
        var external = Metadata(
            nextnLayers: 0,
            "token_embd.weight",
            "blk.64.attn_q.weight");

        Assert.Throws<InvalidOperationException>(
            () => ExternalBaselineTensorParity.ValidateOrThrow(native, external));
    }

    [Fact]
    public void UnexpectedTensor_IsRejectedEvenWhenMtpIsOmitted()
    {
        var native = Metadata(
            nextnLayers: 1,
            "token_embd.weight",
            "blk.64.nextn.eh_proj.weight");
        var external = Metadata(
            nextnLayers: 0,
            "token_embd.weight",
            "unexpected.weight");

        var ex = Assert.Throws<InvalidOperationException>(
            () => ExternalBaselineTensorParity.ValidateOrThrow(native, external));

        Assert.Contains("unexpected.weight", ex.Message, StringComparison.Ordinal);
    }

    private static GgufTensorReadResult Metadata(int nextnLayers, params string[] tensorNames)
    {
        return new GgufTensorReadResult
        {
            Architecture = "qwen35",
            BlockCount = 64 + nextnLayers,
            NextnPredictLayers = nextnLayers,
            TensorNames = tensorNames.ToList(),
            TensorTypes = tensorNames.ToDictionary(x => x, _ => "BF16", StringComparer.Ordinal)
        };
    }
}
