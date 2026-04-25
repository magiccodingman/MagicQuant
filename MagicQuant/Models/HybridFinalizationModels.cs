using System.Security.Cryptography;
using System.Text;
using MQ.DB.Models;

namespace MagicQuant.Models;

public static class TensorConfigIdentity
{
    public static string ToKey(TensorConfig config)
    {
        return string.Join(":",
            config.BaseQuant,
            config.Embeddings,
            config.LmHead,
            config.AttnQ,
            config.AttnKV,
            config.AttnOutput,
            config.FfnUpGate,
            config.FfnDown,
            config.MoeExperts,
            config.MoeRouter);
    }

    public static bool IsPureBaseline(TensorConfig config)
    {
        return config.Embeddings == BaselineQuants.TensorConfigNullSlotValue &&
               config.LmHead == BaselineQuants.TensorConfigNullSlotValue &&
               config.AttnQ == BaselineQuants.TensorConfigNullSlotValue &&
               config.AttnKV == BaselineQuants.TensorConfigNullSlotValue &&
               config.AttnOutput == BaselineQuants.TensorConfigNullSlotValue &&
               config.FfnUpGate == BaselineQuants.TensorConfigNullSlotValue &&
               config.FfnDown == BaselineQuants.TensorConfigNullSlotValue &&
               config.MoeExperts == BaselineQuants.TensorConfigNullSlotValue &&
               config.MoeRouter == BaselineQuants.TensorConfigNullSlotValue;
    }

    public static IReadOnlyList<(TensorGroup Group, byte StoredValue)> EnumerateGroupSlots(TensorConfig config)
    {
        return
        [
            (TReg.Embeddings, config.Embeddings),
            (TReg.LmHead, config.LmHead),
            (TReg.AttnQ, config.AttnQ),
            (TReg.AttnKV, config.AttnKV),
            (TReg.AttnOutput, config.AttnOutput),
            (TReg.FfnUpGate, config.FfnUpGate),
            (TReg.FfnDown, config.FfnDown),
            (TReg.MoeExperts, config.MoeExperts),
            (TReg.MoeRouter, config.MoeRouter)
        ];
    }

    public static string StableHash(string value)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        return Convert.ToHexString(bytes).ToLowerInvariant();
    }
}

public sealed class EffectiveStateResolutionResult
{
    public TensorConfig Config { get; init; }
    public string EffectiveStateKey { get; init; } = string.Empty;
    public bool HasUnknownMappings { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
    public IReadOnlyDictionary<string, string> GroupStates { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);
    public string BaseState { get; init; } = string.Empty;
}

public sealed class PredictedCandidateEvaluation
{
    public TensorConfig Config { get; init; }
    public HybridQuant Quant { get; init; } = default!;
    public ulong PredictedSizeBytes { get; init; }
    public double PredictedKldCost { get; init; }
    public double PredictedPplCost { get; init; }
    public double CompositeScore { get; init; }
    public string EffectiveStateKey { get; init; } = string.Empty;
    public bool HasUnknownMappings { get; init; }
    public byte BaseBitRange { get; init; }
    public bool IsPureBaseline { get; init; }
    public List<string> Notes { get; init; } = new();
}

public sealed class BitRangeBucketDefinition
{
    public string Key { get; init; } = string.Empty;
    public byte LowerBitRange { get; init; }
    public byte UpperBitRange { get; init; }
    public ulong LowerAnchorSizeBytes { get; init; }
    public ulong UpperAnchorSizeBytes { get; init; }
    public bool WasSkipped { get; init; }
    public string? SkipReason { get; init; }
}

public sealed class BucketedCandidate
{
    public PredictedCandidateEvaluation Evaluation { get; init; } = default!;
    public BitRangeBucketDefinition Bucket { get; init; } = default!;
}

public sealed class BucketPruneDiagnostics
{
    public string BucketKey { get; init; } = string.Empty;
    public ulong LowerAnchorSizeBytes { get; init; }
    public ulong UpperAnchorSizeBytes { get; init; }
    public int IncomingCount { get; set; }
    public int RemovedCount { get; set; }
    public int KeptCount { get; set; }
    public bool Skipped { get; set; }
    public string? SkipReason { get; set; }
    public Dictionary<string, int> RemovalReasons { get; } = new(StringComparer.OrdinalIgnoreCase);
    public List<string> Notes { get; } = new();

    public void CountReason(string reason)
    {
        RemovalReasons.TryGetValue(reason, out var current);
        RemovalReasons[reason] = current + 1;
    }
}

public sealed class SurvivalStageReport
{
    public long StartingCount { get; set; }
    public long EndingCount { get; set; }
    public long RemovedCount => StartingCount - EndingCount;
    public Dictionary<string, long> RemovalCounts { get; } = new(StringComparer.OrdinalIgnoreCase);
    public List<BucketPruneDiagnostics> BucketDiagnostics { get; } = new();
    public List<string> Notes { get; } = new();

    public void AddRemoval(string reason, long count)
    {
        RemovalCounts.TryGetValue(reason, out var current);
        RemovalCounts[reason] = current + count;
    }
}

public sealed class BenchmarkSnapshotRecord
{
    public TensorConfig Config { get; init; }
    public HybridQuant Quant { get; init; } = default!;
    public string DisplayName { get; init; } = string.Empty;
    public string ProviderName { get; init; } = string.Empty;
    public string BaselineFamily { get; init; } = string.Empty;
    public bool IsHybrid { get; init; }
    public bool IsExternalPureBaseline { get; init; }
    public ulong SizeBytes { get; init; }
    public double Kld { get; init; }
    public double Ppl { get; init; }
    public string? OutputModelPath { get; init; }
    public string? ExternalRepositoryUrl { get; init; }
}

public sealed class FinalSelectionRow
{
    public int Id { get; set; }
    public bool Enabled { get; set; } = true;
    public BenchmarkSnapshotRecord Snapshot { get; init; } = default!;

    // Planned public/export identity. The CLI previews these names and the export
    // service reuses them, so a user never sees one name in the selection UI and
    // a different name in the produced GGUF/README.
    public string PlannedFileName { get; set; } = string.Empty;
    public string PlannedDisplayName { get; set; } = string.Empty;
    public string PlannedProviderName { get; set; } = string.Empty;
    public string PlannedQuantFamily { get; set; } = string.Empty;
}

public sealed class ExportedArtifactRecord
{
    public BenchmarkSnapshotRecord Snapshot { get; init; } = default!;
    public string DisplayName { get; init; } = string.Empty;
    public string ProviderName { get; init; } = string.Empty;
    public string BaselineFamily { get; init; } = string.Empty;
    public bool IsExternalReference { get; init; }
    public string? FileName { get; init; }
    public string? FullPath { get; init; }
    public string DownloadTarget { get; init; } = string.Empty;
    public ulong ExpectedSizeBytes { get; init; }
    public ulong? ActualSizeBytes { get; set; }
    public EffectiveStateResolutionResult? EffectiveState { get; init; }
}

public sealed class HybridMapEntry
{
    public string ExportedFileName { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string ProviderSource { get; set; } = string.Empty;
    public string BaselineFamily { get; set; } = string.Empty;
    public string? OriginalReferenceBaseline { get; set; }
    public Dictionary<string, string> TensorGroups { get; set; } = new(StringComparer.Ordinal);
    public string EffectiveQuantStateKey { get; set; } = string.Empty;
    public bool HasUnknownMappings { get; set; }
    public List<string> Warnings { get; set; } = new();
    public bool UsedImatrix { get; set; }
    public ulong ExpectedSizeBytes { get; set; }
    public ulong? ActualSizeBytes { get; set; }
    public string? OriginalExternalSource { get; set; }
}

public sealed class FinalRealEliminationResult
{
    public IReadOnlyList<BenchmarkSnapshotRecord> Survivors { get; init; } = Array.Empty<BenchmarkSnapshotRecord>();
    public IReadOnlyList<BenchmarkSnapshotRecord> Eliminated { get; init; } = Array.Empty<BenchmarkSnapshotRecord>();
}

public sealed class CombinationSurvivalExecutionResult
{
    public IReadOnlyList<BenchmarkSnapshotRecord> BenchmarkedSnapshots { get; init; } = Array.Empty<BenchmarkSnapshotRecord>();
    public IReadOnlyList<BenchmarkSnapshotRecord> BrutalSurvivors { get; init; } = Array.Empty<BenchmarkSnapshotRecord>();
    public IReadOnlyList<FinalSelectionRow> SelectedRows { get; init; } = Array.Empty<FinalSelectionRow>();
    public IReadOnlyList<ExportedArtifactRecord> ExportedArtifacts { get; init; } = Array.Empty<ExportedArtifactRecord>();
    public IReadOnlyList<BucketPruneDiagnostics> BucketDiagnostics { get; init; } = Array.Empty<BucketPruneDiagnostics>();
    public SurvivalStageReport SurvivalReport { get; init; } = new();
    public IReadOnlyList<BaselineEliminationRecord> Eliminations { get; init; } = Array.Empty<BaselineEliminationRecord>();
    public IReadOnlyList<CandidateValidationResult> ValidationFailures { get; init; } = Array.Empty<CandidateValidationResult>();
}
