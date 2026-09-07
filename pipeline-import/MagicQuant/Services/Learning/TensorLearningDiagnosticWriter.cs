using System.Text;
using System.Text.Json;
using MagicQuant.Helpers;
using MagicQuant.Models.Learning;
using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Services.Learning;

public sealed class TensorLearningDiagnosticWriter
{
    public async Task<string> WriteFailureAsync(
        string baselineName,
        string schemeName,
        string sourceKind,
        string? sourceRepository,
        string? sourceFileName,
        IReadOnlyDictionary<string, LearnedTensorTruth> truthByTensor,
        TensorGroupingAuditResult audit,
        TensorTruthVerificationResult verification,
        CancellationToken ct = default)
    {
        string dir = GetTensorConfigLogDirectory();
        Directory.CreateDirectory(Path.Combine(Cache.ModelMagicQuantDirectory!, "Logs"));
        await HardDeleteHelper.DeleteDirectoryIfExistsAsync(dir, ct);
        Directory.CreateDirectory(dir);

        string safeBaseline = SanitizeForFileName(baselineName);
        string safeScheme = SanitizeForFileName(schemeName);
        string baseName = $"tensor-group-learning-failure-{safeBaseline}-{safeScheme}";
        string jsonPath = Path.Combine(dir, baseName + ".json");
        string txtPath = Path.Combine(dir, baseName + ".txt");

        var payload = new
        {
            GeneratedUtc = DateTime.UtcNow,
            ModelDirectory = Cache.ModelDirectory,
            ModelMagicQuantDirectory = Cache.ModelMagicQuantDirectory,
            ActiveConfigPath = TReg.TensorGroupsYamlPathOverride,
            Baseline = baselineName,
            Scheme = schemeName,
            SourceKind = sourceKind,
            SourceRepository = sourceRepository,
            SourceFileName = sourceFileName,
            TotalTruthTensors = truthByTensor.Count,
            SemanticMatchedCount = audit.GroupedByTensor.Count(x => x.Value.PrimaryGroup != null),
            BaseQuantExceptionCount = audit.BaseQuantExceptions.Count,
            IllegalUnresolvedCount = audit.IllegalUnresolved.Count,
            AmbiguousCount = audit.Ambiguous.Count,
            MismatchCount = verification.HardMismatches.Count + verification.SoftMismatches.Count,
            HighSeverityMismatchCount = verification.HardMismatches.Count,
            BaseQuantExceptions = audit.BaseQuantExceptions,
            IllegalUnresolved = audit.IllegalUnresolved,
            Ambiguous = audit.Ambiguous,
            HardMismatches = verification.HardMismatches,
            SoftMismatches = verification.SoftMismatches,
            LogOnly = verification.LogOnly
        };

        await File.WriteAllTextAsync(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }), ct);

        var sb = new StringBuilder();
        sb.AppendLine("MagicQuant Tensor Group Learning Failure");
        sb.AppendLine($"GeneratedUtc: {DateTime.UtcNow:O}");
        sb.AppendLine($"ModelDirectory: {Cache.ModelDirectory}");
        sb.AppendLine($"ModelMagicQuantDirectory: {Cache.ModelMagicQuantDirectory}");
        sb.AppendLine($"ActiveConfigPath: {TReg.TensorGroupsYamlPathOverride}");
        sb.AppendLine($"Baseline: {baselineName}");
        sb.AppendLine($"Scheme: {schemeName}");
        sb.AppendLine($"SourceKind: {sourceKind}");
        sb.AppendLine($"SourceRepository: {sourceRepository}");
        sb.AppendLine($"SourceFileName: {sourceFileName}");
        sb.AppendLine($"TotalTruthTensors: {truthByTensor.Count}");
        sb.AppendLine($"SemanticMatchedCount: {audit.GroupedByTensor.Count(x => x.Value.PrimaryGroup != null)}");
        sb.AppendLine($"BaseQuantExceptionCount: {audit.BaseQuantExceptions.Count}");
        sb.AppendLine($"IllegalUnresolvedCount: {audit.IllegalUnresolved.Count}");
        sb.AppendLine($"AmbiguousCount: {audit.Ambiguous.Count}");
        sb.AppendLine($"MismatchCount: {verification.HardMismatches.Count + verification.SoftMismatches.Count}");
        sb.AppendLine($"HighSeverityMismatchCount: {verification.HardMismatches.Count}");
        sb.AppendLine();

        AppendIssues(sb, "Base Quant Exceptions", audit.BaseQuantExceptions);
        AppendIssues(sb, "Illegal Unresolved", audit.IllegalUnresolved);
        AppendIssues(sb, "Ambiguous", audit.Ambiguous);

        sb.AppendLine("Mismatches:");
        foreach (var mismatch in verification.HardMismatches.Concat(verification.SoftMismatches))
            sb.AppendLine($"- {mismatch.TensorName}: log={mismatch.LogQuantType}, gguf={mismatch.GgufQuantType}, severity={(mismatch.IsHighSeverity ? "high" : "soft")}, source decision=GGUF");

        if (verification.LogOnly.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("Log-only tensors ignored:");
            foreach (var entry in verification.LogOnly)
                sb.AppendLine($"- {entry}");
        }

        await File.WriteAllTextAsync(txtPath, sb.ToString(), ct);
        return txtPath;
    }

    private static void AppendIssues(StringBuilder sb, string heading, IReadOnlyList<TensorGroupingAuditIssue> issues)
    {
        sb.AppendLine(heading + ":");
        if (issues.Count == 0)
        {
            sb.AppendLine("- none");
            sb.AppendLine();
            return;
        }

        foreach (var issue in issues)
        {
            var groups = issue.MatchedGroups.Count > 0 ? $", matchedGroups=[{string.Join(", ", issue.MatchedGroups)}]" : string.Empty;
            var pattern = string.IsNullOrWhiteSpace(issue.MatchedExceptionPattern) ? string.Empty : $", exceptionPattern={issue.MatchedExceptionPattern}";
            sb.AppendLine($"- {issue.TensorName}: quant={issue.FinalQuantType}, source={issue.LearningSource}{groups}{pattern}");
        }

        sb.AppendLine();
    }

    private static string GetTensorConfigLogDirectory()
    {
        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new InvalidOperationException("Cache.ModelMagicQuantDirectory is not set.");

        return Path.Combine(Cache.ModelMagicQuantDirectory, "Logs", "TensorConfigs");
    }

    private static string SanitizeForFileName(string value)
    {
        var invalidChars = Path.GetInvalidFileNameChars();
        var cleaned = new string(value.Select(ch => invalidChars.Contains(ch) ? '-' : ch).ToArray());
        return string.IsNullOrWhiteSpace(cleaned) ? "unknown" : cleaned;
    }
}
