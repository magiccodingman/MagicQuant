using System.Runtime.CompilerServices;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class MagicQuantDiagnostics
{
    private const string VerboseEnv = "MAGICQUANT_DIAG_VERBOSE_ISOLATION_PRUNING";
    private const string FocusGroupEnv = "MAGICQUANT_DIAG_GROUP";

    public static bool VerboseIsolationPruning =>
        string.Equals(Environment.GetEnvironmentVariable(VerboseEnv), "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Environment.GetEnvironmentVariable(VerboseEnv), "true", StringComparison.OrdinalIgnoreCase);

    public static string? FocusGroup => Environment.GetEnvironmentVariable(FocusGroupEnv);

    public static bool ShouldLogGroup(TensorGroup group)
    {
        if (!VerboseIsolationPruning)
            return false;

        if (string.IsNullOrWhiteSpace(FocusGroup))
            return true;

        return string.Equals(group.Name, FocusGroup, StringComparison.OrdinalIgnoreCase)
               || string.Equals(group.UniqueId.ToString(), FocusGroup, StringComparison.OrdinalIgnoreCase);
    }

    public static void Log(string tag, string message)
    {
        if (!VerboseIsolationPruning)
            return;
        AnsiConsole.MarkupLine($"[grey][diag:{Markup.Escape(tag)}][/]: {Markup.Escape(message)}");
    }

    public static string CandidateLabel(BaselineQuants candidate) => $"{candidate.Names[0]}(id={candidate.UniqueId})";

    public static void LogRuntimeMutation(
        string phase,
        TensorGroup group,
        BaselineQuants? candidate,
        string reason,
        int before,
        int after,
        [CallerMemberName] string caller = "")
    {
        if (!ShouldLogGroup(group))
            return;

        var candidateText = candidate == null ? "<none>" : CandidateLabel(candidate);
        Log("runtime-ban",
            $"phase={phase} group={group.Name}(id={group.UniqueId}) candidate={candidateText} reason=\"{reason}\" allowedBefore={before} allowedAfter={after} caller={caller}");
    }
}
