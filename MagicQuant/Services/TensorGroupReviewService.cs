using MagicQuant.Models.Learning;
using MagicQuant.Services.Learning;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class TensorGroupReviewService
{
    private readonly TensorGroupingAuditService _auditService = new();

    public async Task<TensorGroupingAuditResult> ReviewNativeTensorGroupingAsync(
        QuantizationService quantizationService,
        string nativeGgufPath,
        bool requireConfirmation,
        CancellationToken ct = default)
    {
        if (quantizationService == null)
            throw new ArgumentNullException(nameof(quantizationService));

        if (string.IsNullOrWhiteSpace(nativeGgufPath) || !File.Exists(nativeGgufPath))
            throw new FileNotFoundException($"Native GGUF path not found for tensor-group review: {nativeGgufPath}");

        var tensorTypes = await quantizationService.ReadExactTensorTypesAsync(nativeGgufPath, ct);
        var truth = tensorTypes
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(
                x => x.Key,
                x => new LearnedTensorTruth(x.Key, x.Value, LearningSource.GgufOnly),
                StringComparer.Ordinal);

        var audit = _auditService.Audit(truth.Keys.ToList(), truth);

        PrintReview(nativeGgufPath, truth, audit);

        if (audit.HasFatalIssues)
        {
            throw new InvalidOperationException(
                "Tensor-group regex review found fatal grouping issues before learning/search could continue. " +
                $"Ambiguous={audit.Ambiguous.Count}, IllegalUnresolved={audit.IllegalUnresolved.Count}. " +
                "Fix tensor_groups.yaml and rerun.");
        }

        if (requireConfirmation)
        {
            bool confirmed = AnsiConsole.Confirm(
                "Continue with this tensor grouping profile? Review the counts above before saying yes.");

            if (!confirmed)
            {
                throw new OperationCanceledException(
                    "Evolution run cancelled by user after tensor-group profile review. No tensor-group-scoped learning/search work was started.");
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[yellow]Tensor-group confirmation skipped by config/CLI.[/]");
        }

        return audit;
    }

    private static void PrintReview(
        string nativeGgufPath,
        IReadOnlyDictionary<string, LearnedTensorTruth> truth,
        TensorGroupingAuditResult audit)
    {
        string snapshotJson = TensorGroupProfileService.BuildSnapshotJson();
        string fingerprint = TensorGroupProfileService.ComputeSnapshotHash(snapshotJson);

        AnsiConsole.Write(new Rule("[yellow]Tensor Group Regex Review[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[grey]Native source:[/] {Markup.Escape(Path.GetFileName(nativeGgufPath))}");
        AnsiConsole.MarkupLine($"[grey]Tensor group profile hash:[/] [cyan]{Markup.Escape(fingerprint[..Math.Min(16, fingerprint.Length)])}[/]");
        AnsiConsole.MarkupLine($"[grey]Tensors inspected:[/] [cyan]{truth.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]BaseQuant exception tensors:[/] [cyan]{audit.BaseQuantExceptions.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]Ambiguous tensors:[/] [{(audit.Ambiguous.Count == 0 ? "green" : "red")}]{audit.Ambiguous.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]Illegal unresolved tensors:[/] [{(audit.IllegalUnresolved.Count == 0 ? "green" : "red")}]{audit.IllegalUnresolved.Count:N0}[/]");

        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("Id")
            .AddColumn("Group")
            .AddColumn(new TableColumn("Tensors").RightAligned())
            .AddColumn("Top native types")
            .AddColumn("Examples");

        foreach (var group in TReg.All.OrderBy(x => x.UniqueId))
        {
            var tensors = audit.GroupedByTensor
                .Where(x => x.Value.PrimaryGroup?.UniqueId == group.UniqueId)
                .Select(x => x.Key)
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();

            var distribution = tensors
                .Select(t => truth.TryGetValue(t, out var row) ? row.FinalQuantType : "unknown")
                .GroupBy(x => x, StringComparer.Ordinal)
                .OrderByDescending(x => x.Count())
                .ThenBy(x => x.Key, StringComparer.Ordinal)
                .Take(4)
                .Select(x => $"{x.Key}:{x.Count():N0}");

            var examples = tensors.Take(4).Select(Markup.Escape);

            table.AddRow(
                group.UniqueId.ToString(),
                Markup.Escape(group.Name),
                tensors.Count.ToString("N0"),
                Markup.Escape(string.Join(", ", distribution)),
                string.Join("\n", examples));
        }

        AnsiConsole.Write(table);

        PrintIssuePreview("Ambiguous group collisions", audit.Ambiguous);
        PrintIssuePreview("Illegal unresolved tensors", audit.IllegalUnresolved);
        PrintIssuePreview("BaseQuant exception tensors", audit.BaseQuantExceptions);
    }

    private static void PrintIssuePreview(string heading, IReadOnlyList<TensorGroupingAuditIssue> issues)
    {
        if (issues.Count == 0)
            return;

        AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(heading)}:[/] showing first {Math.Min(10, issues.Count):N0} of {issues.Count:N0}");
        foreach (var issue in issues.Take(10))
        {
            var extra = issue.MatchedGroups.Count > 0
                ? $" groups=[{string.Join(", ", issue.MatchedGroups)}]"
                : issue.MatchedExceptionPattern != null
                    ? $" pattern={issue.MatchedExceptionPattern}"
                    : string.Empty;

            AnsiConsole.MarkupLine($"  [grey]-[/] {Markup.Escape(issue.TensorName)} [grey]{Markup.Escape(extra)}[/]");
        }
    }
}
