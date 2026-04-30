using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class FinalSurvivorSelectionCliService
{
    private readonly FinalArtifactNamingService _namingService = new();

    public IReadOnlyList<FinalSelectionRow> Prompt(
        IReadOnlyCollection<BenchmarkSnapshotRecord> survivors,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        BenchmarkSnapshotRecord? pplReference = null)
    {
        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);
        var reservedFileNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        double? referencePpl = ResolveReferencePpl(pplReference, pureBaselineSnapshots, survivors);

        var rows = survivors
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .Select((snapshot, index) =>
            {
                var name = _namingService.BuildName(snapshot, namingContext, reservedFileNames);
                return new FinalSelectionRow
                {
                    Id = index + 1,
                    Enabled = true,
                    Snapshot = snapshot,
                    PlannedFileName = name.FileName,
                    PlannedDisplayName = name.DisplayName,
                    PlannedProviderName = ResolveProviderName(snapshot, name),
                    PlannedQuantFamily = name.QuantFamilyOrBaseline
                };
            })
            .ToList();

        if (rows.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No survivors remained after brutal real-truth elimination.[/]");
            return rows;
        }

        while (true)
        {
            Render(rows, referencePpl);

            string input = AnsiConsole.Prompt(
                    new TextPrompt<string>("Toggle [cyan]row number[/], or type [green]ready[/] to continue")
                        .AllowEmpty())
                .Trim();

            if (string.IsNullOrWhiteSpace(input) ||
                string.Equals(input, "ready", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(input, "continue", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(input, "done", StringComparison.OrdinalIgnoreCase))
            {
                if (rows.Any(x => x.Enabled))
                    return rows;

                AnsiConsole.MarkupLine("[red]At least one survivor must remain enabled.[/]");
                continue;
            }

            if (!int.TryParse(input, out var id))
            {
                AnsiConsole.MarkupLine($"[yellow]Unknown selection command:[/] {Markup.Escape(input)}");
                continue;
            }

            var row = rows.FirstOrDefault(x => x.Id == id);
            if (row == null)
            {
                AnsiConsole.MarkupLine($"[yellow]No row exists with ID {id}.[/]");
                continue;
            }

            row.Enabled = !row.Enabled;
        }
    }

    private static void Render(IReadOnlyCollection<FinalSelectionRow> rows, double? referencePpl)
    {
        AnsiConsole.Clear();
        AnsiConsole.Write(new Rule("[yellow]Final Survivor Selection[/]") { Justification = Justify.Left });

        var table = new Table().Border(TableBorder.Rounded).Expand();
        table.AddColumn("ID");
        table.AddColumn("State");
        table.AddColumn("Display / Model");
        table.AddColumn("Provider");
        table.AddColumn("Quant Family");
        table.AddColumn("KLD");
        table.AddColumn("PPL Δ %");
        table.AddColumn("Size (GB)");

        foreach (var row in rows)
        {
            var snap = row.Snapshot;
            string state = row.Enabled ? "[green]ENABLED[/]" : "[red]DISABLED[/]";
            string displayValue = string.IsNullOrWhiteSpace(row.PlannedDisplayName) ? snap.DisplayName : row.PlannedDisplayName;
            string providerValue = string.IsNullOrWhiteSpace(row.PlannedProviderName) ? snap.ProviderName : row.PlannedProviderName;
            string familyValue = string.IsNullOrWhiteSpace(row.PlannedQuantFamily) ? snap.BaselineFamily : row.PlannedQuantFamily;

            string display = row.Enabled ? Markup.Escape(displayValue) : $"[grey]{Markup.Escape(displayValue)}[/]";
            string provider = row.Enabled ? Markup.Escape(providerValue) : $"[grey]{Markup.Escape(providerValue)}[/]";
            string family = row.Enabled ? Markup.Escape(familyValue) : $"[grey]{Markup.Escape(familyValue)}[/]";
            string kld = row.Enabled ? $"[cyan]{snap.Kld:0.000000}[/]" : $"[grey]{snap.Kld:0.000000}[/]";
            string pplDelta = FormatPplDeltaPercent(snap.Ppl, referencePpl);
            string ppl = row.Enabled ? $"[cyan]{pplDelta}[/]" : $"[grey]{pplDelta}[/]";
            string sizeGb = (snap.SizeBytes / 1000d / 1000d / 1000d).ToString("0.00");

            table.AddRow(
                row.Id.ToString(),
                state,
                display,
                provider,
                family,
                kld,
                ppl,
                row.Enabled ? $"[cyan]{sizeGb}[/]" : $"[grey]{sizeGb}[/]");
        }

        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine("[grey]PPL Δ % is measured against the native/reference PPL when available. Negative is better; larger positive values are worse.[/]");
    }

    private static string ResolveProviderName(BenchmarkSnapshotRecord snapshot, FinalArtifactName name)
    {
        if (snapshot.IsHybrid)
            return "MagicQuant";

        if (string.Equals(name.ProviderToken, "MQ", StringComparison.OrdinalIgnoreCase))
            return "MagicQuant";

        return HybridBenchmarkRepository.ResolveProviderName(snapshot.Quant, exportNaming: false);
    }

    private static double? ResolveReferencePpl(
        BenchmarkSnapshotRecord? pplReference,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        IReadOnlyCollection<BenchmarkSnapshotRecord> survivors)
    {
        if (pplReference is { Ppl: > 0d })
            return pplReference.Ppl;

        var bestPure = pureBaselineSnapshots
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .ThenByDescending(x => x.SizeBytes)
            .FirstOrDefault();

        if (bestPure != null)
            return bestPure.Ppl;

        return survivors
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .FirstOrDefault()
            ?.Ppl;
    }

    private static string FormatPplDeltaPercent(double ppl, double? referencePpl)
    {
        if (referencePpl is null or <= 0d || ppl <= 0d)
            return "n/a";

        double delta = ((ppl - referencePpl.Value) / referencePpl.Value) * 100d;
        return $"{delta:0.000}%";
    }
}
