using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class FinalSurvivorSelectionCliService
{
    public IReadOnlyList<FinalSelectionRow> Prompt(IReadOnlyCollection<BenchmarkSnapshotRecord> survivors)
    {
        var rows = survivors
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .Select((snapshot, index) => new FinalSelectionRow
            {
                Id = index + 1,
                Enabled = true,
                Snapshot = snapshot
            })
            .ToList();

        if (rows.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No survivors remained after brutal real-truth elimination.[/]");
            return rows;
        }

        while (true)
        {
            Render(rows);

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

    private static void Render(IReadOnlyCollection<FinalSelectionRow> rows)
    {
        AnsiConsole.Clear();
        AnsiConsole.Write(new Rule("[yellow]Final Survivor Selection[/]") { Justification = Justify.Left });

        var table = new Table().Border(TableBorder.Rounded).Expand();
        table.AddColumn("ID");
        table.AddColumn("State");
        table.AddColumn("Display / Model");
        table.AddColumn("Provider");
        table.AddColumn("Quant Family / Baseline");
        table.AddColumn("KLD");
        table.AddColumn("PPL");
        table.AddColumn("Size (GB)");

        foreach (var row in rows)
        {
            var snap = row.Snapshot;
            string state = row.Enabled ? "[green]ENABLED[/]" : "[red]DISABLED[/]";
            string display = row.Enabled ? Markup.Escape(snap.DisplayName) : $"[grey]{Markup.Escape(snap.DisplayName)}[/]";
            string provider = row.Enabled ? Markup.Escape(snap.ProviderName) : $"[grey]{Markup.Escape(snap.ProviderName)}[/]";
            string family = row.Enabled ? Markup.Escape(snap.BaselineFamily) : $"[grey]{Markup.Escape(snap.BaselineFamily)}[/]";
            string kld = row.Enabled ? $"[cyan]{snap.Kld:0.000000}[/]" : $"[grey]{snap.Kld:0.000000}[/]";
            string ppl = row.Enabled ? $"[cyan]{snap.Ppl:0.0000}[/]" : $"[grey]{snap.Ppl:0.0000}[/]";
            string sizeGb = (snap.SizeBytes / 1024d / 1024d / 1024d).ToString("0.00");

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
        AnsiConsole.MarkupLine("[grey]All rows start enabled. Enter a row number to toggle it, then type ready when done.[/]");
    }
}
