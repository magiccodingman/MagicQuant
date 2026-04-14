using MQ.DB.Models;
using Spectre.Console;
using System.Numerics;
using MQ.DB;

namespace MagicQuant.Helpers;

public static class SearchSpaceDebugPrinter
{
    public static void PrintCurrentSearchSpace(string title = "Current Runtime Search Space")
    {
        AnsiConsole.Write(new Rule($"[yellow]{Markup.Escape(title)}[/]") { Justification = Justify.Left });

        var activeBaselines = RuntimeSearchSpace.GetActiveCombinationBaselines().ToList();
        var disabledBaselines = BaselineQuants.All
            .Where(x => x.BaseConversionBase != null)
            .Where(x => RuntimeSearchSpace.IsCombinationBaselineDisabled(x))
            .OrderBy(x => x.UniqueId)
            .ToList();

        AnsiConsole.MarkupLine($"[green]Active combo baselines:[/] {activeBaselines.Count}");
        foreach (var baseline in activeBaselines)
            AnsiConsole.MarkupLine($"  [cyan]- {string.Join("/", baseline.Names)}[/] (Id={baseline.UniqueId})");

        if (disabledBaselines.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Disabled combo baselines:[/] {disabledBaselines.Count}");
            foreach (var baseline in disabledBaselines)
                AnsiConsole.MarkupLine($"  [grey]- {string.Join("/", baseline.Names)}[/] (Id={baseline.UniqueId})");
        }

        var locked = RuntimeSearchSpace.GetNativeLockedGroups()
            .OrderBy(x => x.UniqueId)
            .ToList();

        AnsiConsole.MarkupLine($"[green]Native-locked groups:[/] {locked.Count}");
        foreach (var group in locked)
            AnsiConsole.MarkupLine($"  [yellow]- {group.Name}[/] (Id={group.UniqueId})");

        var unusedIds = Cache.UnusedTensorGroups
            .Select(x => x.UniqueId)
            .ToHashSet();

        foreach (var baseline in activeBaselines)
        {
            AnsiConsole.Write(new Rule($"[blue]Base: {Markup.Escape(string.Join("/", baseline.Names))}[/]") { Justification = Justify.Left });

            var allowed = ComboLogic.GetAllowedSchemeIdsPerGroup(baseline);

            BigInteger baseCount = BigInteger.One;

            for (int i = 0; i < TReg.All.Length; i++)
            {
                var group = TReg.All.OrderBy(x => x.UniqueId).ElementAt(i);
                var ids = allowed[i];
                baseCount *= ids.Length;

                var names = ids
                    .Select(id =>
                    {
                        if (id == TensorWeightScheme.NULL.UniqueId)
                            return "NULL";

                        var scheme = TensorWeightScheme.All.FirstOrDefault(x => x.UniqueId == id);
                        return scheme?.Names[0] ?? $"Unknown({id})";
                    })
                    .ToList();

                string state =
                    unusedIds.Contains(group.UniqueId) ? "unused->NULL" :
                    RuntimeSearchSpace.IsGroupLockedToNative(group) ? "native-locked" :
                    "variable";

                AnsiConsole.MarkupLine(
                    $"  [cyan]{Markup.Escape(group.Name)}[/] => [green]{ids.Length}[/] choice(s) " +
                    $"[grey][[{Markup.Escape(state)}]][/] :: {Markup.Escape(string.Join(", ", names))}");
                
            }

            AnsiConsole.MarkupLine($"  [bold green]Base total:[/] {baseCount:N0}");
        }

        AnsiConsole.MarkupLine($"[bold yellow]Grand total:[/] {ComboCounter.CountAll():N0}");
    }
}