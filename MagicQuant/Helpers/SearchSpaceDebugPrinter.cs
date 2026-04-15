using System.Numerics;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

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

        var explicitBannedGroups = RuntimeSearchSpace.GetGroupsWithExplicitQuantBanned();
        if (explicitBannedGroups.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Explicit-quant-banned groups:[/] {explicitBannedGroups.Count}");
            foreach (var group in explicitBannedGroups)
                AnsiConsole.MarkupLine($"  [yellow]- {group.Name}[/]");
        }

        var bf16SuppressedGroups = RuntimeSearchSpace.GetBf16SuppressedGroups();
        if (bf16SuppressedGroups.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]BF16 tensor-choice suppressed groups:[/] {bf16SuppressedGroups.Count}");
            foreach (var group in bf16SuppressedGroups)
                AnsiConsole.MarkupLine($"  [yellow]- {group.Name}[/]");
        }

        var unusedIds = Cache.UnusedTensorGroups.Select(x => x.UniqueId).ToHashSet();

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

                var names = ids.Select(id =>
                {
                    if (id == TensorWeightScheme.NULL.UniqueId)
                        return "NULL";

                    var scheme = TensorWeightScheme.All.FirstOrDefault(x => x.UniqueId == id);
                    return scheme?.Names[0] ?? $"Unknown({id})";
                }).ToList();

                string state =
                    unusedIds.Contains(group.UniqueId) ? "unused->NULL" :
                    RuntimeSearchSpace.IsGroupExplicitQuantBanned(group) ? "BF16-only" :
                    RuntimeSearchSpace.IsBf16TensorChoiceSuppressed(group) ? "BF16-suppressed" :
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