using System;
using System.Linq;
using System.Collections.Generic;
using System.Numerics;
using MagicQuant.Services;
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
            .Where(x => x.IsCombinationCarrierCandidate)
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

        var learnedPrunedGroups = RuntimeSearchSpace.GetGroupsWithLearnedBaselineMissingPrunes();
        if (learnedPrunedGroups.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Learned-baseline-pruned groups:[/] {learnedPrunedGroups.Count}");

            foreach (var group in learnedPrunedGroups)
            {
                var learned = RuntimeSearchSpace.GetLearnedBaselineMissingPrunedCandidatesForGroup(group);

                var parts = learned.Select(x =>
                    $"{x.Candidate.Names[0]} (expected={FormatSchemeIds(x.ExpectedTensorWeightSchemeIds)}, matched={FormatSchemeIds(x.MatchedTensorWeightSchemeIds)}, missing={FormatSchemeIds(x.MissingTensorWeightSchemeIds)})");

                AnsiConsole.MarkupLine(
                    $"  [yellow]- {Markup.Escape(group.Name)}[/] :: [grey]{Markup.Escape(string.Join(", ", parts))}[/]");
            }
        }

        foreach (var baseline in activeBaselines)
        {
            AnsiConsole.Write(new Rule($"[blue]Base: {Markup.Escape(string.Join("/", baseline.Names))}[/]")
            {
                Justification = Justify.Left
            });

            var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(baseline);
            BigInteger baseCount = BigInteger.One;

            for (int i = 0; i < TReg.All.Length; i++)
            {
                var group = TReg.All.OrderBy(x => x.UniqueId).ElementAt(i);
                var ids = allowed[i];
                baseCount *= ids.Length;

                var names = ids.Select(id =>
                {
                    if (BaselineQuants.IsNullTensorConfigGroupSlot(id))
                        return "NULL";

                    return BaselineQuants.DecodeTensorConfigGroupSlotToBaseline(id).Names[0];
                }).ToList();

                string state = RuntimeSearchSpace.GetDisplayStateForGroup(group);

                AnsiConsole.MarkupLine(
                    $"  [cyan]{Markup.Escape(group.Name)}[/] => [green]{ids.Length}[/] choice(s) " +
                    $"[grey][[{Markup.Escape(state)}]][/] :: {Markup.Escape(string.Join(", ", names))}");
            }

            AnsiConsole.MarkupLine($"  [bold green]Base total:[/] {baseCount:N0}");
        }

        AnsiConsole.MarkupLine($"[bold yellow]Grand total:[/] {ComboCounter.CountAll():N0}");
    }
    public static void PrintIsolationGroupDecisions(
        string title,
        IEnumerable<IsolationGroupDecision> decisions,
        string winningLabel = "Winning candidate")
    {
        var ordered = decisions
            .OrderBy(x => x.GroupName, StringComparer.Ordinal)
            .ToList();

        if (ordered.Count == 0)
            return;

        AnsiConsole.Write(new Rule($"[yellow]{Markup.Escape(title)}[/]") { Justification = Justify.Left });

        foreach (var gd in ordered)
        {
            AnsiConsole.Write(
                new Rule($"[yellow]Isolation Group: {Markup.Escape(gd.GroupName)}[/]")
                {
                    Justification = Justify.Left
                });

            AnsiConsole.MarkupLine($"[green]Best savings:[/] {gd.BestReductionRatio:P2}");
            AnsiConsole.MarkupLine($"[green]{Markup.Escape(winningLabel)}:[/] {Markup.Escape(gd.WinningCandidate ?? "n/a")}");
            AnsiConsole.MarkupLine($"[green]Explicit quant banned:[/] {(gd.ExplicitQuantBanned ? "[red]yes[/]" : "[green]no[/]")}");
            AnsiConsole.MarkupLine($"[green]BF16 suppressed:[/] {(gd.Bf16Suppressed ? "[yellow]yes[/]" : "[green]no[/]")}");

            foreach (var line in gd.Candidates)
                AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(line)}[/]");
        }
    }

    private static string FormatSchemeIds(IEnumerable<byte> schemeIds)
    {
        var ids = schemeIds
            .Distinct()
            .OrderBy(x => x)
            .ToList();

        if (ids.Count == 0)
            return "<none>";

        var parts = ids.Select(id =>
        {
            var scheme = TensorWeightScheme.All.FirstOrDefault(x => x.UniqueId == id);
            return scheme?.Names[0] ?? id.ToString();
        });

        return string.Join("/", parts);
    }

}
