using System.Collections.Immutable;

namespace MQ.DB.Models;

/// <summary>
/// Represents a static semantic relationship between tensor groups that are
/// independently tracked but expected to behave as a connected optimization unit.
///
/// A synergy group does not replace normal tensor group identity.
/// It exists to describe known architectural coupling between groups.
/// </summary>
public record TensorGroupSynergy(
    byte UniqueId,
    string Name,
    ImmutableArray<TensorGroup> Groups)
{
    public bool Contains(TensorGroup group) =>
        Groups.Any(g => g.UniqueId == group.UniqueId);

    public bool Contains(string groupName) =>
        Groups.Any(g => g.Name.Equals(groupName, StringComparison.OrdinalIgnoreCase));

    public bool ContainsAny(IEnumerable<TensorGroup> groups)
    {
        foreach (var group in groups)
        {
            if (Contains(group))
                return true;
        }

        return false;
    }

    public bool ContainsAll(IEnumerable<TensorGroup> groups)
    {
        foreach (var group in groups)
        {
            if (!Contains(group))
                return false;
        }

        return true;
    }
}

/// <summary>
/// Static registry for tensor-group relationships that should receive synergy-aware
/// second-chance review after normal bad-trade pruning.
///
/// Keep this intentionally small and explicit. Synergy is an architectural exception,
/// not a fuzzy runtime heuristic.
/// </summary>
public static class TensorGroupSynergies
{
    /// <summary>
    /// Feed-forward projection groups are independently measurable, but their
    /// downstream hybrid behavior can be coupled enough that a candidate surviving
    /// in one group deserves a KLD-only second chance in the other when the only
    /// normal bad-trade objection was PPL volatility.
    /// </summary>
    public static TensorGroupSynergy FeedForwardUpGateDown { get; } = new(
        UniqueId: 0,
        Name: "ffn_up_gate+ffn_down",
        Groups: ImmutableArray.Create(TReg.FfnUpGate, TReg.FfnDown));

    public static ImmutableArray<TensorGroupSynergy> All { get; } =
        ImmutableArray.Create(FeedForwardUpGateDown);

    public static ImmutableArray<TensorGroupSynergy> GetContainingSynergies(TensorGroup group) =>
        All.Where(x => x.Contains(group)).ToImmutableArray();

    public static bool IsInAnySynergy(TensorGroup group) =>
        All.Any(x => x.Contains(group));

    public static bool AreInSameSynergy(TensorGroup left, TensorGroup right) =>
        All.Any(x => x.Contains(left) && x.Contains(right));
}
