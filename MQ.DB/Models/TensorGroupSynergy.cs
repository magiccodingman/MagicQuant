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