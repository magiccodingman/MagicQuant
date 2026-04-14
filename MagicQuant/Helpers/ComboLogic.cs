using System.Collections.Immutable;
using System.Numerics;
using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class ComboLogic
{
    private static readonly ImmutableArray<TensorGroup> GroupsOrdered =
        TReg.All.OrderBy(g => g.UniqueId).ToImmutableArray();

    public static ImmutableArray<byte[]> GetAllowedSchemeIdsPerGroup(BaselineQuants baseQuant)
    {
        bool baseRequiresImatrix = baseQuant.RequiresImatrix;

        var schemesForBase = TensorWeightScheme.All
            .Where(s => baseRequiresImatrix || !s.RequiresImatrix)
            .ToImmutableArray();

        if (schemesForBase.IsEmpty)
            throw new InvalidOperationException("No tensor schemes available for this base.");

        var builder = ImmutableArray.CreateBuilder<byte[]>();
        var unusedIds = Cache.UnusedTensorGroups.Select(x => x.UniqueId).ToHashSet();

        foreach (var group in GroupsOrdered)
        {
            if (unusedIds.Contains(group.UniqueId))
            {
                builder.Add([TensorWeightScheme.NULL.UniqueId]);
                continue;
            }

            var ids = schemesForBase
                .Where(s => !s.IsBannedFor(group))
                .Select(s => s.UniqueId)
                .Distinct()
                .ToArray();

            if (ids.Length == 0)
            {
                throw new InvalidOperationException(
                    $"Group '{group.Name}' has no valid tensor schemes for base '{string.Join("/", baseQuant.Names)}'.");
            }

            builder.Add(ids);
        }

        return builder.ToImmutable();
    }

    public static BigInteger CountCombinations(in BaselineQuants baseQuant)
    {
        var allowed = GetAllowedSchemeIdsPerGroup(baseQuant);

        BigInteger total = BigInteger.One;
        for (int i = 0; i < allowed.Length; i++)
            total *= allowed[i].Length;

        return total;
    }
}

public static class ComboCounter
{
    public static BigInteger CountForBase(BaselineQuants baseQuant)
    {
        var allowed = ComboLogic.GetAllowedSchemeIdsPerGroup(baseQuant);

        BigInteger total = BigInteger.One;
        for (int i = 0; i < allowed.Length; i++)
            total *= allowed[i].Length;

        return total;
    }

    public static BigInteger CountAll()
    {
        BigInteger sum = BigInteger.Zero;

        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
            sum += CountForBase(baseline);

        return sum;
    }
}