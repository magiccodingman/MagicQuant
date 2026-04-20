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
        bool imatrixAvailable = RuntimeSearchSpace.HasUsableImatrix();

        var schemesForRun = TensorWeightScheme.All_Allowed_Hybrid_Quants
            .Where(s => imatrixAvailable || !s.RequiresImatrix)
            .ToImmutableArray();

        if (schemesForRun.IsEmpty)
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

            var ids = new List<byte>();

            if (!RuntimeSearchSpace.IsBf16TensorChoiceSuppressed(group))
                ids.Add(TensorWeightScheme.BF16_F16.UniqueId);

            foreach (var scheme in schemesForRun)
            {
                if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId || scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
                    continue;

                if (RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(group, scheme))
                    continue;

                ids.Add(scheme.UniqueId);
            }

            ids = ids.Distinct().OrderBy(x => x).ToList();

            if (ids.Count == 0)
                throw new InvalidOperationException($"Group '{group.Name}' has no valid tensor schemes for base '{string.Join("/", baseQuant.Names)}'.");

            builder.Add(ids.ToArray());
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
