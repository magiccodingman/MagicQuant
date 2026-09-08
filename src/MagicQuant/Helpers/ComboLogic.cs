using System.Collections.Immutable;
using System.Numerics;
using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class ComboLogic
{
    private static readonly ImmutableArray<TensorGroup> GroupsOrdered =
        TReg.All.OrderBy(g => g.UniqueId).ToImmutableArray();

    public static ImmutableArray<byte[]> GetAllowedCandidateIdsPerGroup(BaselineQuants baseQuant)
    {
        bool imatrixAvailable = RuntimeSearchSpace.HasUsableImatrix();
        var builder = ImmutableArray.CreateBuilder<byte[]>();
        var unusedIds = Cache.UnusedTensorGroups.Select(x => x.UniqueId).ToHashSet();

        foreach (var group in GroupsOrdered)
        {
            if (unusedIds.Contains(group.UniqueId))
            {
                builder.Add([BaselineQuants.TensorConfigNullSlotValue]);
                continue;
            }

            var ids = new List<byte>();

            foreach (var alias in BaselineQuants.GetExactHighPrecisionAliases(RuntimeSearchSpace.AllowHighPrecisionHybrids))
            {
                if (RuntimeSearchSpace.IsBf16TensorChoiceSuppressed(group))
                    continue;

                ids.Add(BaselineQuants.EncodeTensorConfigGroupSlot(alias));
            }

            var realCandidates = RuntimeSearchSpace.GetAllowedRealExplicitCombinationCandidatesForGroup(group);
            ids.AddRange(realCandidates.Select(BaselineQuants.EncodeTensorConfigGroupSlot));

            ids = ids.Distinct().ToList();

            if (ids.Count == 0)
                ids.Add(BaselineQuants.EncodeTensorConfigGroupSlot(BaselineQuants.GetDefaultExplicitFallbackBaseline()));

            builder.Add(ids.ToArray());
        }

        return builder.ToImmutable();
    }

    public static BigInteger CountCombinations(in BaselineQuants baseQuant)
    {
        var allowed = GetAllowedCandidateIdsPerGroup(baseQuant);

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
        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(baseQuant);

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
