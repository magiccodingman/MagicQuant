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
        var candidatesForRun = BaselineQuants.GetGroupCombinationCandidates(imatrixAvailable, allowHighPrecisionHybrids: false)
            .ToImmutableArray();

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

            // Strict policy (Option A):
            // - normal explicit hybrid families come only from GetGroupCombinationCandidates(..., false)
            // - BF16/F16 are injected only here and only when AllowHighPrecisionHybrids is enabled
            if (RuntimeSearchSpace.AllowHighPrecisionHybrids && !RuntimeSearchSpace.IsBf16TensorChoiceSuppressed(group))
            {
                ids.Add(BaselineQuants.BF16_Hybrid.UniqueId);
                ids.Add(BaselineQuants.F16_Hybrid.UniqueId);
            }

            foreach (var candidate in candidatesForRun)
            {
                if (RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, candidate))
                    continue;

                ids.Add(candidate.UniqueId);
            }

            ids = ids.Distinct().OrderBy(x => x).ToList();

            if (ids.Count == 0)
            {
                ids.Add(BaselineQuants.GetDefaultExplicitFallbackBaseline().UniqueId);
            }

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
