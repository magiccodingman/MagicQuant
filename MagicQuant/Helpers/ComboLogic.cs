using System.Numerics;
using MagicQuant.Models;
using System.Collections.Immutable;

namespace MagicQuant.Helpers;

public static class ComboLogic
{
    // Order must match TensorConfig ctor field order
    private static readonly ImmutableArray<TensorGroup> GroupsOrdered =
        TReg.All.OrderBy(g => g.UniqueId).ToImmutableArray();

    public static ImmutableArray<sbyte[]> GetAllowedSchemeIdsPerGroup(BaselineQuants baseQuant)
    {
        bool baseRequiresImatrix = baseQuant.RequiresImatrix;

        var schemesForBase =
            TensorWeightScheme.All
                .Where(s => baseRequiresImatrix || !s.RequiresImatrix)
                .ToImmutableArray();

        if (schemesForBase.IsEmpty)
            throw new InvalidOperationException("No tensor schemes available for this base.");

        var builder = ImmutableArray.CreateBuilder<sbyte[]>();

        foreach (var group in GroupsOrdered)
        {
            var ids =
                schemesForBase
                    .Where(s =>
                        s.BannedGroups.Count == 0 ||
                        !s.BannedGroups.Contains(group))
                    .Select(s => s.UniqueId)
                    .ToArray();

            if (ids.Length == 0)
                throw new InvalidOperationException(
                    $"Group '{group.Name}' has no valid tensor schemes for base '{string.Join("/", baseQuant.Names)}'.");

            builder.Add(ids);
        }

        var result = builder.ToImmutable();

        // 🔒 Absolute safety check (keep this during development)
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] == null)
                throw new InvalidOperationException($"Allowed scheme array at index {i} is null.");
        }

        return result;
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

        foreach (var b in BaselineQuants.All.Where(b => b.BaseConversionBase != null))
            sum += CountForBase(b);

        return sum;
    }
}