using MagicQuant.Helpers;
using System.Text;
using MagicQuant.Models;
using MQ.DB.Models;

namespace MagicQuant.Services;

public sealed class EffectiveCandidateStateResolverService
{
    private readonly HybridBenchmarkRepository _repository;

    public EffectiveCandidateStateResolverService(HybridBenchmarkRepository repository)
    {
        _repository = repository;
    }

    public async Task<EffectiveStateResolutionResult> ResolveAsync(TensorConfig config, CancellationToken ct = default)
    {
        return await ResolveAsync((HybridQuant)config, config, ct);
    }

    public async Task<EffectiveStateResolutionResult> ResolveAsync(HybridQuant quant, CancellationToken ct = default)
    {
        return await ResolveAsync(quant, (TensorConfig)quant, ct);
    }

    private async Task<EffectiveStateResolutionResult> ResolveAsync(HybridQuant quant, TensorConfig config, CancellationToken ct)
    {
        var warnings = new List<string>();
        var groupStates = new Dictionary<string, string>(StringComparer.Ordinal);

        string baseState = await ResolveBaseStateAsync(quant.BaseQuant, warnings, ct);

        foreach (var (group, storedValue) in TensorConfigIdentity.EnumerateGroupSlots(config))
        {
            string state;
            if (BaselineQuants.IsNullTensorConfigGroupSlot(storedValue))
            {
                state = "base";
            }
            else
            {
                byte baselineId = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedValue);

                if (BaselineQuants.IsNativeExactAlias(baselineId))
                {
                    var exactScheme = BaselineQuants.ResolveExactOverrideScheme(baselineId);
                    state = $"exact:{exactScheme.Names[0]}";
                }
                else
                {
                    var baseline = BaselineQuants.FromId(baselineId);
                    state = await ResolveGroupStateAsync(baseline, group, warnings, ct);
                }
            }

            groupStates[group.Name] = state;
        }

        var keyBuilder = new StringBuilder();
        keyBuilder.Append("base=").Append(baseState);
        foreach (var kv in groupStates.OrderBy(x => x.Key, StringComparer.Ordinal))
            keyBuilder.Append('|').Append(kv.Key).Append('=').Append(kv.Value);

        return new EffectiveStateResolutionResult
        {
            Config = config,
            EffectiveStateKey = keyBuilder.ToString(),
            HasUnknownMappings = warnings.Count > 0,
            Warnings = warnings,
            GroupStates = groupStates,
            BaseState = baseState
        };
    }

    private async Task<string> ResolveBaseStateAsync(BaselineQuants baseline, List<string> warnings, CancellationToken ct)
    {
        if (!baseline.IsExternalRepositoryBaseline)
            return baseline.CanonicalKey;

        var blanket = await _repository.LoadLearnedTensorMappingsAsync(
            canonicalBaselineKey: baseline.CanonicalKey,
            groupId: null,
            preferredSourceScheme: baseline.DefaultTensorScheme,
            allowDominantFallback: true,
            ct: ct);

        if (blanket.Count == 0)
        {
            warnings.Add($"No learned blanket mapping found for external baseline '{baseline.Names[0]}'. Falling back to canonical key.");
            return baseline.CanonicalKey;
        }

        var payload = string.Join("|", blanket.OrderBy(x => x.Key, StringComparer.Ordinal).Select(x => $"{x.Key}={NormalizeOrPreserveRaw(x.Value, warnings)}"));
        return $"{baseline.CanonicalKey}:{TensorConfigIdentity.StableHash(payload)}";
    }

    private async Task<string> ResolveGroupStateAsync(
        BaselineQuants baseline,
        TensorGroup group,
        List<string> warnings,
        CancellationToken ct)
    {
        var mappings = await _repository.LoadLearnedTensorMappingsAsync(
            canonicalBaselineKey: baseline.CanonicalKey,
            groupId: group.UniqueId,
            preferredSourceScheme: baseline.DefaultTensorScheme,
            allowDominantFallback: true,
            ct: ct);

        if (mappings.Count == 0)
        {
            warnings.Add($"No learned mapping found for group '{group.Name}' baseline '{baseline.Names[0]}'. Falling back to requested baseline identity.");
            return $"requested:{baseline.CanonicalKey}";
        }

        var normalized = mappings
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(x => $"{x.Key}={NormalizeOrPreserveRaw(x.Value, warnings)}")
            .ToList();

        if (normalized.Select(x => x.Split('=')[1]).Distinct(StringComparer.Ordinal).Count() == 1)
            return $"effective:{normalized[0].Split('=')[1]}";

        return $"effective-map:{TensorConfigIdentity.StableHash(string.Join("|", normalized))}";
    }

    private static string NormalizeOrPreserveRaw(string raw, List<string> warnings)
    {
        string normalizedRaw = (raw ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(normalizedRaw))
        {
            warnings.Add("Encountered an empty learned tensor state and preserved it as unknown metadata.");
            return "unknown:";
        }

        var resolved = TensorWeightScheme.All.FirstOrDefault(x =>
            x.Names.Any(n => string.Equals(n, normalizedRaw, StringComparison.OrdinalIgnoreCase)));
        if (resolved != null)
            return resolved.Names[0];

        var nativeResolved = NativePrecisionNormalization.ResolveSchemeIdsForLearnedFinalQuantType(normalizedRaw);
        if (nativeResolved.Count > 0)
        {
            var nativeScheme = TensorWeightScheme.All.FirstOrDefault(x => x.UniqueId == nativeResolved.First());
            if (nativeScheme != null)
                return nativeScheme.Names[0];
        }

        warnings.Add($"Unknown or partially unmapped learned tensor state '{normalizedRaw}' was preserved as raw metadata.");
        return $"unknown:{normalizedRaw}";
    }
}
