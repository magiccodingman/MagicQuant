namespace MQ.DB.Models;

public static class IsolationRules
{
    /// <summary>
    /// If the smallest explicit non-imatrix probe for a tensor group cannot shrink
    /// the total model by at least this much versus the carrier-base-only sample,
    /// we stop isolated explicit quant exploration for that group for this run.
    /// </summary>
    public const double MinimumIsolationReductionToContinue = 0.04d;

    /// <summary>
    /// If a base-only baseline sample only shrinks the model by less than this versus
    /// the native BF16/F16/F32 source, it can be disabled as a future combination baseline.
    /// </summary>
    public const double MinimumMeaningfulBaseOnlyReductionRatio = 0.01d;

    /// <summary>
    /// Hard damage cutoff for isolated tensor options.
    /// 5% = 0.05 ratio.
    /// </summary>
    public const double MaximumIsolationPplDeltaRatio = 0.05d;

    /// <summary>
    /// Hard KLD cutoff for isolated tensor options.
    /// </summary>
    public const double MaximumIsolationKld = 0.10d;

    /// <summary>
    /// Used when comparing floating-point metrics so near-identical values do not
    /// cause unstable eliminations.
    /// </summary>
    public const double MetricComparisonEpsilon = 1e-9d;
}