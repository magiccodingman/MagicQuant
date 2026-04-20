using MagicQuant.Helpers;
using MQ.DB.Models;

namespace MagicQuant.Services;

/// <summary>
/// Centralized authority for isolation probe planning in baseline-family candidate space.
/// </summary>
public sealed class IsolationPlanningService
{
    public RequiredSampleGenerationResult BuildInitialPlan(List<TensorGroup>? missingTensorGroups = null)
        => TensorConfigGenerator.GenerateInitialIsolationSamplePlan(missingTensorGroups);

    public RequiredSampleGenerationResult BuildContinuationPlan(IEnumerable<byte> groupIdsToContinue, List<TensorGroup>? missingTensorGroups = null)
        => TensorConfigGenerator.GenerateContinuationIsolationSamplePlan(groupIdsToContinue, missingTensorGroups);

    public List<HybridQuant> BuildRequiredStartupCombos(List<TensorGroup>? missingTensorGroups = null)
        => TensorConfigGenerator.GenerateRequiredDataSampleCombos(missingTensorGroups);
}
