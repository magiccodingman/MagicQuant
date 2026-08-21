using MagicQuant.Commands;
using MagicQuant.Configuration;
using MagicQuant.Models;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models.DbModels;
using Xunit;

namespace MagicQuant.Tests;

public class QuantizationRunAndBuildHybridsRegressionTests
{
    [Fact]
    public async Task QuantizationRun_PersistsAndLoads_ImatrixDefinitionForeignKey()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "mq-quant-run-fk-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        string? priorMagicQuantDirectory = Cache.MagicQuantDirectory;

        try
        {
            Cache.MagicQuantDirectory = tempRoot;

            await using var db = new MagicQuantContext();

            var model = new AiModelHash
            {
                UniqueHash = "model-" + Guid.NewGuid().ToString("N")
            };

            var combo = new TensorCombo();
            var architecture = new ArchitectureFamily
            {
                NormalizedName = "test-architecture-" + Guid.NewGuid().ToString("N"),
                DisplayName = "Test architecture",
                TensorSignatureHash = "signature-" + Guid.NewGuid().ToString("N"),
                TensorCount = 1
            };
            var profile = new TensorGroupProfile
            {
                ArchitectureFamily = architecture,
                FingerprintHash = "profile-" + Guid.NewGuid().ToString("N"),
                SnapshotJson = "{}"
            };
            db.AiModelHashes.Add(model);
            db.TensorCombos.Add(combo);
            db.ArchitectureFamilies.Add(architecture);
            db.TensorGroupProfiles.Add(profile);
            await db.SaveChangesAsync();

            var imatrix = new ImatrixDefinition
            {
                AiModelHashId = model.Id,
                IdentityHash = "imatrix-" + Guid.NewGuid().ToString("N"),
                SourceKind = "test"
            };
            db.ImatrixDefinitions.Add(imatrix);
            await db.SaveChangesAsync();

            var run = new QuantizationRun
            {
                ArchitectureFamilyId = architecture.Id,
                TensorGroupProfileId = profile.Id,
                AiModelHashId = model.Id,
                ImatrixDefinitionId = imatrix.Id,
                TensorComboId = combo.Id,
                StartedUtc = DateTime.UtcNow.AddSeconds(-1),
                CompletedUtc = DateTime.UtcNow,
                DurationMs = 1000,
                Succeeded = true,
                OutputModelPath = Path.Combine(tempRoot, "output.gguf")
            };
            db.QuantizationRuns.Add(run);
            await db.SaveChangesAsync();

            var loaded = await db.QuantizationRuns
                .Include(x => x.ImatrixDefinition)
                .SingleAsync(x => x.Id == run.Id);

            Assert.Equal(imatrix.Id, loaded.ImatrixDefinitionId);
            Assert.NotNull(loaded.ImatrixDefinition);
            Assert.Equal(imatrix.IdentityHash, loaded.ImatrixDefinition!.IdentityHash);
        }
        finally
        {
            Cache.MagicQuantDirectory = priorMagicQuantDirectory;
            if (Directory.Exists(tempRoot))
                Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public async Task BuildHybrids_RunWithoutModel_RoutesThroughEvolutionValidation()
    {
        var priorConfig = Config.Current;
        try
        {
            Config.Load(MagicQuantYamlConfig.CreateDefault());
            var command = new BuildHybrids();
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => command.Run(new List<CliArg>()));
            Assert.Contains("model directory", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Config.Load(priorConfig);
        }
    }
}
