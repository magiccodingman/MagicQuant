using MagicQuant.Commands;
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

        Cache.MagicQuantDirectory = tempRoot;

        await using var db = new MagicQuantContext();

        var model = new AiModelHash
        {
            UniqueHash = "model-" + Guid.NewGuid().ToString("N")
        };

        var combo = new TensorCombo();
        db.AiModelHashes.Add(model);
        db.TensorCombos.Add(combo);
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

    [Fact]
    public async Task BuildHybrids_RunWithoutHelp_ThrowsNotImplementedException()
    {
        var command = new BuildHybrids();
        var ex = await Assert.ThrowsAsync<NotImplementedException>(() => command.Run(new List<CliArg>()));
        Assert.Contains("disabled", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
