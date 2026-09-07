using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class PredictionEngineAnomalyDetect2 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_AnomalyInteractionRules_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ReferenceQuantId_GroupSetHash_RuleDirection",
                table: "AnomalyInteractionRules");

            migrationBuilder.AddColumn<string>(
                name: "ProbeDisplayName",
                table: "AnomalyProbeObservations",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ProbeInternalName",
                table: "AnomalyProbeObservations",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ProbePlanClass",
                table: "AnomalyProbeObservations",
                type: "TEXT",
                maxLength: 64,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ReferenceDisplayName",
                table: "AnomalyProbeObservations",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ReferenceInternalName",
                table: "AnomalyProbeObservations",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "SeedClass",
                table: "AnomalyProbeObservations",
                type: "TEXT",
                maxLength: 64,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "SeedPriority",
                table: "AnomalyProbeObservations",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "CandidateDisplayName",
                table: "AnomalyInteractionRules",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "CandidateInternalName",
                table: "AnomalyInteractionRules",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ReferenceDisplayName",
                table: "AnomalyInteractionRules",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "ReferenceInternalName",
                table: "AnomalyInteractionRules",
                type: "TEXT",
                maxLength: 512,
                nullable: false,
                defaultValue: "");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ReferenceQuantId_ReferenceContextKey_GroupSetHash_RuleDirection",
                table: "AnomalyInteractionRules",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "ReferenceQuantId", "ReferenceContextKey", "GroupSetHash", "RuleDirection" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_AnomalyInteractionRules_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ReferenceQuantId_ReferenceContextKey_GroupSetHash_RuleDirection",
                table: "AnomalyInteractionRules");

            migrationBuilder.DropColumn(
                name: "ProbeDisplayName",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "ProbeInternalName",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "ProbePlanClass",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "ReferenceDisplayName",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "ReferenceInternalName",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "SeedClass",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "SeedPriority",
                table: "AnomalyProbeObservations");

            migrationBuilder.DropColumn(
                name: "CandidateDisplayName",
                table: "AnomalyInteractionRules");

            migrationBuilder.DropColumn(
                name: "CandidateInternalName",
                table: "AnomalyInteractionRules");

            migrationBuilder.DropColumn(
                name: "ReferenceDisplayName",
                table: "AnomalyInteractionRules");

            migrationBuilder.DropColumn(
                name: "ReferenceInternalName",
                table: "AnomalyInteractionRules");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ReferenceQuantId_GroupSetHash_RuleDirection",
                table: "AnomalyInteractionRules",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "ReferenceQuantId", "GroupSetHash", "RuleDirection" },
                unique: true);
        }
    }
}
