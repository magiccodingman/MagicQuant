using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    public partial class AddImatrixContextIdentity : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ImatrixDefinitions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    IdentityHash = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    CanonicalPath = table.Column<string>(type: "TEXT", maxLength: 2048, nullable: true),
                    SourceKind = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    MetadataJson = table.Column<string>(type: "TEXT", maxLength: 8000, nullable: true),
                    TokenCount = table.Column<int>(type: "INTEGER", nullable: true),
                    BuildFingerprint = table.Column<string>(type: "TEXT", maxLength: 512, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ImatrixDefinitions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ImatrixDefinitions_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.AddColumn<int>(name: "ImatrixDefinitionId", table: "AiBenchmarks", type: "INTEGER", nullable: true);
            migrationBuilder.AddColumn<int>(name: "ImatrixDefinitionId", table: "QuantizationRuns", type: "INTEGER", nullable: true);
            migrationBuilder.AddColumn<int>(name: "ImatrixDefinitionId", table: "BenchmarkRuns", type: "INTEGER", nullable: true);
            migrationBuilder.AddColumn<int>(name: "ImatrixDefinitionId", table: "ExecutionPlanProbeCaches", type: "INTEGER", nullable: true);

            migrationBuilder.CreateIndex(name: "IX_ImatrixDefinitions_AiModelHashId_IdentityHash", table: "ImatrixDefinitions", columns: new[] { "AiModelHashId", "IdentityHash" }, unique: true);
            migrationBuilder.CreateIndex(name: "IX_AiBenchmarks_ImatrixDefinitionId", table: "AiBenchmarks", column: "ImatrixDefinitionId");
            migrationBuilder.CreateIndex(name: "IX_QuantizationRuns_ImatrixDefinitionId", table: "QuantizationRuns", column: "ImatrixDefinitionId");
            migrationBuilder.CreateIndex(name: "IX_BenchmarkRuns_ImatrixDefinitionId", table: "BenchmarkRuns", column: "ImatrixDefinitionId");
            migrationBuilder.CreateIndex(name: "IX_ExecutionPlanProbeCaches_ImatrixDefinitionId", table: "ExecutionPlanProbeCaches", column: "ImatrixDefinitionId");

            migrationBuilder.DropIndex(name: "IX_AiBenchmarks_AiModelHashId_TensorComboId", table: "AiBenchmarks");
            migrationBuilder.CreateIndex(name: "IX_AiBenchmarks_AiModelHashId_ImatrixDefinitionId_TensorComboId", table: "AiBenchmarks", columns: new[] { "AiModelHashId", "ImatrixDefinitionId", "TensorComboId" }, unique: true);

            migrationBuilder.DropIndex(name: "IX_ExecutionPlanProbeCaches_AiModelHashId_HardwareFingerprint_QuantizedModelFingerprint_QuantizationKey_DiscoveryTokenTarget", table: "ExecutionPlanProbeCaches");
            migrationBuilder.CreateIndex(name: "IX_ExecutionPlanProbeCaches_AiModelHashId_ImatrixDefinitionId_HardwareFingerprint_QuantizedModelFingerprint_QuantizationKey_DiscoveryTokenTarget", table: "ExecutionPlanProbeCaches", columns: new[] { "AiModelHashId", "ImatrixDefinitionId", "HardwareFingerprint", "QuantizedModelFingerprint", "QuantizationKey", "DiscoveryTokenTarget" }, unique: true);

            migrationBuilder.AddForeignKey(name: "FK_AiBenchmarks_ImatrixDefinitions_ImatrixDefinitionId", table: "AiBenchmarks", column: "ImatrixDefinitionId", principalTable: "ImatrixDefinitions", principalColumn: "Id", onDelete: ReferentialAction.Restrict);
            migrationBuilder.AddForeignKey(name: "FK_QuantizationRuns_ImatrixDefinitions_ImatrixDefinitionId", table: "QuantizationRuns", column: "ImatrixDefinitionId", principalTable: "ImatrixDefinitions", principalColumn: "Id", onDelete: ReferentialAction.Restrict);
            migrationBuilder.AddForeignKey(name: "FK_BenchmarkRuns_ImatrixDefinitions_ImatrixDefinitionId", table: "BenchmarkRuns", column: "ImatrixDefinitionId", principalTable: "ImatrixDefinitions", principalColumn: "Id", onDelete: ReferentialAction.Restrict);
            migrationBuilder.AddForeignKey(name: "FK_ExecutionPlanProbeCaches_ImatrixDefinitions_ImatrixDefinitionId", table: "ExecutionPlanProbeCaches", column: "ImatrixDefinitionId", principalTable: "ImatrixDefinitions", principalColumn: "Id", onDelete: ReferentialAction.Restrict);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(name: "FK_AiBenchmarks_ImatrixDefinitions_ImatrixDefinitionId", table: "AiBenchmarks");
            migrationBuilder.DropForeignKey(name: "FK_QuantizationRuns_ImatrixDefinitions_ImatrixDefinitionId", table: "QuantizationRuns");
            migrationBuilder.DropForeignKey(name: "FK_BenchmarkRuns_ImatrixDefinitions_ImatrixDefinitionId", table: "BenchmarkRuns");
            migrationBuilder.DropForeignKey(name: "FK_ExecutionPlanProbeCaches_ImatrixDefinitions_ImatrixDefinitionId", table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropTable(name: "ImatrixDefinitions");

            migrationBuilder.DropIndex(name: "IX_AiBenchmarks_AiModelHashId_ImatrixDefinitionId_TensorComboId", table: "AiBenchmarks");
            migrationBuilder.DropIndex(name: "IX_ExecutionPlanProbeCaches_AiModelHashId_ImatrixDefinitionId_HardwareFingerprint_QuantizedModelFingerprint_QuantizationKey_DiscoveryTokenTarget", table: "ExecutionPlanProbeCaches");
            migrationBuilder.DropIndex(name: "IX_AiBenchmarks_ImatrixDefinitionId", table: "AiBenchmarks");
            migrationBuilder.DropIndex(name: "IX_QuantizationRuns_ImatrixDefinitionId", table: "QuantizationRuns");
            migrationBuilder.DropIndex(name: "IX_BenchmarkRuns_ImatrixDefinitionId", table: "BenchmarkRuns");
            migrationBuilder.DropIndex(name: "IX_ExecutionPlanProbeCaches_ImatrixDefinitionId", table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(name: "ImatrixDefinitionId", table: "AiBenchmarks");
            migrationBuilder.DropColumn(name: "ImatrixDefinitionId", table: "QuantizationRuns");
            migrationBuilder.DropColumn(name: "ImatrixDefinitionId", table: "BenchmarkRuns");
            migrationBuilder.DropColumn(name: "ImatrixDefinitionId", table: "ExecutionPlanProbeCaches");

            migrationBuilder.CreateIndex(name: "IX_AiBenchmarks_AiModelHashId_TensorComboId", table: "AiBenchmarks", columns: new[] { "AiModelHashId", "TensorComboId" }, unique: true);
            migrationBuilder.CreateIndex(name: "IX_ExecutionPlanProbeCaches_AiModelHashId_HardwareFingerprint_QuantizedModelFingerprint_QuantizationKey_DiscoveryTokenTarget", table: "ExecutionPlanProbeCaches", columns: new[] { "AiModelHashId", "HardwareFingerprint", "QuantizedModelFingerprint", "QuantizationKey", "DiscoveryTokenTarget" }, unique: true);
        }
    }
}
