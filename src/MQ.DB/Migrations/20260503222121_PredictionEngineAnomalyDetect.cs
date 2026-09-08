using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class PredictionEngineAnomalyDetect : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AnomalyInteractionRules",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true),
                    BenchmarkCategory = table.Column<byte>(type: "INTEGER", nullable: false),
                    ReferenceQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
                    ReferenceContextKey = table.Column<string>(type: "TEXT", maxLength: 512, nullable: false),
                    ReferenceEffectiveGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    CandidateEffectiveGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    InactiveGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    FullTensorConfigKey = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    RuleType = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    RuleDirection = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    RuleStatus = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    MovementClassification = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    GroupSetHash = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    GroupCount = table.Column<int>(type: "INTEGER", nullable: false),
                    MeanActualGainVsTwin = table.Column<double>(type: "REAL", nullable: false),
                    BestActualGainVsTwin = table.Column<double>(type: "REAL", nullable: false),
                    MeanPredictionSpaceGap = table.Column<double>(type: "REAL", nullable: false),
                    BestPredictionSpaceGap = table.Column<double>(type: "REAL", nullable: false),
                    AppliedPredictionSpaceAdjustmentKld = table.Column<double>(type: "REAL", nullable: false),
                    EvidenceCount = table.Column<int>(type: "INTEGER", nullable: false),
                    Confidence = table.Column<double>(type: "REAL", nullable: false),
                    ShrinkFactor = table.Column<double>(type: "REAL", nullable: false),
                    Status = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UpdatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    MetadataJson = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AnomalyInteractionRules", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AnomalyInteractionRules_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyInteractionRules_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyInteractionRules_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AnomalyInteractionRules_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "AnomalyProbeSessions",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true),
                    BenchmarkCategory = table.Column<byte>(type: "INTEGER", nullable: false),
                    StartedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    CompletedUtc = table.Column<DateTime>(type: "TEXT", nullable: true),
                    SourceRunLabel = table.Column<string>(type: "TEXT", maxLength: 256, nullable: false),
                    ConfigJson = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AnomalyProbeSessions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeSessions_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeSessions_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeSessions_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeSessions_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "AnomalyInteractionRuleGroupStates",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    RuleId = table.Column<Guid>(type: "TEXT", nullable: false),
                    TensorGroupId = table.Column<byte>(type: "INTEGER", nullable: false),
                    CandidateQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
                    ReferenceQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
                    Movement = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    SortOrder = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AnomalyInteractionRuleGroupStates", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AnomalyInteractionRuleGroupStates_AnomalyInteractionRules_RuleId",
                        column: x => x.RuleId,
                        principalTable: "AnomalyInteractionRules",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "AnomalyProbeObservations",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    SessionId = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true),
                    BenchmarkCategory = table.Column<byte>(type: "INTEGER", nullable: false),
                    ReferenceTensorComboId = table.Column<Guid>(type: "TEXT", nullable: true),
                    ProbeTensorComboId = table.Column<Guid>(type: "TEXT", nullable: true),
                    ProbeType = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    Classification = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    HypothesisLabel = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    MovementClassification = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    ChangedGroupSetHash = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    ChangedGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    CandidateQuantsJson = table.Column<string>(type: "TEXT", nullable: false),
                    ReferenceEffectiveGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    CandidateEffectiveGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    InactiveGroupsJson = table.Column<string>(type: "TEXT", nullable: false),
                    ReferenceTensorConfigKey = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    ProbeTensorConfigKey = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    IsContextualAnomalyProbe = table.Column<bool>(type: "INTEGER", nullable: false),
                    OldBf16Isolation = table.Column<bool>(type: "INTEGER", nullable: false),
                    AllActiveGroupsExplicit = table.Column<bool>(type: "INTEGER", nullable: false),
                    ReferenceQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
                    ActualKld = table.Column<double>(type: "REAL", nullable: false),
                    PredictedKld = table.Column<double>(type: "REAL", nullable: false),
                    ReferenceActualKld = table.Column<double>(type: "REAL", nullable: false),
                    ReferencePredictedKld = table.Column<double>(type: "REAL", nullable: false),
                    ActualGainVsTwin = table.Column<double>(type: "REAL", nullable: false),
                    PredictionSpaceGapVsTwin = table.Column<double>(type: "REAL", nullable: false),
                    SizeSavingsBytes = table.Column<ulong>(type: "INTEGER", nullable: false),
                    UpgradeCount = table.Column<int>(type: "INTEGER", nullable: false),
                    DowngradeCount = table.Column<int>(type: "INTEGER", nullable: false),
                    SameCount = table.Column<int>(type: "INTEGER", nullable: false),
                    UnknownCount = table.Column<int>(type: "INTEGER", nullable: false),
                    NetBitDelta = table.Column<int>(type: "INTEGER", nullable: false),
                    RuleDirection = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    Accepted = table.Column<bool>(type: "INTEGER", nullable: false),
                    FailureCode = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    Message = table.Column<string>(type: "TEXT", maxLength: 4000, nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AnomalyProbeObservations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_AnomalyProbeSessions_SessionId",
                        column: x => x.SessionId,
                        principalTable: "AnomalyProbeSessions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_TensorCombos_ProbeTensorComboId",
                        column: x => x.ProbeTensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_TensorCombos_ReferenceTensorComboId",
                        column: x => x.ReferenceTensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AnomalyProbeObservations_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRuleGroupStates_RuleId_TensorGroupId",
                table: "AnomalyInteractionRuleGroupStates",
                columns: new[] { "RuleId", "TensorGroupId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_AiModelHashId",
                table: "AnomalyInteractionRules",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ReferenceQuantId_GroupSetHash_RuleDirection",
                table: "AnomalyInteractionRules",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "ReferenceQuantId", "GroupSetHash", "RuleDirection" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_RuleDirection_RuleStatus",
                table: "AnomalyInteractionRules",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "RuleDirection", "RuleStatus" });

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_FullTensorConfigKey",
                table: "AnomalyInteractionRules",
                column: "FullTensorConfigKey");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_ImatrixDefinitionId",
                table: "AnomalyInteractionRules",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyInteractionRules_TensorGroupProfileId",
                table: "AnomalyInteractionRules",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_AiModelHashId",
                table: "AnomalyProbeObservations",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ChangedGroupSetHash",
                table: "AnomalyProbeObservations",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "ChangedGroupSetHash" });

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_ReferenceTensorComboId_ProbeTensorComboId_ProbeType",
                table: "AnomalyProbeObservations",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "ReferenceTensorComboId", "ProbeTensorComboId", "ProbeType" });

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ImatrixDefinitionId",
                table: "AnomalyProbeObservations",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ProbeTensorComboId",
                table: "AnomalyProbeObservations",
                column: "ProbeTensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ProbeTensorConfigKey",
                table: "AnomalyProbeObservations",
                column: "ProbeTensorConfigKey");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ReferenceTensorComboId",
                table: "AnomalyProbeObservations",
                column: "ReferenceTensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_ReferenceTensorConfigKey",
                table: "AnomalyProbeObservations",
                column: "ReferenceTensorConfigKey");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_SessionId",
                table: "AnomalyProbeObservations",
                column: "SessionId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeObservations_TensorGroupProfileId",
                table: "AnomalyProbeObservations",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeSessions_AiModelHashId",
                table: "AnomalyProbeSessions",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeSessions_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_BenchmarkCategory_StartedUtc",
                table: "AnomalyProbeSessions",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "BenchmarkCategory", "StartedUtc" });

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeSessions_ImatrixDefinitionId",
                table: "AnomalyProbeSessions",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_AnomalyProbeSessions_TensorGroupProfileId",
                table: "AnomalyProbeSessions",
                column: "TensorGroupProfileId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AnomalyInteractionRuleGroupStates");

            migrationBuilder.DropTable(
                name: "AnomalyProbeObservations");

            migrationBuilder.DropTable(
                name: "AnomalyInteractionRules");

            migrationBuilder.DropTable(
                name: "AnomalyProbeSessions");
        }
    }
}
