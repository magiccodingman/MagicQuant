using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AiModelHashes",
                columns: table => new
                {
                    Id = table.Column<uint>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    UniqueHash = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AiModelHashes", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "ArchitectureFamilies",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    NormalizedName = table.Column<string>(type: "TEXT", maxLength: 256, nullable: false),
                    DisplayName = table.Column<string>(type: "TEXT", maxLength: 256, nullable: false),
                    TensorSignatureHash = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    TensorCount = table.Column<int>(type: "INTEGER", nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ArchitectureFamilies", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "TensorCombos",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AttnKV = table.Column<byte>(type: "INTEGER", nullable: false),
                    AttnOutput = table.Column<byte>(type: "INTEGER", nullable: false),
                    AttnQ = table.Column<byte>(type: "INTEGER", nullable: false),
                    BaseQuant = table.Column<byte>(type: "INTEGER", nullable: false),
                    Embeddings = table.Column<byte>(type: "INTEGER", nullable: false),
                    FfnDown = table.Column<byte>(type: "INTEGER", nullable: false),
                    FfnUpGate = table.Column<byte>(type: "INTEGER", nullable: false),
                    LmHead = table.Column<byte>(type: "INTEGER", nullable: false),
                    MoeExperts = table.Column<byte>(type: "INTEGER", nullable: false),
                    MoeRouter = table.Column<byte>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TensorCombos", x => x.Id);
                });

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

            migrationBuilder.CreateTable(
                name: "ArchitectureFamilyModelHashes",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    IsCanonical = table.Column<bool>(type: "INTEGER", nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ArchitectureFamilyModelHashes", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ArchitectureFamilyModelHashes_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_ArchitectureFamilyModelHashes_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "BaselineQuantDefinitions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: true),
                    RuntimeBaselineId = table.Column<byte>(type: "INTEGER", nullable: false),
                    CanonicalKey = table.Column<string>(type: "TEXT", maxLength: 512, nullable: false),
                    NormalizedCanonicalKey = table.Column<string>(type: "TEXT", maxLength: 512, nullable: false),
                    BaselineName = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    DisplayName = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    QuantizeBaseArgumentName = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    DefaultTensorSchemeId = table.Column<byte>(type: "INTEGER", nullable: false),
                    DefaultTensorSchemeName = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    SourceKind = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    SourceOwner = table.Column<string>(type: "TEXT", maxLength: 128, nullable: true),
                    SourceRepository = table.Column<string>(type: "TEXT", maxLength: 256, nullable: true),
                    NormalizedSourceRepository = table.Column<string>(type: "TEXT", maxLength: 256, nullable: true),
                    SourceFileName = table.Column<string>(type: "TEXT", maxLength: 512, nullable: true),
                    NormalizedSourceFileName = table.Column<string>(type: "TEXT", maxLength: 512, nullable: true),
                    ShortSourceName = table.Column<string>(type: "TEXT", maxLength: 64, nullable: true),
                    BaselineFamily = table.Column<string>(type: "TEXT", maxLength: 128, nullable: true),
                    IsCustomBaseline = table.Column<bool>(type: "INTEGER", nullable: false),
                    IsLearningBaseline = table.Column<bool>(type: "INTEGER", nullable: false),
                    IsCombinationCarrierCandidate = table.Column<bool>(type: "INTEGER", nullable: false),
                    IsExplicitGroupCombinationCandidate = table.Column<bool>(type: "INTEGER", nullable: false),
                    RequiresImatrix = table.Column<bool>(type: "INTEGER", nullable: false),
                    BitRange = table.Column<byte>(type: "INTEGER", nullable: false),
                    ExplicitCandidateSortOrder = table.Column<int>(type: "INTEGER", nullable: false),
                    IsActiveInCurrentConfig = table.Column<bool>(type: "INTEGER", nullable: false),
                    FirstSeenUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    LastSeenUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    LastUpdatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_BaselineQuantDefinitions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_BaselineQuantDefinitions_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "TensorGroupProfiles",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    FingerprintHash = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    SnapshotJson = table.Column<string>(type: "TEXT", nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    IsActive = table.Column<bool>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TensorGroupProfiles", x => x.Id);
                    table.ForeignKey(
                        name: "FK_TensorGroupProfiles_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "AiBenchmarks",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    Ngl = table.Column<byte>(type: "INTEGER", nullable: false),
                    SizeBytes = table.Column<ulong>(type: "INTEGER", nullable: false),
                    TokensPerSecond = table.Column<double>(type: "REAL", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorComboId = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AiBenchmarks", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AiBenchmarks_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AiBenchmarks_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AiBenchmarks_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AiBenchmarks_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AiBenchmarks_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "ExecutionPlanProbeCaches",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true),
                    HardwareFingerprint = table.Column<string>(type: "TEXT", maxLength: 1024, nullable: false),
                    QuantizedModelFingerprint = table.Column<string>(type: "TEXT", maxLength: 2048, nullable: false),
                    QuantizationKey = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    DiscoveryTokenTarget = table.Column<int>(type: "INTEGER", nullable: false),
                    StaticNgl = table.Column<int>(type: "INTEGER", nullable: false),
                    UsesGpu = table.Column<bool>(type: "INTEGER", nullable: false),
                    GroupSize = table.Column<int>(type: "INTEGER", nullable: false),
                    SlotsJson = table.Column<string>(type: "TEXT", maxLength: 8000, nullable: false),
                    ProbeSchemaVersion = table.Column<int>(type: "INTEGER", nullable: false),
                    Q8ModelSizeBytes = table.Column<ulong>(type: "INTEGER", nullable: false),
                    Q8StableNgl = table.Column<int>(type: "INTEGER", nullable: false),
                    NativeModelSizeBytes = table.Column<ulong>(type: "INTEGER", nullable: false),
                    NativeStableNgl = table.Column<int>(type: "INTEGER", nullable: false),
                    NativeQuantizationKey = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    MaxCandidateNgl = table.Column<int>(type: "INTEGER", nullable: false),
                    GpuMemoryLimitsJson = table.Column<string>(type: "TEXT", maxLength: 4000, nullable: false),
                    TensorSplitJson = table.Column<string>(type: "TEXT", maxLength: 4000, nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UpdatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ExecutionPlanProbeCaches", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ExecutionPlanProbeCaches_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_ExecutionPlanProbeCaches_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_ExecutionPlanProbeCaches_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_ExecutionPlanProbeCaches_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "AiBenchmarkLearnedSources",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorComboId = table.Column<Guid>(type: "TEXT", nullable: false),
                    TensorGroupId = table.Column<byte>(type: "INTEGER", nullable: false),
                    BaselineQuantDefinitionId = table.Column<int>(type: "INTEGER", nullable: false),
                    SourceLearningBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: true),
                    BaselineCanonicalKey = table.Column<string>(type: "TEXT", maxLength: 512, nullable: false),
                    CreatedUtc = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AiBenchmarkLearnedSources", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AiBenchmarkLearnedSources_AiBenchmarks_AiBenchmarkId",
                        column: x => x.AiBenchmarkId,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AiBenchmarkLearnedSources_AiBenchmarks_SourceLearningBenchmarkId",
                        column: x => x.SourceLearningBenchmarkId,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                    table.ForeignKey(
                        name: "FK_AiBenchmarkLearnedSources_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_AiBenchmarkLearnedSources_BaselineQuantDefinitions_BaselineQuantDefinitionId",
                        column: x => x.BaselineQuantDefinitionId,
                        principalTable: "BaselineQuantDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AiBenchmarkLearnedSources_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AiBenchmarkLearnedSources_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "CategoryBenchmark",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Category = table.Column<byte>(type: "INTEGER", nullable: false),
                    Kld = table.Column<double>(type: "REAL", nullable: false),
                    Ppl = table.Column<double>(type: "REAL", nullable: false),
                    PplError = table.Column<double>(type: "REAL", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CategoryBenchmark", x => x.Id);
                    table.ForeignKey(
                        name: "FK_CategoryBenchmark_AiBenchmarks_AiBenchmarkId",
                        column: x => x.AiBenchmarkId,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "LearnedBaselineTensorQuants",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    BaselineQuantDefinitionId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorComboId = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    BaselineQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
                    BaselineCanonicalKey = table.Column<string>(type: "TEXT", maxLength: 512, nullable: false),
                    BaselineSourceKind = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    BaselineSourceRepository = table.Column<string>(type: "TEXT", maxLength: 256, nullable: true),
                    BaselineSourceFileName = table.Column<string>(type: "TEXT", maxLength: 512, nullable: true),
                    TensorWeightSchemeId = table.Column<byte>(type: "INTEGER", nullable: false),
                    TensorGroupId = table.Column<byte>(type: "INTEGER", nullable: false),
                    TensorName = table.Column<string>(type: "TEXT", maxLength: 512, nullable: false),
                    FinalQuantType = table.Column<string>(type: "TEXT", maxLength: 32, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_LearnedBaselineTensorQuants", x => x.Id);
                    table.ForeignKey(
                        name: "FK_LearnedBaselineTensorQuants_AiBenchmarks_AiBenchmarkId",
                        column: x => x.AiBenchmarkId,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_LearnedBaselineTensorQuants_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_LearnedBaselineTensorQuants_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_LearnedBaselineTensorQuants_BaselineQuantDefinitions_BaselineQuantDefinitionId",
                        column: x => x.BaselineQuantDefinitionId,
                        principalTable: "BaselineQuantDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_LearnedBaselineTensorQuants_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_LearnedBaselineTensorQuants_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "QuantizationRuns",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true),
                    TensorComboId = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: true),
                    StartedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    CompletedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    DurationMs = table.Column<long>(type: "INTEGER", nullable: false),
                    Succeeded = table.Column<bool>(type: "INTEGER", nullable: false),
                    Error = table.Column<string>(type: "TEXT", maxLength: 4000, nullable: true),
                    OutputModelPath = table.Column<string>(type: "TEXT", maxLength: 2048, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_QuantizationRuns", x => x.Id);
                    table.ForeignKey(
                        name: "FK_QuantizationRuns_AiBenchmarks_AiBenchmarkId",
                        column: x => x.AiBenchmarkId,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                    table.ForeignKey(
                        name: "FK_QuantizationRuns_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_QuantizationRuns_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_QuantizationRuns_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_QuantizationRuns_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_QuantizationRuns_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "BenchmarkRuns",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ArchitectureFamilyId = table.Column<int>(type: "INTEGER", nullable: false),
                    TensorGroupProfileId = table.Column<int>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    ImatrixDefinitionId = table.Column<int>(type: "INTEGER", nullable: true),
                    TensorComboId = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: false),
                    CategoryBenchmarkId = table.Column<Guid>(type: "TEXT", nullable: true),
                    Category = table.Column<byte>(type: "INTEGER", nullable: false),
                    StartedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    CompletedUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    DurationMs = table.Column<long>(type: "INTEGER", nullable: false),
                    Succeeded = table.Column<bool>(type: "INTEGER", nullable: false),
                    Error = table.Column<string>(type: "TEXT", maxLength: 4000, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_BenchmarkRuns", x => x.Id);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_AiBenchmarks_AiBenchmarkId",
                        column: x => x.AiBenchmarkId,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_AiModelHashes_AiModelHashId",
                        column: x => x.AiModelHashId,
                        principalTable: "AiModelHashes",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_ArchitectureFamilies_ArchitectureFamilyId",
                        column: x => x.ArchitectureFamilyId,
                        principalTable: "ArchitectureFamilies",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_CategoryBenchmark_CategoryBenchmarkId",
                        column: x => x.CategoryBenchmarkId,
                        principalTable: "CategoryBenchmark",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_ImatrixDefinitions_ImatrixDefinitionId",
                        column: x => x.ImatrixDefinitionId,
                        principalTable: "ImatrixDefinitions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_TensorGroupProfiles_TensorGroupProfileId",
                        column: x => x.TensorGroupProfileId,
                        principalTable: "TensorGroupProfiles",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarkLearnedSources_AiBenchmarkId_TensorGroupId",
                table: "AiBenchmarkLearnedSources",
                columns: new[] { "AiBenchmarkId", "TensorGroupId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarkLearnedSources_ArchitectureFamilyId_TensorGroupProfileId_BaselineQuantDefinitionId",
                table: "AiBenchmarkLearnedSources",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "BaselineQuantDefinitionId" });

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarkLearnedSources_BaselineQuantDefinitionId",
                table: "AiBenchmarkLearnedSources",
                column: "BaselineQuantDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarkLearnedSources_SourceLearningBenchmarkId",
                table: "AiBenchmarkLearnedSources",
                column: "SourceLearningBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarkLearnedSources_TensorComboId",
                table: "AiBenchmarkLearnedSources",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarkLearnedSources_TensorGroupProfileId",
                table: "AiBenchmarkLearnedSources",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_AiModelHashId",
                table: "AiBenchmarks",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_TensorComboId",
                table: "AiBenchmarks",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "TensorComboId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_ArchitectureFamilyId_TensorGroupProfileId_TensorComboId",
                table: "AiBenchmarks",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "TensorComboId" });

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_ImatrixDefinitionId",
                table: "AiBenchmarks",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_TensorComboId",
                table: "AiBenchmarks",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_TensorGroupProfileId",
                table: "AiBenchmarks",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_AiModelHashes_UniqueHash",
                table: "AiModelHashes",
                column: "UniqueHash");

            migrationBuilder.CreateIndex(
                name: "IX_ArchitectureFamilies_NormalizedName",
                table: "ArchitectureFamilies",
                column: "NormalizedName",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_ArchitectureFamilies_TensorSignatureHash_TensorCount",
                table: "ArchitectureFamilies",
                columns: new[] { "TensorSignatureHash", "TensorCount" });

            migrationBuilder.CreateIndex(
                name: "IX_ArchitectureFamilyModelHashes_AiModelHashId",
                table: "ArchitectureFamilyModelHashes",
                column: "AiModelHashId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_ArchitectureFamilyModelHashes_ArchitectureFamilyId_AiModelHashId",
                table: "ArchitectureFamilyModelHashes",
                columns: new[] { "ArchitectureFamilyId", "AiModelHashId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_ArchitectureFamilyId_NormalizedCanonicalKey",
                table: "BaselineQuantDefinitions",
                columns: new[] { "ArchitectureFamilyId", "NormalizedCanonicalKey" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_ArchitectureFamilyId_NormalizedSourceRepository_NormalizedSourceFileName",
                table: "BaselineQuantDefinitions",
                columns: new[] { "ArchitectureFamilyId", "NormalizedSourceRepository", "NormalizedSourceFileName" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_ArchitectureFamilyId_RuntimeBaselineId",
                table: "BaselineQuantDefinitions",
                columns: new[] { "ArchitectureFamilyId", "RuntimeBaselineId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_IsActiveInCurrentConfig",
                table: "BaselineQuantDefinitions",
                column: "IsActiveInCurrentConfig");

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_RuntimeBaselineId_ArchitectureFamilyId",
                table: "BaselineQuantDefinitions",
                columns: new[] { "RuntimeBaselineId", "ArchitectureFamilyId" });

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_AiBenchmarkId",
                table: "BenchmarkRuns",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_AiBenchmarkId_Category",
                table: "BenchmarkRuns",
                columns: new[] { "AiBenchmarkId", "Category" });

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_AiModelHashId",
                table: "BenchmarkRuns",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_ArchitectureFamilyId",
                table: "BenchmarkRuns",
                column: "ArchitectureFamilyId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_CategoryBenchmarkId",
                table: "BenchmarkRuns",
                column: "CategoryBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_ImatrixDefinitionId",
                table: "BenchmarkRuns",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_StartedUtc",
                table: "BenchmarkRuns",
                column: "StartedUtc");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_TensorComboId",
                table: "BenchmarkRuns",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_TensorGroupProfileId",
                table: "BenchmarkRuns",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_CategoryBenchmark_AiBenchmarkId",
                table: "CategoryBenchmark",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_AiModelHashId",
                table: "ExecutionPlanProbeCaches",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_ArchitectureFamilyId",
                table: "ExecutionPlanProbeCaches",
                column: "ArchitectureFamilyId");

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_ArchitectureFamilyId_TensorGroupProfileId_AiModelHashId_ImatrixDefinitionId_HardwareFingerprint_QuantizedModelFingerprint_QuantizationKey_DiscoveryTokenTarget",
                table: "ExecutionPlanProbeCaches",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "AiModelHashId", "ImatrixDefinitionId", "HardwareFingerprint", "QuantizedModelFingerprint", "QuantizationKey", "DiscoveryTokenTarget" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_ImatrixDefinitionId",
                table: "ExecutionPlanProbeCaches",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_TensorGroupProfileId",
                table: "ExecutionPlanProbeCaches",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_ImatrixDefinitions_AiModelHashId_IdentityHash",
                table: "ImatrixDefinitions",
                columns: new[] { "AiModelHashId", "IdentityHash" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_AiBenchmarkId",
                table: "LearnedBaselineTensorQuants",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_AiModelHashId",
                table: "LearnedBaselineTensorQuants",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_ArchitectureFamilyId_TensorGroupProfileId_BaselineQuantDefinitionId_TensorGroupId",
                table: "LearnedBaselineTensorQuants",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "BaselineQuantDefinitionId", "TensorGroupId" });

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_ArchitectureFamilyId_TensorGroupProfileId_BaselineQuantDefinitionId_TensorWeightSchemeId_TensorName",
                table: "LearnedBaselineTensorQuants",
                columns: new[] { "ArchitectureFamilyId", "TensorGroupProfileId", "BaselineQuantDefinitionId", "TensorWeightSchemeId", "TensorName" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_BaselineQuantDefinitionId",
                table: "LearnedBaselineTensorQuants",
                column: "BaselineQuantDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_TensorComboId",
                table: "LearnedBaselineTensorQuants",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_TensorGroupProfileId",
                table: "LearnedBaselineTensorQuants",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_AiBenchmarkId",
                table: "QuantizationRuns",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_AiModelHashId",
                table: "QuantizationRuns",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_ArchitectureFamilyId",
                table: "QuantizationRuns",
                column: "ArchitectureFamilyId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_ImatrixDefinitionId",
                table: "QuantizationRuns",
                column: "ImatrixDefinitionId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_StartedUtc",
                table: "QuantizationRuns",
                column: "StartedUtc");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_TensorComboId",
                table: "QuantizationRuns",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_TensorGroupProfileId",
                table: "QuantizationRuns",
                column: "TensorGroupProfileId");

            migrationBuilder.CreateIndex(
                name: "IX_TensorCombos_BaseQuant_Embeddings_LmHead_AttnQ_AttnKV_AttnOutput_FfnUpGate_FfnDown_MoeExperts_MoeRouter",
                table: "TensorCombos",
                columns: new[] { "BaseQuant", "Embeddings", "LmHead", "AttnQ", "AttnKV", "AttnOutput", "FfnUpGate", "FfnDown", "MoeExperts", "MoeRouter" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TensorGroupProfiles_ArchitectureFamilyId_FingerprintHash",
                table: "TensorGroupProfiles",
                columns: new[] { "ArchitectureFamilyId", "FingerprintHash" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TensorGroupProfiles_ArchitectureFamilyId_IsActive",
                table: "TensorGroupProfiles",
                columns: new[] { "ArchitectureFamilyId", "IsActive" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AiBenchmarkLearnedSources");

            migrationBuilder.DropTable(
                name: "ArchitectureFamilyModelHashes");

            migrationBuilder.DropTable(
                name: "BenchmarkRuns");

            migrationBuilder.DropTable(
                name: "ExecutionPlanProbeCaches");

            migrationBuilder.DropTable(
                name: "LearnedBaselineTensorQuants");

            migrationBuilder.DropTable(
                name: "QuantizationRuns");

            migrationBuilder.DropTable(
                name: "CategoryBenchmark");

            migrationBuilder.DropTable(
                name: "BaselineQuantDefinitions");

            migrationBuilder.DropTable(
                name: "AiBenchmarks");

            migrationBuilder.DropTable(
                name: "ImatrixDefinitions");

            migrationBuilder.DropTable(
                name: "TensorCombos");

            migrationBuilder.DropTable(
                name: "TensorGroupProfiles");

            migrationBuilder.DropTable(
                name: "AiModelHashes");

            migrationBuilder.DropTable(
                name: "ArchitectureFamilies");
        }
    }
}
