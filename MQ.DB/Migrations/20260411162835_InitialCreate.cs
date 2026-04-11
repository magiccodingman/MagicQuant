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
                name: "TensorCombos",
                columns: table => new
                {
                    Id = table.Column<uint>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
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
                name: "AiBenchmarks",
                columns: table => new
                {
                    Id = table.Column<uint>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Ngl = table.Column<byte>(type: "INTEGER", nullable: false),
                    SizeBytes = table.Column<ulong>(type: "INTEGER", nullable: false),
                    TokensPerSecond = table.Column<double>(type: "REAL", nullable: false),
                    TensorComboId = table.Column<uint>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false)
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
                        name: "FK_AiBenchmarks_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "CategoryBenchmark",
                columns: table => new
                {
                    Id = table.Column<uint>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    AiBenchmarkId = table.Column<uint>(type: "INTEGER", nullable: false),
                    AiBenchmarkId1 = table.Column<uint>(type: "INTEGER", nullable: false),
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
                    table.ForeignKey(
                        name: "FK_CategoryBenchmark_AiBenchmarks_AiBenchmarkId1",
                        column: x => x.AiBenchmarkId1,
                        principalTable: "AiBenchmarks",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_AiModelHashId_TensorComboId",
                table: "AiBenchmarks",
                columns: new[] { "AiModelHashId", "TensorComboId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_AiBenchmarks_TensorComboId",
                table: "AiBenchmarks",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_AiModelHashes_UniqueHash",
                table: "AiModelHashes",
                column: "UniqueHash");

            migrationBuilder.CreateIndex(
                name: "IX_CategoryBenchmark_AiBenchmarkId",
                table: "CategoryBenchmark",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_CategoryBenchmark_AiBenchmarkId1",
                table: "CategoryBenchmark",
                column: "AiBenchmarkId1");

            migrationBuilder.CreateIndex(
                name: "IX_TensorCombos_BaseQuant_Embeddings_LmHead_AttnQ_AttnKV_AttnOutput_FfnUpGate_FfnDown_MoeExperts_MoeRouter",
                table: "TensorCombos",
                columns: new[] { "BaseQuant", "Embeddings", "LmHead", "AttnQ", "AttnKV", "AttnOutput", "FfnUpGate", "FfnDown", "MoeExperts", "MoeRouter" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "CategoryBenchmark");

            migrationBuilder.DropTable(
                name: "AiBenchmarks");

            migrationBuilder.DropTable(
                name: "AiModelHashes");

            migrationBuilder.DropTable(
                name: "TensorCombos");
        }
    }
}
