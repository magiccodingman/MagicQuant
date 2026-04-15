using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    public partial class AddLearnedBaselineTables : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "BaselineQuantDefinitions",
                columns: table => new
                {
                    BaselineQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
                    BaselineName = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false),
                    DefaultTensorSchemeId = table.Column<byte>(type: "INTEGER", nullable: false),
                    DefaultTensorSchemeName = table.Column<string>(type: "TEXT", maxLength: 64, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_BaselineQuantDefinitions", x => x.BaselineQuantId);
                });

            migrationBuilder.CreateTable(
                name: "LearnedBaselineTensorQuants",
                columns: table => new
                {
                    Id = table.Column<ulong>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    AiBenchmarkId = table.Column<uint>(type: "INTEGER", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    BaselineQuantId = table.Column<byte>(type: "INTEGER", nullable: false),
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
                });

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_BaselineName",
                table: "BaselineQuantDefinitions",
                column: "BaselineName",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_DefaultTensorSchemeId",
                table: "BaselineQuantDefinitions",
                column: "DefaultTensorSchemeId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_BaselineQuantDefinitions_DefaultTensorSchemeName",
                table: "BaselineQuantDefinitions",
                column: "DefaultTensorSchemeName",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_AiBenchmarkId",
                table: "LearnedBaselineTensorQuants",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_AiModelHashId_BaselineQuantId_TensorWeightSchemeId_TensorGroupId",
                table: "LearnedBaselineTensorQuants",
                columns: new[] { "AiModelHashId", "BaselineQuantId", "TensorWeightSchemeId", "TensorGroupId" });

            migrationBuilder.CreateIndex(
                name: "IX_LearnedBaselineTensorQuants_AiModelHashId_BaselineQuantId_TensorWeightSchemeId_TensorName",
                table: "LearnedBaselineTensorQuants",
                columns: new[] { "AiModelHashId", "BaselineQuantId", "TensorWeightSchemeId", "TensorName" },
                unique: true);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "LearnedBaselineTensorQuants");

            migrationBuilder.DropTable(
                name: "BaselineQuantDefinitions");
        }
    }
}
