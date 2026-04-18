using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class newUpdate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ExecutionPlanProbeCaches",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    HardwareFingerprint = table.Column<string>(type: "TEXT", maxLength: 1024, nullable: false),
                    QuantizedModelFingerprint = table.Column<string>(type: "TEXT", maxLength: 2048, nullable: false),
                    QuantizationKey = table.Column<string>(type: "TEXT", maxLength: 128, nullable: false),
                    DiscoveryTokenTarget = table.Column<int>(type: "INTEGER", nullable: false),
                    StaticNgl = table.Column<int>(type: "INTEGER", nullable: false),
                    UsesGpu = table.Column<bool>(type: "INTEGER", nullable: false),
                    GroupSize = table.Column<int>(type: "INTEGER", nullable: false),
                    SlotsJson = table.Column<string>(type: "TEXT", maxLength: 8000, nullable: false),
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
                });

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_AiModelHashId",
                table: "ExecutionPlanProbeCaches",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_ExecutionPlanProbeCaches_AiModelHashId_HardwareFingerprint_QuantizedModelFingerprint_QuantizationKey_DiscoveryTokenTarget",
                table: "ExecutionPlanProbeCaches",
                columns: new[] { "AiModelHashId", "HardwareFingerprint", "QuantizedModelFingerprint", "QuantizationKey", "DiscoveryTokenTarget" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "ExecutionPlanProbeCaches");
        }
    }
}
