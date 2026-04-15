using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class AddExecutionTimingTables : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "BenchmarkRuns",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    TensorComboId = table.Column<uint>(type: "INTEGER", nullable: false),
                    AiBenchmarkId = table.Column<uint>(type: "INTEGER", nullable: false),
                    CategoryBenchmarkId = table.Column<uint>(type: "INTEGER", nullable: true),
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
                        name: "FK_BenchmarkRuns_CategoryBenchmark_CategoryBenchmarkId",
                        column: x => x.CategoryBenchmarkId,
                        principalTable: "CategoryBenchmark",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                    table.ForeignKey(
                        name: "FK_BenchmarkRuns_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateTable(
                name: "QuantizationRuns",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    AiModelHashId = table.Column<uint>(type: "INTEGER", nullable: false),
                    TensorComboId = table.Column<uint>(type: "INTEGER", nullable: false),
                    AiBenchmarkId = table.Column<uint>(type: "INTEGER", nullable: true),
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
                        name: "FK_QuantizationRuns_TensorCombos_TensorComboId",
                        column: x => x.TensorComboId,
                        principalTable: "TensorCombos",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

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
                name: "IX_BenchmarkRuns_CategoryBenchmarkId",
                table: "BenchmarkRuns",
                column: "CategoryBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_StartedUtc",
                table: "BenchmarkRuns",
                column: "StartedUtc");

            migrationBuilder.CreateIndex(
                name: "IX_BenchmarkRuns_TensorComboId",
                table: "BenchmarkRuns",
                column: "TensorComboId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_AiBenchmarkId",
                table: "QuantizationRuns",
                column: "AiBenchmarkId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_AiModelHashId",
                table: "QuantizationRuns",
                column: "AiModelHashId");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_StartedUtc",
                table: "QuantizationRuns",
                column: "StartedUtc");

            migrationBuilder.CreateIndex(
                name: "IX_QuantizationRuns_TensorComboId",
                table: "QuantizationRuns",
                column: "TensorComboId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "BenchmarkRuns");

            migrationBuilder.DropTable(
                name: "QuantizationRuns");
        }
    }
}
