using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class BenchmarkPerformanceUpgrade : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "GpuMemoryLimitsJson",
                table: "ExecutionPlanProbeCaches",
                type: "TEXT",
                maxLength: 4000,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "MaxCandidateNgl",
                table: "ExecutionPlanProbeCaches",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<ulong>(
                name: "NativeModelSizeBytes",
                table: "ExecutionPlanProbeCaches",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0ul);

            migrationBuilder.AddColumn<string>(
                name: "NativeQuantizationKey",
                table: "ExecutionPlanProbeCaches",
                type: "TEXT",
                maxLength: 128,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "NativeStableNgl",
                table: "ExecutionPlanProbeCaches",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "ProbeSchemaVersion",
                table: "ExecutionPlanProbeCaches",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<ulong>(
                name: "Q8ModelSizeBytes",
                table: "ExecutionPlanProbeCaches",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0ul);

            migrationBuilder.AddColumn<int>(
                name: "Q8StableNgl",
                table: "ExecutionPlanProbeCaches",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "TensorSplitJson",
                table: "ExecutionPlanProbeCaches",
                type: "TEXT",
                maxLength: 4000,
                nullable: false,
                defaultValue: "");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "GpuMemoryLimitsJson",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "MaxCandidateNgl",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "NativeModelSizeBytes",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "NativeQuantizationKey",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "NativeStableNgl",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "ProbeSchemaVersion",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "Q8ModelSizeBytes",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "Q8StableNgl",
                table: "ExecutionPlanProbeCaches");

            migrationBuilder.DropColumn(
                name: "TensorSplitJson",
                table: "ExecutionPlanProbeCaches");
        }
    }
}
