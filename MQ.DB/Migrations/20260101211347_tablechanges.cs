using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MQ.DB.Migrations
{
    /// <inheritdoc />
    public partial class tablechanges : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "TensorCombo",
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
                    table.PrimaryKey("PK_TensorCombo", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_TensorCombo_BaseQuant_Embeddings_LmHead_AttnQ_AttnKV_AttnOutput_FfnUpGate_FfnDown_MoeExperts_MoeRouter",
                table: "TensorCombo",
                columns: new[] { "BaseQuant", "Embeddings", "LmHead", "AttnQ", "AttnKV", "AttnOutput", "FfnUpGate", "FfnDown", "MoeExperts", "MoeRouter" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "TensorCombo");
        }
    }
}
