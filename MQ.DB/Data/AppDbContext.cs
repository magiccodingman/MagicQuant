using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Sqlite;
using System.IO;

namespace MQ.DB.Data;

public class AppDbContext : DbContext
{
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        var directory = Cache.MagicQuantDirectory;
        var dbPath = Path.Combine(directory, "MagicQuant_SQLite.db");
        optionsBuilder.UseSqlite($"Data Source={dbPath};Foreign Keys=True;");
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // This is where you configure composite keys, default values, etc.
        base.OnModelCreating(modelBuilder);
    }
}