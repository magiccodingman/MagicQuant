using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Sqlite;
using System.IO;

namespace MQ.DB.Data;

public class AppDbContext : DbContext
{
    // This represents a table in your DB. Add more DbSets here as you create models.
    // public DbSet<Trade> Trades { get; set; } 

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        // 1. Get the directory from your existing static Cache
        var directory = Cache.MagicQuantDirectory;
            
        // 2. Combine with the filename
        var dbPath = Path.Combine(directory, "MagicQuant_SQLite.db");

        // 3. Configure SQLite
        // EF Core for SQLite enables Foreign Keys by default (PRAGMA foreign_keys = ON),
        // so you don't typically need extra configuration for that, but it handles it here.
        optionsBuilder.UseSqlite($"Data Source={dbPath}");
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // This is where you configure composite keys, default values, etc.
        base.OnModelCreating(modelBuilder);
    }
}