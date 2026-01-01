using Microsoft.EntityFrameworkCore;
using MQ.DB.Data;

namespace MQ.DB.Services;

public static class SqlService
{
    // Static flag to track if we've checked/migrated in this runtime session
    private static bool _isInitialized = false;
        
    // Lock object to prevent race conditions if two threads call GetContext simultaneously at start
    private static readonly object _initLock = new object();

    /// <summary>
    /// Gets a ready-to-use DbContext. 
    /// Automatically handles directory creation and DB Migrations on the first call.
    /// </summary>
    public static AppDbContext GetContext()
    {
        // 1. Check if we've already initialized in this runtime
        if (!_isInitialized)
        {
            lock (_initLock)
            {
                if (!_isInitialized)
                {
                    InitializeDatabase();
                    _isInitialized = true;
                }
            }
        }

        // 2. Return a new instance for the unit of work
        return new AppDbContext();
    }

    private static void InitializeDatabase()
    {
        // Ensure the directory exists (using your Cache static path)
        var directory = Cache.MagicQuantDirectory;
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        
        using (var db = new AppDbContext())
        {
            db.Database.Migrate();
        }
    }
}