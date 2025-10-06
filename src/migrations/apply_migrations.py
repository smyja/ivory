import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_duckdb_connection


def apply_migrations():
    """Apply all SQL migrations in the migrations directory."""
    migrations_dir = os.path.dirname(os.path.abspath(__file__))

    # Get all SQL files in the migrations directory
    migration_files = [f for f in os.listdir(migrations_dir) if f.endswith(".sql")]
    migration_files.sort()  # Apply migrations in alphabetical order

    with get_duckdb_connection() as conn:
        for migration_file in migration_files:
            print(f"Applying migration: {migration_file}")
            with open(os.path.join(migrations_dir, migration_file), "r") as f:
                sql = f.read()
                try:
                    conn.execute(sql)
                    print(f"Successfully applied {migration_file}")
                except Exception as e:
                    print(f"Error applying {migration_file}: {str(e)}")
                    raise


if __name__ == "__main__":
    apply_migrations()
