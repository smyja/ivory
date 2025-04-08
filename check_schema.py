import duckdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_schema():
    conn = duckdb.connect("datasets.db")

    try:
        # Get all tables
        tables = conn.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """
        ).fetchall()

        print("\nTables in database:")
        for table in tables:
            print(f"\n{table[0]}:")
            # Get columns for each table
            columns = conn.execute(
                f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table[0]}'
            """
            ).fetchall()
            for col in columns:
                print(f"  {col[0]}: {col[1]}")

    except Exception as e:
        logger.error(f"Error checking schema: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    check_schema()
