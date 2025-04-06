import duckdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conn = duckdb.connect("datasets.db")

try:
    # Create sequences for auto-incrementing IDs
    conn.execute("CREATE SEQUENCE IF NOT EXISTS dataset_metadata_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS categories_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS subclusters_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS texts_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS clustering_history_id_seq")

    # Create dataset_metadata table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            id INTEGER PRIMARY KEY DEFAULT(nextval('dataset_metadata_id_seq')),
            name VARCHAR NOT NULL,
            subset VARCHAR,
            split VARCHAR,
            download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_clustered BOOLEAN DEFAULT FALSE,
            status VARCHAR DEFAULT 'pending',
            clustering_status VARCHAR DEFAULT 'not_started'
        )
    """
    )

    # Create categories table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY DEFAULT(nextval('categories_id_seq')),
            dataset_id INTEGER,
            name VARCHAR,
            total_rows INTEGER,
            percentage FLOAT,
            version INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id)
        )
    """
    )

    # Add version column to categories if it doesn't exist
    try:
        conn.execute(
            "ALTER TABLE categories ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1"
        )
        print("Ensured version column exists in categories table")
    except Exception as e:
        logger.error(f"Error adding version column to categories: {str(e)}")

    # Create subclusters table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subclusters (
            id INTEGER PRIMARY KEY DEFAULT(nextval('subclusters_id_seq')),
            category_id INTEGER NOT NULL,
            title VARCHAR NOT NULL,
            row_count INTEGER NOT NULL,
            percentage DOUBLE NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(category_id) REFERENCES categories(id)
        )
    """
    )

    # Add version column to subclusters if it doesn't exist
    try:
        conn.execute(
            "ALTER TABLE subclusters ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1"
        )
        print("Ensured version column exists in subclusters table")
    except Exception as e:
        logger.error(f"Error adding version column to subclusters: {str(e)}")

    # Create texts table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY DEFAULT(nextval('texts_id_seq')),
            text VARCHAR NOT NULL
        )
    """
    )

    # Create text_clusters table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS text_clusters (
            text_id INTEGER,
            subcluster_id INTEGER,
            membership_score DOUBLE NOT NULL,
            PRIMARY KEY (text_id, subcluster_id),
            FOREIGN KEY(text_id) REFERENCES texts(id),
            FOREIGN KEY(subcluster_id) REFERENCES subclusters(id)
        )
    """
    )

    # Create clustering_history table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clustering_history (
            id INTEGER PRIMARY KEY DEFAULT(nextval('clustering_history_id_seq')),
            dataset_id INTEGER,
            clustering_status VARCHAR NOT NULL,
            titling_status VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            error_message VARCHAR,
            clustering_version INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id)
        )
    """
    )

    # Add clustering_version column if it doesn't exist
    # DuckDB doesn't support adding columns with constraints, so we need to recreate the table
    try:
        # Check if clustering_version column exists
        columns = conn.execute("DESCRIBE clustering_history").fetchall()
        column_names = [col[0] for col in columns]

        if "clustering_version" not in column_names:
            # Create a new table with the desired schema
            conn.execute(
                """
                CREATE TABLE clustering_history_new (
                    id INTEGER PRIMARY KEY,
                    dataset_id INTEGER,
                    clustering_status VARCHAR NOT NULL,
                    titling_status VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    error_message VARCHAR,
                    clustering_version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id)
                )
                """
            )

            # Copy data from the old table to the new table
            conn.execute(
                """
                INSERT INTO clustering_history_new 
                SELECT id, dataset_id, clustering_status, titling_status, created_at, completed_at, error_message, 1
                FROM clustering_history
                """
            )

            # Drop the old table
            conn.execute("DROP TABLE clustering_history")

            # Rename the new table to the original name
            conn.execute(
                "ALTER TABLE clustering_history_new RENAME TO clustering_history"
            )

            print("Added clustering_version column to clustering_history table")
    except Exception as e:
        logger.error(f"Error adding clustering_version column: {str(e)}")

    print("Database initialized successfully!")

except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
    raise

finally:
    conn.close()
