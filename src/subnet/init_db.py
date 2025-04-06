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
            dataset_id INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            total_rows INTEGER NOT NULL,
            percentage DOUBLE NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id)
        )
    """
    )

    # Create subclusters table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subclusters (
            id INTEGER PRIMARY KEY DEFAULT(nextval('subclusters_id_seq')),
            category_id INTEGER NOT NULL,
            title VARCHAR NOT NULL,
            row_count INTEGER NOT NULL,
            percentage DOUBLE NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(id)
        )
    """
    )

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
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id)
        )
    """
    )

    print("Database initialized successfully!")

except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
    raise

finally:
    conn.close()
