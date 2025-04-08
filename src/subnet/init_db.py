import duckdb
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if database file exists and remove it
db_file = "datasets.db"
if os.path.exists(db_file):
    logger.info(f"Removing existing database file: {db_file}")
    os.remove(db_file)

conn = duckdb.connect(db_file)

try:
    # Create sequences for auto-incrementing IDs
    conn.execute("CREATE SEQUENCE IF NOT EXISTS dataset_metadata_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS texts_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS categories_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS level1_clusters_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS text_assignments_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS clustering_history_id_seq")

    # Create dataset_metadata table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            id INTEGER PRIMARY KEY DEFAULT(nextval('dataset_metadata_id_seq')),
            name VARCHAR NOT NULL UNIQUE,
            description VARCHAR,
            source VARCHAR,
            identifier VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR DEFAULT 'pending',
            verification_status VARCHAR DEFAULT 'pending',
            clustering_status VARCHAR DEFAULT 'pending',
            is_clustered BOOLEAN DEFAULT FALSE,
            total_rows INTEGER,
            error_message VARCHAR,
            file_path VARCHAR
        )
    """
    )

    # Create texts table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY DEFAULT(nextval('texts_id_seq')),
            dataset_id INTEGER NOT NULL,
            text VARCHAR NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id),
            UNIQUE(dataset_id, text)
        )
    """
    )

    # Create categories table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY DEFAULT(nextval('categories_id_seq')),
            dataset_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            l2_cluster_id INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id),
            UNIQUE(dataset_id, version, l2_cluster_id)
        )
    """
    )

    # Create level1_clusters table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS level1_clusters (
            id INTEGER PRIMARY KEY DEFAULT(nextval('level1_clusters_id_seq')),
            category_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            l1_cluster_id INTEGER NOT NULL,
            title VARCHAR NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(id),
            UNIQUE(category_id, version, l1_cluster_id)
        )
    """
    )

    # Create text_assignments table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS text_assignments (
            id INTEGER PRIMARY KEY DEFAULT(nextval('text_assignments_id_seq')),
            text_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            level1_cluster_id INTEGER NOT NULL,
            l1_probability DOUBLE NOT NULL,
            l2_probability DOUBLE NOT NULL,
            FOREIGN KEY(text_id) REFERENCES texts(id),
            FOREIGN KEY(level1_cluster_id) REFERENCES level1_clusters(id),
            UNIQUE(text_id, version)
        )
    """
    )

    # Create clustering_history table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clustering_history (
            id INTEGER PRIMARY KEY DEFAULT(nextval('clustering_history_id_seq')),
            dataset_id INTEGER NOT NULL,
            clustering_version INTEGER NOT NULL,
            clustering_status VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            error_message VARCHAR,
            details VARCHAR,
            FOREIGN KEY(dataset_id) REFERENCES dataset_metadata(id),
            UNIQUE(dataset_id, clustering_version)
        )
    """
    )

    print("Database initialized successfully!")

except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
    raise

finally:
    conn.close()
