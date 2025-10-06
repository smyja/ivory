import duckdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dataset_state():
    # Point to the correct database file path
    db_path = "src/datasets.db"
    conn = duckdb.connect(db_path)

    try:
        # Check dataset metadata
        logger.info("Checking dataset metadata...")
        dataset = conn.execute(
            """
            SELECT id, name, status, clustering_status, is_clustered 
            FROM dataset_metadata 
            WHERE id = 1
        """
        ).fetchall()
        print("\nDataset Metadata:")
        print(dataset)

        # Check texts
        logger.info("Checking texts...")
        text_count = conn.execute(
            """
            SELECT COUNT(*) 
            FROM texts
        """
        ).fetchone()[0]
        print(f"\nNumber of texts: {text_count}")

        # Check clustering history
        logger.info("Checking clustering history...")
        history = conn.execute(
            """
            SELECT clustering_version, clustering_status, created_at, completed_at, error_message
            FROM clustering_history 
            WHERE dataset_id = 1
            ORDER BY clustering_version
        """
        ).fetchall()
        print("\nClustering History:")
        for entry in history:
            print(entry)

        # Check categories
        logger.info("Checking categories...")
        categories = conn.execute(
            """
            SELECT id, version, l2_cluster_id, name
            FROM categories 
            WHERE dataset_id = 1
            ORDER BY version, l2_cluster_id
            """
        ).fetchall()
        print("\nCategories:")
        for cat in categories:
            print(cat)

        # Check level1_clusters
        logger.info("Checking level1_clusters...")
        subclusters = conn.execute(
            """
            SELECT l1.id, l1.version, l1.l1_cluster_id, l1.title, c.version as category_version, c.name as category_name
            FROM level1_clusters l1
            JOIN categories c ON l1.category_id = c.id
            WHERE c.dataset_id = 1
            ORDER BY l1.version, l1.l1_cluster_id
            """
        ).fetchall()
        print("\nLevel1 Clusters:")
        for cluster in subclusters:
            print(cluster)

        # Check text assignments
        logger.info("Checking text assignments...")
        text_clusters = conn.execute(
            """
            SELECT ta.text_id, ta.version, ta.level1_cluster_id, ta.l1_probability, ta.l2_probability, l1.title as l1_title
            FROM text_assignments ta
            JOIN level1_clusters l1 ON ta.level1_cluster_id = l1.id
            JOIN categories c ON l1.category_id = c.id
            WHERE c.dataset_id = 1
            ORDER BY ta.text_id
            LIMIT 10
            """
        ).fetchall()
        print("\nText Assignments (first 10):")
        for tc in text_clusters:
            print(tc)

    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    check_dataset_state()
