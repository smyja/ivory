import os
import logging
from datetime import datetime
import duckdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_datasets_folder():
    """Scan the datasets folder and populate the database with available datasets."""
    try:
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        if not os.path.exists(datasets_dir):
            logger.error(f"Datasets directory not found at {datasets_dir}")
            return

        # Connect to the database
        with duckdb.connect("datasets.db") as conn:
            # Get existing datasets to avoid duplicates
            existing_datasets = set()
            try:
                result = conn.execute("SELECT name FROM dataset_metadata").fetchall()
                existing_datasets = {row[0] for row in result}
            except Exception as e:
                logger.warning(f"Error fetching existing datasets: {e}")

            # Scan datasets directory
            for dataset_name in os.listdir(datasets_dir):
                dataset_path = os.path.join(datasets_dir, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue

                if dataset_name in existing_datasets:
                    logger.info(f"Dataset {dataset_name} already exists in database")
                    continue

                # Look for splits (train, test, etc.)
                splits = []
                for split in os.listdir(dataset_path):
                    split_path = os.path.join(dataset_path, split)
                    if os.path.isdir(split_path):
                        # Check if the split contains parquet files
                        if any(f.endswith(".parquet") for f in os.listdir(split_path)):
                            splits.append(split)

                if not splits:
                    logger.warning(f"No valid splits found for dataset {dataset_name}")
                    continue

                # Add dataset entry for each split
                for split in splits:
                    try:
                        conn.execute(
                            """
                            INSERT INTO dataset_metadata (
                                name, subset, split, download_date, 
                                is_clustered, status, clustering_status
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            [
                                dataset_name,
                                dataset_name,  # Using dataset name as subset for now
                                split,
                                datetime.now(),
                                False,
                                "completed",  # Status of download
                                "not_started",  # Status of clustering
                            ],
                        )
                        logger.info(f"Added dataset {dataset_name} with split {split}")
                    except Exception as e:
                        logger.error(
                            f"Error adding dataset {dataset_name} split {split}: {e}"
                        )

            logger.info("Dataset scan completed successfully")

    except Exception as e:
        logger.error(f"Error scanning datasets: {e}")
        raise


if __name__ == "__main__":
    scan_datasets_folder()
