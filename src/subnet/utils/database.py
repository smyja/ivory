### utils/database.py
import os
import requests
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, Session
from models import (
    Base,
    DatasetMetadata,
    DownloadStatus,
    DatasetRequest,
    TextDB,
    Category,
    Level1Cluster,
    TextAssignment,
    ClusteringHistory,
)
from huggingface_hub import hf_hub_download
from datasets import load_dataset as hf_load_dataset
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from fastapi import HTTPException
import logging
import duckdb
from contextlib import contextmanager
import time
from datetime import datetime

# Import sequences if needed elsewhere, though Base.metadata should handle creation
from models import (
    dataset_id_seq,
    text_id_seq,
    category_id_seq,
    l1_cluster_id_seq,
    assignment_id_seq,
    history_id_seq,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a single DuckDB connection method
SQLALCHEMY_DATABASE_URL = "duckdb:///datasets.db"


def create_db_engine():
    """Create a SQLAlchemy engine with retries for database initialization."""
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            engine = create_engine(
                SQLALCHEMY_DATABASE_URL,
                echo=False,
                pool_pre_ping=True,  # Enable connection health checks
                pool_size=5,  # Increase pool size to handle more concurrent connections
                max_overflow=10,  # Allow overflow connections
                pool_timeout=30,  # Timeout for getting a connection from the pool
                pool_recycle=1800,  # Recycle connections after 30 minutes
            )
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # Use text() for raw SQL
                conn.commit()
            return engine
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {str(e)}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(
                    f"Failed to connect to database after {max_retries} attempts"
                )
                raise


def get_active_downloads(db: Session):
    active_downloads = (
        db.query(DatasetMetadata)
        .filter(DatasetMetadata.status == DownloadStatus.IN_PROGRESS)
        .all()
    )

    return [
        {
            "id": download.id,
            "dataset": download.name,
            "status": download.status.value,
            "download_date": download.download_date,
        }
        for download in active_downloads
    ]


def scan_existing_datasets(db: Session):
    """Scan the datasets directory and recreate database entries for existing datasets, including loading texts."""
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        logger.warning("Datasets directory not found")
        return

    logger.info(f"Scanning directory: {datasets_dir} for existing datasets...")
    processed_datasets = 0

    for dataset_dir_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_dir_name)
        if not os.path.isdir(dataset_path):
            continue

        # Use the directory name as the dataset identifier found on disk
        dataset_name_on_disk = dataset_dir_name

        # Check if dataset metadata already exists in database
        existing_dataset = (
            db.query(DatasetMetadata)
            .filter(DatasetMetadata.name == dataset_name_on_disk)
            .first()
        )
        if existing_dataset:
            # Optional: Could check if texts are missing and load them here, but simpler to assume if metadata exists, it's processed.
            # logger.debug(f"Dataset {dataset_name_on_disk} already exists in database metadata. Skipping scan processing.")
            continue

        logger.info(f"Found potential new dataset: {dataset_name_on_disk}")

        # Look for parquet files in the dataset directory and its subdirectories (splits)
        parquet_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_path = os.path.join(root, file)
                    parquet_files.append(parquet_path)
                    logger.info(f"  Found parquet file: {parquet_path}")

        if not parquet_files:
            logger.warning(
                f"No parquet files found for dataset {dataset_name_on_disk}. Cannot create metadata or load texts."
            )
            continue

        # --- Create Metadata Entry FIRST ---
        new_dataset = DatasetMetadata(
            name=dataset_name_on_disk,
            status="completed",  # Assume completed if found on disk
            description=f"Dataset found on disk in {dataset_path}",
            source="disk_scan",
            identifier=dataset_path,
            file_path=parquet_files[0],  # Store path to first parquet found
            verification_status="valid",  # Assume valid if found
            clustering_status="pending",
            is_clustered=False,
            created_at=datetime.now(),
            total_rows=0,  # Will update after loading texts
        )
        db.add(new_dataset)
        # Flush to get the ID for text insertion without committing the transaction yet
        try:
            db.flush()
            db.refresh(new_dataset)
            logger.info(
                f"Created metadata entry for {dataset_name_on_disk} with ID {new_dataset.id}. Attempting text load."
            )
        except Exception as e_flush:
            logger.error(
                f"Error flushing metadata for {dataset_name_on_disk}: {e_flush}. Rolling back."
            )
            db.rollback()
            continue  # Skip to next dataset directory

        # --- Load Texts from Parquet ---
        total_texts_loaded = 0
        try:
            # Process all found parquet files (e.g., from different splits)
            all_texts = []
            potential_text_columns = [
                "text",
                "content",
                "data",
                "prompt",
                "response",
                "sentence",
                "document",
            ]  # Add more common names if needed
            chosen_column = None

            for parquet_file in parquet_files:
                logger.info(f"  Reading texts from: {parquet_file}")
                try:
                    df = pd.read_parquet(parquet_file)

                    # --- Flexible Column Detection ---
                    text_column_name = None

                    # 1. Check common names
                    for col_name in potential_text_columns:
                        if col_name in df.columns:
                            text_column_name = col_name
                            logger.info(
                                f"    Found common text column: '{text_column_name}'"
                            )
                            break

                    # 2. Fallback: Find longest string column if no common name found
                    if text_column_name is None:
                        string_cols = df.select_dtypes(
                            include=["string", "object"]
                        )  # Include object for potential strings
                        if not string_cols.empty:
                            longest_col = None
                            max_avg_len = -1
                            for col in string_cols.columns:
                                # Calculate average length, handling potential non-strings or NaNs gracefully
                                try:
                                    # Convert to string and calculate length, fillna avoids errors
                                    avg_len = (
                                        string_cols[col]
                                        .astype(str)
                                        .str.len()
                                        .fillna(0)
                                        .mean()
                                    )
                                    if avg_len > max_avg_len:
                                        max_avg_len = avg_len
                                        longest_col = col
                                except Exception as e_len:
                                    logger.warning(
                                        f"      Could not calculate avg length for column '{col}': {e_len}"
                                    )
                                    continue

                            if longest_col:
                                text_column_name = longest_col
                                logger.info(
                                    f"    No common text column found. Using longest string column as fallback: '{text_column_name}' (Avg Length: {max_avg_len:.2f})"
                                )

                    # --- End Flexible Column Detection ---

                    if text_column_name is None:
                        logger.warning(
                            f"    Could not identify a suitable text column in {parquet_file}. Columns found: {list(df.columns)}. Skipping this file."
                        )
                        continue

                    # Use the identified column
                    texts_in_file = (
                        df[text_column_name].dropna().astype(str).tolist()
                    )  # Ensure string type
                    all_texts.extend(texts_in_file)
                    logger.info(
                        f"    Read {len(texts_in_file)} texts from column '{text_column_name}' in {parquet_file}."
                    )
                    if (
                        chosen_column is None
                    ):  # Store the first successfully used column name
                        chosen_column = text_column_name

                except Exception as e_read:
                    logger.error(
                        f"    Error reading or processing parquet file {parquet_file}: {e_read}"
                    )
                    continue  # Try next parquet file

            if not all_texts:
                logger.warning(
                    f"No valid texts found in any parquet files for {dataset_name_on_disk}. Metadata created, but no texts loaded."
                )
                new_dataset.error_message = (
                    "Could not identify or load texts from available parquet files."
                )
                # Commit metadata even without texts
                db.commit()
                processed_datasets += 1
                continue

            # --- Deduplicate texts before bulk insert ---
            unique_texts = list(set(all_texts))
            if len(unique_texts) < len(all_texts):
                logger.warning(
                    f"  Removed {len(all_texts) - len(unique_texts)} duplicate text entries for dataset {dataset_name_on_disk}."
                )

            # Prepare records for bulk insert using unique texts
            text_records = [
                {"dataset_id": new_dataset.id, "text": text_content}
                for text_content in unique_texts  # Use the deduplicated list
            ]

            if not text_records:  # Check if list is empty after deduplication
                logger.warning(
                    f"No unique texts found to insert for {dataset_name_on_disk}. Skipping text insertion."
                )
                new_dataset.error_message = (
                    "Dataset contained only duplicate or empty texts."
                )
                db.commit()
                processed_datasets += 1
                continue

            logger.info(
                f"  Attempting bulk insert of {len(text_records)} unique texts for dataset ID {new_dataset.id}... (Column Used: '{chosen_column}')"
            )
            db.bulk_insert_mappings(TextDB, text_records)
            db.flush()  # Ensure inserts are processed before updating count
            total_texts_loaded = len(
                text_records
            )  # Count based on unique texts inserted
            logger.info(
                f"  Successfully inserted {total_texts_loaded} unique texts for {dataset_name_on_disk}."
            )

            # Update total_rows in metadata
            new_dataset.total_rows = total_texts_loaded
            new_dataset.error_message = None  # Clear any previous error
            db.commit()
            processed_datasets += 1
            logger.info(
                f"Successfully processed dataset {dataset_name_on_disk} including text loading."
            )

        except Exception as e_texts:
            logger.error(
                f"Error loading texts for {dataset_name_on_disk} (ID: {new_dataset.id}): {e_texts}",
                exc_info=True,
            )
            db.rollback()  # Rollback text insert and metadata update attempt
            # Try to commit *just* the metadata with an error message
            try:
                # Need to re-add the object to the session if rollback detached it
                # Simpler: just update the object already added before flush if possible
                # But since we rolled back, the object might be in a transient state. Re-querying might be safest if complex.
                # For simplicity here, let's assume we can add it back or it stays associated.
                existing_meta = (
                    db.query(DatasetMetadata)
                    .filter(DatasetMetadata.id == new_dataset.id)
                    .first()
                )
                if existing_meta:  # Check if it exists after rollback
                    existing_meta.status = (
                        "failed"  # Mark as failed due to text load issue
                    )
                    existing_meta.error_message = (
                        f"Failed during text loading: {str(e_texts)[:500]}"
                    )
                    db.add(existing_meta)  # Re-add if needed
                    db.commit()
                    logger.info(
                        f"Committed metadata for {dataset_name_on_disk} with text loading failure status."
                    )
                else:
                    logger.error(
                        f"Could not even commit failed metadata status for {dataset_name_on_disk} after text load error."
                    )
            except Exception as e_commit_fail:
                logger.error(
                    f"Failed to commit error status for {dataset_name_on_disk}: {e_commit_fail}"
                )
                db.rollback()  # Final rollback

    logger.info(
        f"Finished scanning existing datasets. Processed {processed_datasets} new datasets."
    )


def init_db():
    """Initialize the database with required tables based on the new models."""
    logger.info("Initializing database schema...")
    max_retries = 3
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            # Use SQLAlchemy engine to create tables defined in models.py
            engine = create_db_engine()
            Base.metadata.create_all(bind=engine)
            logger.info(
                "Database schema initialized successfully using SQLAlchemy models."
            )

            # Initial scan might still be useful
            # with SessionLocal() as db:
            #    scan_existing_datasets(db)

            return  # Success
        except Exception as e:
            logger.error(
                f"Failed to initialize database schema (attempt {attempt + 1}/{max_retries}): {e}",
                exc_info=True,
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.critical(
                    "CRITICAL: Could not initialize database schema after multiple retries."
                )
                raise
        finally:
            if "engine" in locals() and engine:
                engine.dispose()


# Initialize the engine
try:
    engine = create_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    metadata = MetaData()

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Scan for existing datasets and recreate database entries
    with SessionLocal() as db:
        scan_existing_datasets(db)

except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise

active_downloads: Dict[int, Dict] = {}

# Global DuckDB connection pool
_duckdb_connections = []


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_duckdb_connection():
    """Get a DuckDB connection."""
    conn = None
    try:
        conn = duckdb.connect("datasets.db")
        yield conn
    except Exception as e:
        logger.error(f"Error with DuckDB connection: {str(e)}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing DuckDB connection: {str(e)}")
                pass


BASE_API_URL = "https://datasets-server.huggingface.co/rows"

logger = logging.getLogger(__name__)


def download_and_save_dataset(db: Session, request: DatasetRequest):
    new_metadata = None
    try:
        new_metadata = DatasetMetadata(
            name=request.dataset_name,
            subset=request.subset,
            split=request.split,
            status=DownloadStatus.IN_PROGRESS,
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)

        if request.num_rows:
            # Use the API to download a specific number of rows
            logger.info(
                f"Fetching {request.num_rows} rows via API for dataset: {request.dataset_name}"
            )

            config_param = f"&config={request.config}" if request.config else ""
            api_url = (
                f"https://datasets-server.huggingface.co/rows?"
                f"dataset={request.dataset_name}"
                f"{config_param}"
                f"&split={request.split}"
                f"&offset=0"
                f"&length={min(request.num_rows or 100, 100)}"
            )

            logger.info(f"Fetching data from URL: {api_url}")

            response = requests.get(api_url)
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404, detail=f"Resource not found: {response.text}"
                )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error fetching dataset: {response.text}",
                )

            api_data = response.json()
            rows = api_data["rows"]
            data = [row["row"] for row in rows]
            df = pd.DataFrame(data)

        else:
            # Use the HF library to download the entire dataset
            logger.info(f"Loading full dataset directly: {request.dataset_name}")

            dataset = hf_load_dataset(
                request.dataset_name,
                name=request.subset,
                split=request.split,
                cache_dir="temp_datasets",
            )

            if isinstance(dataset, dict):
                df = pd.concat(
                    [split_dataset.to_pandas() for split_dataset in dataset.values()]
                )
            else:
                df = dataset.to_pandas()

        # Save the DataFrame to a Parquet file
        final_path = os.path.join("datasets", request.dataset_name.replace("/", "_"))
        if request.subset:
            final_path = os.path.join(final_path, request.subset)
        if request.split:
            final_path = os.path.join(final_path, request.split)

        os.makedirs(final_path, exist_ok=True)
        parquet_file = os.path.join(final_path, "data.parquet")
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Dataset saved to: {parquet_file}")

        if os.path.exists(parquet_file):
            logger.info(f"Verified Parquet file exists in: {final_path}")
            new_metadata.status = DownloadStatus.COMPLETED
        else:
            logger.error(f"Failed to create Parquet file in: {final_path}")
            new_metadata.status = DownloadStatus.FAILED

        db.commit()
        logger.info(f"Updated dataset status to: {new_metadata.status}")

        # After df.to_parquet(parquet_file, index=False):
        logger.info(f"Loading texts from {parquet_file} into TextDB...")
        df = pd.read_parquet(parquet_file)
        # Assuming a 'text' column exists (adjust if different)
        if "text" not in df.columns:
            logger.error(
                f"'text' column not found in {parquet_file}. Cannot load texts."
            )
            # Update metadata status to failed?
            raise ValueError("Missing 'text' column in dataset Parquet file.")

        text_records = [
            {"dataset_id": new_metadata.id, "text": row["text"]}
            for _, row in df.iterrows()
            if pd.notna(row["text"])
        ]  # Handle potential NaNs

        if not text_records:
            logger.warning("No valid text records found in Parquet file.")
            # Update status?
            db.commit()  # Commit metadata update even if no texts
            return  # Or raise error?

        try:
            logger.info(f"Bulk inserting {len(text_records)} texts...")
            # Use Core API for RETURNING ID efficiently with DuckDB/SQLite
            stmt = (
                TextDB.__table__.insert()
                .values(text_records)
                .returning(TextDB.id, TextDB.text)
            )
            results = db.execute(stmt).fetchall()  # List of (id, text) tuples
            db.flush()  # Ensure inserts are processed

            text_content_to_db_id = {text: id for id, text in results}
            logger.info(f"Successfully inserted texts and created ID map.")

            # --- CRITICAL: Need to store `text_content_to_db_id` ---
            # This map is needed by `save_clustering_results`.
            # Options:
            # 1. Return it from this function (makes signature complex).
            # 2. Store it temporarily (e.g., Redis, file) - adds complexity.
            # 3. Re-query TextDB before saving assignments (inefficient for large datasets).
            # --> Simplest for now might be to pass it around, but needs careful handling in the route.

            new_metadata.total_rows = len(text_records)  # Update total rows
            new_metadata.status = (
                "verified"  # Assuming download implies verification for now
            )
            new_metadata.verification_status = "valid"
            db.commit()
            logger.info(
                f"Dataset download and text loading complete for {request.name}."
            )
            # Return the map? Or rely on caller to manage it? For now, let's assume caller handles it.
            # return text_content_to_db_id

        except Exception as e_insert:
            logger.error(f"Failed to bulk insert texts: {e_insert}", exc_info=True)
            db.rollback()
            new_metadata.status = "failed"
            new_metadata.error_message = f"Failed text insertion: {e_insert}"
            db.commit()
            raise

    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        if new_metadata:
            new_metadata.status = DownloadStatus.FAILED
            db.commit()
        raise e

    finally:
        # Clean up temporary files if any
        if os.path.exists("temp_datasets"):
            import shutil

            shutil.rmtree("temp_datasets")


def verify_and_update_dataset_status(db: Session, dataset_id: int):
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        logger.error(f"Dataset with id {dataset_id} not found in database")
        return False, "Dataset not found in database"

    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    logger.info(f"Verifying dataset at path: {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        dataset_metadata.status = DownloadStatus.FAILED
        db.commit()
        return False, f"Dataset directory not found: {dataset_path}"

    try:
        # Check for the Parquet file
        parquet_file = os.path.join(dataset_path, "data.parquet")
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Parquet file not found at {parquet_file}")
        logger.info(f"Found Parquet file at {parquet_file}")

        # Optionally, check for dataset_info.json if you're creating it
        info_file = os.path.join(dataset_path, "dataset_info.json")
        if os.path.exists(info_file):
            logger.info(f"Found dataset_info.json at {info_file}")
        else:
            logger.warning(f"dataset_info.json not found at {info_file}")

        # Verify the Parquet file can be read
        try:
            pd.read_parquet(parquet_file)
            logger.info("Successfully read Parquet file")
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {str(e)}")

        # If we've made it this far, consider the dataset valid
        dataset_metadata.status = DownloadStatus.COMPLETED
        db.commit()
        logger.info(f"Dataset {dataset_metadata.name} verified successfully")
        return True, "Dataset verified successfully"

    except Exception as e:
        dataset_metadata.status = DownloadStatus.FAILED
        db.commit()
        error_message = f"Error verifying dataset: {str(e)}"
        logger.error(error_message)
        return False, error_message


# --- Bulk Data Saving for Clustering Results ---


def save_clustering_results(db: Session, results: Dict[str, Any]):
    """
    Saves the results from cluster_texts_scalable using bulk operations.
    Assumes results dictionary contains 'assignments', 'level1_titles', 'category_names', 'metadata'.
    """
    metadata = results.get("metadata", {})
    dataset_id = metadata.get("dataset_id")
    version = metadata.get("version")
    assignments = results.get("assignments", [])
    l1_titles_map = results.get("level1_titles", {})  # l1_cluster_id -> title
    category_names_map = results.get("category_names", {})  # l2_cluster_id -> name

    if not dataset_id or version is None or not assignments:
        logger.error("Missing critical data in clustering results. Cannot save.")
        raise ValueError("Invalid clustering results structure for saving.")

    logger.info(
        f"Saving clustering results for dataset {dataset_id}, version {version}."
    )

    try:
        # 1. Prepare and Bulk Insert Categories
        # Create a mapping from L2 Cluster ID (from HDBSCAN) to Category DB ID
        l2_id_to_category_db_id: Dict[int, int] = {}
        category_records = []
        for l2_cluster_id, name in category_names_map.items():
            category_records.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "l2_cluster_id": l2_cluster_id,
                    "name": name,
                }
            )

        if category_records:
            # Use bulk_insert_mappings for flexibility, requires primary key after insert
            # Alternative: Use Core API's insert().values([...]).returning(Category.id) if dialect supports RETURNING
            inserted_categories = db.bulk_insert_mappings(
                Category, category_records, return_defaults=True
            )
            db.flush()  # Ensure IDs are populated
            for i, record in enumerate(category_records):
                # The order should be preserved, map L2 ID using the original record
                l2_id = record["l2_cluster_id"]
                db_id = record["id"]  # ID is populated by return_defaults=True
                l2_id_to_category_db_id[l2_id] = db_id
            logger.info(f"Bulk inserted {len(category_records)} categories.")
        else:
            logger.warning("No L2 categories generated to save.")

        # 2. Prepare and Bulk Insert Level1Clusters
        # Create a mapping from L1 Cluster ID (from HDBSCAN) to Level1Cluster DB ID
        l1_id_to_l1cluster_db_id: Dict[int, int] = {}
        l1_cluster_records = []
        # Need to infer the category_db_id for each L1 cluster
        # This requires mapping L1 ID -> L1 Title -> L2 ID -> Category DB ID
        title_to_l2_id = {  # Reconstruct this map (ideally passed from clustering)
            title: l2_id
            for l2_id, titles in l2_cluster_l1_titles.items()  # Need l2_cluster_l1_titles map!
            for title in titles
        }  # This reconstruction is inefficient, better to pass necessary maps from clustering func

        # --- !!! We need the title -> L2 ID mapping from the clustering function !!! ---
        # --- Assuming we get it somehow for now ---
        # Let's pretend 'results' includes 'title_to_l2_id_map' for demonstration
        title_to_l2_id_map = results.get(
            "title_to_l2_id_map", {}
        )  # Example: {'Title A': 0, 'Title B': 0, 'Title C': 1}

        for l1_cluster_id, title in l1_titles_map.items():
            l2_cluster_id = title_to_l2_id_map.get(
                title, -1
            )  # Find L2 ID based on L1 title
            category_db_id = l2_id_to_category_db_id.get(
                l2_cluster_id
            )  # Find Category DB ID
            if category_db_id is None:
                # Handle L1 clusters whose titles were noise at L2 or uncategorized
                # Option 1: Create a default "Uncategorized" category?
                # Option 2: Skip saving this L1 cluster? (Chosen for now)
                logger.warning(
                    f"L1 Cluster {l1_cluster_id} ('{title}') maps to L2 ID {l2_cluster_id} which has no category DB ID. Skipping L1 cluster record."
                )
                continue

            l1_cluster_records.append(
                {
                    "category_id": category_db_id,
                    "version": version,
                    "l1_cluster_id": l1_cluster_id,
                    "title": title,
                }
            )

        if l1_cluster_records:
            inserted_l1_clusters = db.bulk_insert_mappings(
                Level1Cluster, l1_cluster_records, return_defaults=True
            )
            db.flush()
            for i, record in enumerate(l1_cluster_records):
                l1_id = record["l1_cluster_id"]
                db_id = record["id"]
                l1_id_to_l1cluster_db_id[l1_id] = db_id
            logger.info(f"Bulk inserted {len(l1_cluster_records)} Level 1 clusters.")
        else:
            logger.warning(
                "No Level 1 clusters generated or mapped to categories to save."
            )

        # 3. Prepare and Bulk Insert TextAssignments
        assignment_records = []
        # Need mapping from text index/content to its DB ID
        # This should be done earlier, perhaps during initial text insertion
        # Assuming we have a map: text_content -> text_db_id
        # --- !!! Need text_content_to_db_id map !!! ---
        text_content_to_db_id = results.get(
            "text_content_to_db_id_map", {}
        )  # Example needed

        original_texts = results.get("original_texts", [])  # Need original texts list
        if not text_content_to_db_id or not original_texts:
            logger.error(
                "Missing text_content_to_db_id map or original_texts list in results. Cannot save assignments."
            )
            raise ValueError("Missing data for saving assignments.")

        for i, assignment in enumerate(assignments):
            text_content = original_texts[i]  # Get text by index
            text_db_id = text_content_to_db_id.get(text_content)
            if text_db_id is None:
                logger.warning(
                    f"Could not find DB ID for text: '{text_content[:50]}...'. Skipping assignment."
                )
                continue

            l1_cluster_id_hdbscan = assignment["l1_cluster_id"]
            level1_cluster_db_id = l1_id_to_l1cluster_db_id.get(l1_cluster_id_hdbscan)

            if level1_cluster_db_id is None:
                # Handle texts assigned to noise L1 clusters or L1 clusters that failed L2 mapping
                # Option 1: Assign to a default "Noise" Level1Cluster?
                # Option 2: Skip saving assignment? (Chosen for now)
                logger.warning(
                    f"Text ID {text_db_id} assigned to L1 ID {l1_cluster_id_hdbscan} which has no L1 cluster DB ID. Skipping assignment."
                )
                continue

            assignment_records.append(
                {
                    "text_id": text_db_id,
                    "version": version,
                    "level1_cluster_id": level1_cluster_db_id,
                    "l1_probability": assignment["l1_prob"],
                    "l2_probability": assignment["l2_prob"],
                }
            )

        if assignment_records:
            # Can use bulk_insert_mappings without return_defaults if IDs aren't needed immediately
            db.bulk_insert_mappings(TextAssignment, assignment_records)
            logger.info(f"Bulk inserted {len(assignment_records)} text assignments.")
        else:
            logger.warning("No text assignments mapped to saved L1 clusters.")

        # 4. Commit Transaction
        db.commit()
        logger.info(
            f"Successfully saved all clustering results for dataset {dataset_id}, version {version}."
        )

    except Exception as e:
        logger.error(
            f"Error saving clustering results for dataset {dataset_id}, version {version}: {e}",
            exc_info=True,
        )
        db.rollback()  # Rollback on any error during the saving process
        raise  # Re-raise the exception to be handled by the caller
