### utils/database.py
import os
import requests
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, Session
from models import Base, DatasetMetadata, DownloadStatus, DatasetRequest
from datasets import load_dataset
import pandas as pd
from typing import Dict
from fastapi import HTTPException
import logging
import duckdb
from contextlib import contextmanager
import time

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


# Initialize the engine
try:
    engine = create_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    metadata = MetaData()
    Base.metadata.create_all(bind=engine)
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
    """Get a DuckDB connection from the pool with proper cleanup."""
    conn = None
    try:
        # Try to reuse an existing connection
        if _duckdb_connections:
            conn = _duckdb_connections.pop()
        else:
            # Create a new connection if none available
            conn = duckdb.connect(
                "datasets.db",
                read_only=False,  # Use write mode for all operations
            )
        yield conn
    finally:
        if conn:
            try:
                # Return the connection to the pool instead of closing it
                _duckdb_connections.append(conn)
            except Exception as e:
                logger.error(f"Error managing DuckDB connection: {str(e)}")
                try:
                    conn.close()
                except:
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

            dataset = load_dataset(
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
