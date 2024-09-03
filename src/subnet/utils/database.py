### utils/database.py
import os
import shutil
from sqlalchemy import create_engine, Table, MetaData, Column, String
from sqlalchemy.orm import sessionmaker, Session
from models import Base, DatasetMetadata, DownloadStatus
from datasets import load_dataset
import pandas as pd
from typing import Dict
from datetime import datetime
import pyarrow.parquet as pq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = "duckdb:///datasets.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()
Base.metadata.create_all(bind=engine)

active_downloads: Dict[int, Dict] = {}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def download_and_save_dataset(db: Session, request):
    new_metadata = None
    try:
        new_metadata = DatasetMetadata(
            name=request.dataset_name,
            subset=request.subset,
            split=request.split,
            status=DownloadStatus.IN_PROGRESS
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)

        # Load the dataset
        dataset = load_dataset(
            request.dataset_name,
            name=request.subset,
            split=request.split,
            cache_dir="temp_datasets"
        )

        # Define the final path
        final_path = os.path.join("datasets", request.dataset_name.replace("/", "_"))
        if request.subset:
            final_path = os.path.join(final_path, request.subset)

        os.makedirs(final_path, exist_ok=True)
        logger.info(f"Created directory: {final_path}")

        # Handle different types of dataset objects
        if isinstance(dataset, dict):  # DatasetDict
            for split_name, split_dataset in dataset.items():
                split_path = os.path.join(final_path, split_name)
                os.makedirs(split_path, exist_ok=True)
                parquet_file = os.path.join(split_path, "data.parquet")
                df = split_dataset.to_pandas()
                df.to_parquet(parquet_file, index=False)
                logger.info(f"Saved {split_name} split to: {parquet_file}")
        else:  # Single Dataset
            split_name = request.split if request.split else "train"
            split_path = os.path.join(final_path, split_name)
            os.makedirs(split_path, exist_ok=True)
            parquet_file = os.path.join(split_path, "data.parquet")
            df = dataset.to_pandas()
            df.to_parquet(parquet_file, index=False)
            logger.info(f"Saved dataset to: {parquet_file}")

        # Verify at least one file was created
        if any(os.path.exists(os.path.join(final_path, split, "data.parquet")) for split in os.listdir(final_path) if os.path.isdir(os.path.join(final_path, split))):
            logger.info(f"Verified at least one Parquet file exists in: {final_path}")
            new_metadata.status = DownloadStatus.COMPLETED
        else:
            logger.error(f"Failed to create any Parquet files in: {final_path}")
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
        # Clean up temporary files
        if os.path.exists("temp_datasets"):
            import shutil
            shutil.rmtree("temp_datasets")


def verify_and_update_dataset_status(db: Session, dataset_id: int):
    dataset_metadata = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
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
        # Check for dataset_dict.json
        dataset_dict_path = os.path.join(dataset_path, "dataset_dict.json")
        if not os.path.exists(dataset_dict_path):
            raise FileNotFoundError(f"dataset_dict.json not found at {dataset_dict_path}")
        logger.info(f"Found dataset_dict.json at {dataset_dict_path}")

        # Check for train directory
        train_path = os.path.join(dataset_path, "train")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"train directory not found at {train_path}")
        logger.info(f"Found train directory at {train_path}")

        # Check for files in train directory
        required_files = ['data-00000-of-00001.arrow', 'dataset_info.json', 'state.json']
        for file in required_files:
            file_path = os.path.join(train_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file} not found at {file_path}")
            logger.info(f"Found {file} at {file_path}")

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

def get_active_downloads():
    return active_downloads
