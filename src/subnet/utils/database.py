from sqlalchemy import create_engine, Table, MetaData, Column, String
from sqlalchemy.orm import sessionmaker
from models import Base, DatasetMetadata, DownloadStatus
from datasets import load_dataset
import pandas as pd
from typing import Dict
from datetime import datetime
import pyarrow.parquet as pq

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

def download_and_save_dataset(db, request):
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
        
        active_downloads[new_metadata.id] = {
            "start_time": datetime.now(),
            "status": DownloadStatus.IN_PROGRESS
        }

        dataset = load_dataset(
            request.dataset_name,
            name=request.subset,
            split=request.split,
            cache_dir=request.cache_dir
        )
        
        # Check if the dataset is already in Parquet format
        if hasattr(dataset, '_data_files') and dataset._data_files and dataset._data_files[0].endswith('.parquet'):
            # If it's a Parquet file, read it directly
            df = pq.read_table(dataset._data_files[0]).to_pandas()
        else:
            # If it's not Parquet, convert to pandas DataFrame
            df = dataset.to_pandas()
        
        table_name = f"{request.dataset_name.replace('/', '_')}_{request.subset or 'default'}_{request.split or 'all'}"
        
        # Create table
        columns = [Column(name, String) for name in df.columns]
        table = Table(table_name, metadata, *columns, extend_existing=True)
        metadata.create_all(engine)
        
        # Save to DuckDB
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        new_metadata.status = DownloadStatus.COMPLETED
        db.commit()
        
        active_downloads[new_metadata.id]["status"] = DownloadStatus.COMPLETED
        
    except Exception as e:
        if new_metadata:
            new_metadata.status = DownloadStatus.FAILED
            db.commit()
            active_downloads[new_metadata.id]["status"] = DownloadStatus.FAILED
        db.rollback()
        raise e
    finally:
        if new_metadata and new_metadata.id in active_downloads:
            active_downloads[new_metadata.id]["end_time"] = datetime.now()

def get_active_downloads():
    return active_downloads