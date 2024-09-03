from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import logging
import os
import duckdb
import pandas as pd
from models import DatasetRequest, DatasetMetadataResponse, DatasetMetadata
from utils.database import get_db, download_and_save_dataset, get_active_downloads, verify_and_update_dataset_status

router = APIRouter(prefix="/datasets")
logger = logging.getLogger(__name__)

# Helper function to get DuckDB connection
def get_duckdb_connection():
    return duckdb.connect('datasets.db')

@router.post("")
async def download_dataset(request: DatasetRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        background_tasks.add_task(download_and_save_dataset, db, request)
        return {"message": "Dataset download started in the background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=List[DatasetMetadataResponse])
async def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(DatasetMetadata).all()
    return datasets

@router.get("/active_downloads")
async def list_active_downloads():
    return get_active_downloads()

@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: int, db: Session = Depends(get_db)):
    dataset_metadata = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail=f"Dataset with id {dataset_id} not found in database")

    logger.info(f"Found dataset metadata for id {dataset_id}: {dataset_metadata.__dict__}")

    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    if dataset_metadata.subset:
        dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

    logger.info(f"Constructed dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_path}")

    # Load data into DuckDB
    con = get_duckdb_connection()
    con.execute(f"CREATE TABLE IF NOT EXISTS dataset_{dataset_id} AS SELECT * FROM parquet_scan('{dataset_path}/data.parquet')")
    
    # Get splits info
    splits = [split for split in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, split))]
    
    if not splits:
        raise HTTPException(status_code=404, detail=f"No splits found in dataset directory: {dataset_path}")

    split_info = {}
    for split in splits:
        data_file = os.path.join(dataset_path, split, "data.parquet")
        if os.path.exists(data_file):
            df = pd.read_parquet(data_file)
            split_info[split] = {
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "sample_rows": df.head(5).to_dict(orient="records")
            }

    con.close()
    return {
        "id": dataset_metadata.id,
        "name": dataset_metadata.name,
        "subset": dataset_metadata.subset,
        "status": dataset_metadata.status,
        "splits": split_info
    }

@router.get("/{dataset_id}/records")
async def get_dataset_records(
    dataset_id: int,
    split: Optional[str] = Query(None, description="Dataset split to query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    dataset_metadata = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    if dataset_metadata.subset:
        dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

    # If split is not provided, find the first available split
    if split is None:
        available_splits = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if not available_splits:
            raise HTTPException(status_code=404, detail="No splits found for this dataset")
        split = available_splits[0]

    data_file = os.path.join(dataset_path, split, "data.parquet")
    if not os.path.exists(data_file):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {data_file}")

    try:
        df = pd.read_parquet(data_file)
        total_records = len(df)  # Get the total number of records in the dataset
        start = (page - 1) * page_size
        end = start + page_size

        # Ensure the end index does not exceed the total number of records
        if end > total_records:
            end = total_records

        return {
            "split": split,
            "total_records": total_records,  # Correct total record count
            "page": page,
            "page_size": page_size,
            "records": df.iloc[start:end].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

@router.get("/{dataset_id}/query")
async def query_dataset(
    dataset_id: int,
    query: str = Query(..., description="SQL-like query string"),
    split: Optional[str] = Query(None, description="Dataset split to query"),
    db: Session = Depends(get_db)
):
    dataset_metadata = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    if dataset_metadata.subset:
        dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

    # If split is not provided, find the first available split
    if split is None:
        available_splits = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if not available_splits:
            raise HTTPException(status_code=404, detail="No splits found for this dataset")
        split = available_splits[0]

    data_file = os.path.join(dataset_path, split, "data.parquet")
    if not os.path.exists(data_file):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {data_file}")

    try:
        con = get_duckdb_connection()
        con.execute(f"CREATE TABLE IF NOT EXISTS dataset_{dataset_id} AS SELECT * FROM parquet_scan('{data_file}')")

        result_df = con.execute(query).fetchdf()
        con.close()

        return {
            "split": split,
            "total_records": len(result_df),
            "records": result_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error querying dataset: {str(e)}")

@router.post("/{dataset_id}/verify")
async def verify_dataset(dataset_id: int, db: Session = Depends(get_db)):
    logger.info(f"Received verification request for dataset id: {dataset_id}")
    success, message = verify_and_update_dataset_status(db, dataset_id)
    logger.info(f"Verification result: success={success}, message={message}")
    if success:
        return {"status": "success", "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)
