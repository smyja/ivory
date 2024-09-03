from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Dict
from models import DatasetRequest, DatasetMetadataResponse, DownloadStatus
from utils.database import get_db, download_and_save_dataset, get_active_downloads

router = APIRouter()

@router.post("/download")
async def download_dataset(request: DatasetRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        background_tasks.add_task(download_and_save_dataset, db, request)
        return {"message": "Dataset download started in the background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=List[DatasetMetadataResponse])
async def list_datasets(db: Session = Depends(get_db)):
    from models import DatasetMetadata
    datasets = db.query(DatasetMetadata).all()
    return datasets

@router.get("/query/{dataset_name}")
async def query_dataset(dataset_name: str, query: str, db: Session = Depends(get_db)):
    try:
        result = db.execute(f"SELECT * FROM {dataset_name} WHERE {query}")
        return [dict(row) for row in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active_downloads", response_model=Dict[int, Dict])
async def list_active_downloads():
    return get_active_downloads()