from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import logging

from models import (
    DatasetRequest,
    DatasetMetadata,
    DatasetMetadataResponse,
    DownloadStatusEnum,
    DownloadStatusUpdate,
)
from utils.database import (
    get_db,
    download_and_save_dataset,
    scan_existing_datasets,
    get_active_downloads,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=DatasetMetadataResponse)
async def create_dataset_endpoint(
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a new dataset entry and start the download process."""
    try:
        # Check if dataset already exists
        existing_dataset = (
            db.query(DatasetMetadata)
            .filter(
                DatasetMetadata.name == request.name,
            )
            .first()
        )

        if existing_dataset:
            if existing_dataset.status == DownloadStatusEnum.COMPLETED.value:
                result = DatasetMetadataResponse.model_validate(existing_dataset)
                return result
            elif existing_dataset.status == DownloadStatusEnum.IN_PROGRESS.value:
                result = DatasetMetadataResponse.model_validate(existing_dataset)
                return result
            else:
                # If failed, delete the old record and try again
                db.delete(existing_dataset)
                db.commit()

        # Create a new metadata entry
        new_metadata = DatasetMetadata(
            name=request.name,
            description=request.description,
            source=request.source,
            identifier=request.identifier,
            hf_dataset_name=request.hf_dataset_name,
            hf_config=request.hf_config,
            hf_split=request.hf_split,
            hf_revision=request.hf_revision,
            status=DownloadStatusEnum.IN_PROGRESS.value,
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)

        # Start the download in the background, passing the ID and request data
        background_tasks.add_task(
            download_and_save_dataset, db, new_metadata.id, request
        )

        result = DatasetMetadataResponse.model_validate(new_metadata)
        return result
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init")
async def download_dataset(
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start a dataset download in the background."""
    try:
        # Check if dataset already exists
        existing_dataset = (
            db.query(DatasetMetadata)
            .filter(
                DatasetMetadata.name == request.name,
            )
            .first()
        )

        if existing_dataset:
            if existing_dataset.status == DownloadStatusEnum.COMPLETED.value:
                return {"message": "Dataset already exists", "id": existing_dataset.id}
            elif existing_dataset.status == DownloadStatusEnum.IN_PROGRESS.value:
                return {
                    "message": "Dataset download already in progress",
                    "id": existing_dataset.id,
                }
            else:
                # If failed, delete the old record and try again
                db.delete(existing_dataset)
                db.commit()

        # Start the download in the background
        # We need to create the metadata entry first to get an ID
        new_metadata = DatasetMetadata(
            name=request.name,
            description=request.description,
            source=request.source,
            identifier=request.identifier,
            hf_dataset_name=request.hf_dataset_name,
            hf_config=request.hf_config,
            hf_split=request.hf_split,
            hf_revision=request.hf_revision,
            status=DownloadStatusEnum.IN_PROGRESS.value,
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)

        # Pass the new ID and the request to the background task
        background_tasks.add_task(
            download_and_save_dataset, db, new_metadata.id, request
        )

        return {"message": "Download started", "id": new_metadata.id}
    except Exception as e:
        logger.error(f"Error starting dataset download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active_downloads")
async def list_active_downloads(db: Session = Depends(get_db)):
    return get_active_downloads(db)


@router.get("/{dataset_id}/verify")
async def verify_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Verify the status of a dataset and update it in the database."""
    try:
        from utils.database import verify_and_update_dataset_status

        return await verify_and_update_dataset_status(dataset_id, db)
    except Exception as e:
        logger.error(f"Error verifying dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update_status")
async def update_download_status(
    dataset_id: int, status_update: DownloadStatusUpdate, db: Session = Depends(get_db)
):
    """Update the status of a dataset download."""
    try:
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        dataset.status = status_update.status
        dataset.message = status_update.message
        db.commit()
        return {"message": "Status updated", "id": dataset_id}
    except Exception as e:
        logger.error(f"Error updating dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
