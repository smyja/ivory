from fastapi import APIRouter, HTTPException, Path, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional
from enum import Enum
import logging

from models import DatasetMetadata, DownloadStatusEnum
from utils.database import get_db
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class StatusType(str, Enum):
    DOWNLOAD = "download"
    CLUSTERING = "clustering"


# Define clustering status constants since ClusteringStatusEnum is not available
CLUSTERING_STATUS_COMPLETED = "completed"
CLUSTERING_STATUS_FAILED = "failed"
CLUSTERING_STATUS_PROCESSING = "processing"


class StatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None


@router.get("/{dataset_id}/status", response_model=StatusResponse)
async def get_dataset_status(
    dataset_id: int = Path(..., description="Dataset ID"),
    status_type: StatusType = Query(
        StatusType.DOWNLOAD, description="Type of status to retrieve"
    ),
    db: Session = Depends(get_db),
):
    """Get the status of a dataset's download or clustering process."""
    try:
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )

        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"Dataset with ID {dataset_id} not found"
            )

        if status_type == StatusType.DOWNLOAD:
            # Get download status
            if dataset.status == DownloadStatusEnum.COMPLETED.value:
                return StatusResponse(
                    status="completed",
                    progress=100,
                    message="Download completed successfully",
                )
            elif dataset.status == DownloadStatusEnum.FAILED.value:
                error_message = "Download failed"
                if hasattr(dataset, "error_message") and dataset.error_message:
                    error_message = dataset.error_message

                return StatusResponse(
                    status="failed", progress=0, message=error_message
                )
            elif dataset.status == DownloadStatusEnum.IN_PROGRESS.value:
                # Attempt to estimate progress (this would need to be implemented)
                # For now, return a default progress value
                return StatusResponse(
                    status="downloading",
                    progress=50,  # Placeholder - in a real implementation you'd track actual progress
                    message="Download in progress",
                )
            else:
                return StatusResponse(
                    status="pending", progress=0, message="Download not started yet"
                )

        elif status_type == StatusType.CLUSTERING:
            # Get clustering status
            if not hasattr(dataset, "clustering_status"):
                return StatusResponse(
                    status="not_started",
                    progress=0,
                    message="Clustering not available for this dataset",
                )

            if dataset.clustering_status == CLUSTERING_STATUS_COMPLETED:
                return StatusResponse(
                    status="completed",
                    progress=100,
                    message="Clustering completed successfully",
                )
            elif dataset.clustering_status == CLUSTERING_STATUS_FAILED:
                return StatusResponse(
                    status="failed", progress=0, message="Clustering failed"
                )
            elif dataset.clustering_status == CLUSTERING_STATUS_PROCESSING:
                # You could add more detailed progress tracking here
                return StatusResponse(
                    status="processing",
                    progress=50,  # Placeholder - in a real implementation you'd track actual progress
                    message="Clustering in progress",
                )
            else:
                return StatusResponse(
                    status="not_started",
                    progress=0,
                    message="Clustering not started yet",
                )

        # Should never reach here due to enum validation
        raise HTTPException(status_code=400, detail="Invalid status type")

    except Exception as e:
        logger.error(f"Error in get_dataset_status: {str(e)}")
        return StatusResponse(
            status="error", progress=0, message=f"Error fetching status: {str(e)}"
        )
