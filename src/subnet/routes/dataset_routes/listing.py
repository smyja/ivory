from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
import logging
import os
import pandas as pd

from models import (
    DatasetMetadataResponse,
    DatasetMetadata,
    DatasetDetailResponse,
    ClusteringHistory,
    DownloadStatusEnum,
)
from utils.database import (
    get_db,
    scan_existing_datasets,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[DatasetMetadataResponse])
async def list_datasets_endpoint(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """List available datasets with their status."""
    try:
        # First scan for any new datasets
        scan_existing_datasets(db)

        # Then get the datasets with pagination
        datasets = (
            db.query(DatasetMetadata)
            .order_by(DatasetMetadata.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        response_data = []
        for dataset in datasets:
            # Find the latest successful version for each dataset
            latest_history = (
                db.query(ClusteringHistory.clustering_version)
                .filter(ClusteringHistory.dataset_id == dataset.id)
                .filter(ClusteringHistory.clustering_status == "completed")
                .order_by(ClusteringHistory.clustering_version.desc())
                .first()
            )
            latest_version = latest_history[0] if latest_history else None
            # Use model_validate instead of from_orm
            response_item = DatasetMetadataResponse.model_validate(dataset)
            response_item.latest_version = latest_version
            response_data.append(response_item)
        return response_data
    except Exception as e:
        logger.exception(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}")
async def get_dataset_info(
    dataset_id: int,
    detail_level: Optional[str] = Query(
        "basic", description="Level of detail (basic/full/data)"
    ),
    page: Optional[int] = Query(1, ge=1, description="Page number for data retrieval"),
    page_size: Optional[int] = Query(
        50, ge=1, le=1000, description="Number of records per page"
    ),
    db: Session = Depends(get_db),
):
    """Get dataset information with configurable detail level."""
    try:
        dataset_metadata = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset_metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if detail_level == "data":
            # Use the Enum value for comparison
            if dataset_metadata.status != DownloadStatusEnum.COMPLETED.value:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset is not ready. Current status: {dataset_metadata.status}",
                )

            try:
                # Construct base path using only the dataset name
                dataset_name = dataset_metadata.name
                if dataset_metadata.hf_dataset_name:
                    dataset_name = dataset_metadata.hf_dataset_name

                # Replace slashes in the dataset name with underscores
                safe_name = dataset_name.replace("/", "_")
                dataset_path = os.path.join("datasets", safe_name)

                # Make sure the path exists
                if not os.path.exists(dataset_path):
                    logger.warning(f"Dataset directory not found: {dataset_path}")
                    return {
                        "error": "Dataset files not available",
                        "metadata": {
                            "id": dataset_metadata.id,
                            "name": dataset_metadata.name,
                            "status": dataset_metadata.status,
                        },
                        "pagination": {
                            "total": 0,
                            "page": 1,
                            "page_size": page_size,
                            "total_pages": 0,
                        },
                        "data": [],
                        "columns": [],
                    }

                # Find splits by listing directories in the dataset path
                available_splits = [
                    d
                    for d in os.listdir(dataset_path)
                    if os.path.isdir(os.path.join(dataset_path, d))
                ]

                if not available_splits:
                    logger.warning(
                        f"No splits found in dataset directory: {dataset_path}"
                    )
                    return {
                        "error": "No dataset splits available",
                        "metadata": {
                            "id": dataset_metadata.id,
                            "name": dataset_metadata.name,
                            "status": dataset_metadata.status,
                        },
                        "pagination": {
                            "total": 0,
                            "page": 1,
                            "page_size": page_size,
                            "total_pages": 0,
                        },
                        "data": [],
                        "columns": [],
                    }

                # Use the first available split or the hf_split if specified
                split = available_splits[0]
                if hasattr(dataset_metadata, "hf_split") and dataset_metadata.hf_split:
                    if dataset_metadata.hf_split in available_splits:
                        split = dataset_metadata.hf_split

                split_path = os.path.join(dataset_path, split)

                # Check for data files - prioritize Parquet
                parquet_path = os.path.join(split_path, "data.parquet")

                if os.path.exists(parquet_path):
                    # Use Parquet file directly - this is our preferred format
                    df = pd.read_parquet(parquet_path)

                    # Get column names
                    column_names = df.columns.tolist()

                    # Calculate total count
                    total_count = len(df)

                    # Get paginated data
                    offset = (page - 1) * page_size
                    end_idx = min(offset + page_size, total_count)
                    paginated_df = df.iloc[offset:end_idx]

                    # Convert to list of dicts
                    result_data = paginated_df.to_dict(orient="records")
                    file_format = "parquet"

                else:
                    logger.warning(f"Parquet data file not found at {parquet_path}")
                    return {
                        "error": "Dataset file not found",
                        "metadata": {
                            "id": dataset_metadata.id,
                            "name": dataset_metadata.name,
                            "status": dataset_metadata.status,
                            "path": parquet_path,
                        },
                        "pagination": {
                            "total": 0,
                            "page": 1,
                            "page_size": page_size,
                            "total_pages": 0,
                        },
                        "data": [],
                        "columns": [],
                    }

                # Return data with pagination info
                return {
                    "metadata": {
                        "id": dataset_metadata.id,
                        "name": dataset_metadata.name,
                        "status": dataset_metadata.status,
                    },
                    "data": result_data,
                    "pagination": {
                        "total": total_count,
                        "page": page,
                        "page_size": page_size,
                        "total_pages": (total_count + page_size - 1) // page_size,
                    },
                    "columns": column_names,
                    "format": file_format,
                }

            except Exception as e:
                logger.error(f"Error accessing dataset data: {str(e)}")
                return {
                    "error": str(e),
                    "metadata": {
                        "id": dataset_metadata.id,
                        "name": dataset_metadata.name,
                        "status": dataset_metadata.status,
                    },
                    "pagination": {
                        "total": 0,
                        "page": 1,
                        "page_size": page_size,
                        "total_pages": 0,
                    },
                    "data": [],
                    "columns": [],
                }

        # For basic or full detail level
        result = {"id": dataset_metadata.id, "name": dataset_metadata.name}

        if detail_level == "full":
            # Add more detailed info with safe attribute access
            # First build a base dict with all safe fields
            base_fields = {
                "status": dataset_metadata.status,
                "clustering_status": dataset_metadata.clustering_status,
                "is_clustered": dataset_metadata.is_clustered,
                "created_at": dataset_metadata.created_at,
            }

            # Carefully add optional fields with hasattr checks
            optional_fields = {}
            for field in [
                "download_date",
                "message",
                "updated_at",
                "source",
                "identifier",
            ]:
                if hasattr(dataset_metadata, field):
                    optional_fields[field] = getattr(dataset_metadata, field)

            # Update the result with both mandatory and optional fields
            result.update(base_fields)
            result.update(optional_fields)

            # Add HF specific fields if they exist
            hf_fields = {}
            for field in ["hf_dataset_name", "hf_config", "hf_split", "hf_revision"]:
                if hasattr(dataset_metadata, field) and getattr(
                    dataset_metadata, field
                ):
                    hf_fields[field] = getattr(dataset_metadata, field)

            if hf_fields:
                result.update(hf_fields)

        return result
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/status")
async def get_dataset_status(
    dataset_id: int,
    status_type: Optional[str] = Query(
        None, description="Type of status to return (general/clustering)"
    ),
    db: Session = Depends(get_db),
):
    """Get the status of a dataset (download or clustering)."""
    try:
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if status_type == "clustering":
            # Get clustering status
            latest_clustering = (
                db.query(ClusteringHistory)
                .filter(ClusteringHistory.dataset_id == dataset_id)
                .order_by(ClusteringHistory.clustering_version.desc())
                .first()
            )
            if not latest_clustering:
                return {"status": "not_started", "message": "Clustering not started"}

            return {
                "status": latest_clustering.clustering_status,
                "message": latest_clustering.message,
                "version": latest_clustering.clustering_version,
                "started_at": latest_clustering.started_at,
                "completed_at": latest_clustering.completed_at,
            }
        else:
            # Get general status (download status)
            return {
                "status": dataset.status,
                "message": dataset.message,
                "download_date": dataset.download_date,
                "is_clustered": dataset.is_clustered,
                "clustering_status": dataset.clustering_status,
            }
    except Exception as e:
        logger.error(f"Error getting dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
