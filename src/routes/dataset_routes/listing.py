from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any, Union
import logging
import os
import pandas as pd
from pydantic import BaseModel
import json

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

# --- Pydantic Models for Data Response ---


class DatasetDataPagination(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int


class DatasetSimpleMetadata(BaseModel):
    id: int
    name: str
    status: str


class DatasetData(BaseModel):
    # Define structure for individual records if known, otherwise use Dict
    # Using Dict[str, Any] for flexibility as columns can vary
    pass  # Let FastAPI handle dict serialization for now, assuming basic types


class SplitInfo(BaseModel):
    name: str
    count: int


class DatasetDataResponse(BaseModel):
    metadata: DatasetSimpleMetadata
    data: List[Dict[str, Any]]
    pagination: DatasetDataPagination
    columns: List[str]
    format: str
    splits_info: Optional[List[SplitInfo]] = None
    error: Optional[str] = None


# Response model for basic/full info (can refine later if needed)
class DatasetInfoResponse(BaseModel):
    id: int
    name: str
    status: Optional[str] = None
    clustering_status: Optional[str] = None
    is_clustered: Optional[bool] = None
    created_at: Optional[Any] = None  # Use Any for datetime for now
    download_date: Optional[Any] = None
    message: Optional[str] = None
    updated_at: Optional[Any] = None
    source: Optional[str] = None
    identifier: Optional[str] = None
    hf_dataset_name: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None
    hf_revision: Optional[str] = None
    error: Optional[str] = None  # Include optional error field


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


@router.get(
    "/{dataset_id}", response_model=Union[DatasetDataResponse, DatasetInfoResponse]
)
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
                # Construct base path using the dataset name stored in metadata
                dataset_name = dataset_metadata.name
                safe_name = dataset_name.replace("/", "_").replace("\\", "_")
                base_dataset_path = os.path.join("datasets", safe_name)

                # --- Determine the specific sub-path based on stored metadata --- #
                path_parts = [base_dataset_path]

                # Use stored config if available
                config_part = None
                if (
                    hasattr(dataset_metadata, "hf_config")
                    and dataset_metadata.hf_config
                ):
                    config_part = dataset_metadata.hf_config.replace("/", "_")
                    path_parts.append(config_part)
                else:
                    # Attempt to find a default or first config directory if not stored
                    # This might be needed for datasets not imported via HF flow
                    if os.path.exists(base_dataset_path):
                        potential_configs = [
                            d
                            for d in os.listdir(base_dataset_path)
                            if os.path.isdir(os.path.join(base_dataset_path, d))
                            and d != "_all_splits_"
                        ]  # Exclude the split marker
                        if potential_configs:  # Check if list is not empty
                            config_part = potential_configs[
                                0
                            ]  # Take the first found dir as config
                            path_parts.append(config_part)
                        # If no config dirs, maybe data is directly under base_dataset_path/_all_splits_ ?

                # Use stored split if available, otherwise default to _all_splits_
                split_part = "_all_splits_"  # Default
                if hasattr(dataset_metadata, "hf_split") and dataset_metadata.hf_split:
                    split_part = dataset_metadata.hf_split.replace("/", "_")
                elif os.path.exists(os.path.join(*path_parts, "_all_splits_")):
                    # Keep the default if _all_splits_ exists
                    pass
                elif os.path.exists(base_dataset_path):  # Check base path again
                    # If specific split wasn't stored and _all_splits_ doesn't exist,
                    # try to find the first actual split directory
                    config_path_to_check = os.path.join(
                        *path_parts
                    )  # Path including potential config
                    if os.path.exists(config_path_to_check):
                        potential_splits = [
                            d
                            for d in os.listdir(config_path_to_check)
                            if os.path.isdir(os.path.join(config_path_to_check, d))
                        ]
                        if potential_splits:  # Check if list is not empty
                            split_part = potential_splits[0]

                primary_path_parts = path_parts + [
                    split_part
                ]  # Keep original calculation separate

                # Final path to the directory containing data.parquet (Primary Attempt)
                final_data_dir = os.path.join(*primary_path_parts)
                parquet_path = os.path.join(final_data_dir, "data.parquet")
                logger.info(
                    f"Attempting to load data from primary path: {parquet_path}"
                )

                # --- Fallback Logic ---
                if not os.path.exists(parquet_path):
                    logger.warning(
                        f"Primary path not found: {parquet_path}. Trying alternatives..."
                    )
                    found_alternative = False
                    alternative_paths_to_check = []

                    # Alt 1: The CORRECT standard location - _all_splits_ at same level as train
                    alternative_paths_to_check.append(
                        os.path.join(base_dataset_path, "_all_splits_", "data.parquet")
                    )

                    # Alt 2: Train split (if we are going to support individual split selection later)
                    alternative_paths_to_check.append(
                        os.path.join(base_dataset_path, "train", "data.parquet")
                    )

                    # Legacy fallbacks for backward compatibility with existing dataset structures:

                    # Alt 3: 'train' split under config (if config exists) - Old structure
                    if config_part:
                        alternative_paths_to_check.append(
                            os.path.join(
                                base_dataset_path, config_part, "train", "data.parquet"
                            )
                        )

                    # Alt 4: Nested _all_splits_ inside train - Old incorrect structure
                    if config_part:
                        alternative_paths_to_check.append(
                            os.path.join(
                                base_dataset_path,
                                config_part,
                                "train",
                                "_all_splits_",
                                "data.parquet",
                            )
                        )
                    alternative_paths_to_check.append(
                        os.path.join(
                            base_dataset_path, "train", "_all_splits_", "data.parquet"
                        )
                    )

                    # Alt 5: Directly under config (if config exists)
                    if config_part:
                        alternative_paths_to_check.append(
                            os.path.join(base_dataset_path, config_part, "data.parquet")
                        )

                    # Alt 6: Directly under base
                    alternative_paths_to_check.append(
                        os.path.join(base_dataset_path, "data.parquet")
                    )

                    for alt_path in alternative_paths_to_check:
                        logger.info(f"Checking alternative path: {alt_path}")
                        if os.path.exists(alt_path):
                            parquet_path = alt_path
                            logger.info(
                                f"Found data at alternative path: {parquet_path}"
                            )
                            found_alternative = True
                            break  # Use the first alternative found

                    if not found_alternative:
                        logger.error(
                            f"Parquet data file not found at primary path or alternatives."
                        )
                        logger.warning(
                            f"Primary path checked: {os.path.join(final_data_dir, 'data.parquet')}"
                        )
                        logger.warning(
                            f"Base dataset path checked: {base_dataset_path}"
                        )
                        return DatasetDataResponse(
                            error="Dataset Parquet file not found at expected location or alternatives.",
                            metadata=DatasetSimpleMetadata(
                                id=dataset_metadata.id,
                                name=dataset_metadata.name,
                                status=dataset_metadata.status,
                            ),
                            pagination=DatasetDataPagination(
                                total=0, page=page, page_size=page_size, total_pages=0
                            ),
                            data=[],
                            columns=[],
                            format="unknown",
                        )
                else:
                    logger.info(f"Found data at primary path: {parquet_path}")

                # Use Parquet file directly (parquet_path is now either the primary or a found alternative)
                df = pd.read_parquet(parquet_path)

                # Extract available splits and counts if column exists
                splits_info = None
                if "_hf_split" in df.columns:
                    split_counts = df["_hf_split"].value_counts().reset_index()
                    split_counts.columns = [
                        "name",
                        "count",
                    ]  # Rename columns for clarity
                    splits_info = split_counts.to_dict("records")
                    logger.info(f"Found splits and counts: {splits_info}")

                # Get column names, EXCLUDING the internal _hf_split column
                column_names = [col for col in df.columns if col != "_hf_split"]

                # Calculate total count
                total_count = len(df)

                # Get paginated data
                offset = (page - 1) * page_size
                end_idx = min(offset + page_size, total_count)
                paginated_df = df.iloc[offset:end_idx]

                # Convert to serializable format using pandas json methods
                # Drop the internal _hf_split column before serialization for the data payload
                data_to_serialize = paginated_df.drop(
                    columns=["_hf_split"], errors="ignore"
                )
                result_data = json.loads(
                    data_to_serialize.to_json(orient="records", date_format="iso")
                )

                # Return data with pagination info and splits
                return DatasetDataResponse(
                    metadata=DatasetSimpleMetadata(
                        id=dataset_metadata.id,
                        name=dataset_metadata.name,
                        status=dataset_metadata.status,
                    ),
                    data=result_data,
                    pagination=DatasetDataPagination(
                        total=total_count,
                        page=page,
                        page_size=page_size,
                        total_pages=(total_count + page_size - 1) // page_size,
                    ),
                    columns=column_names,  # Return columns without _hf_split
                    format="parquet",
                    splits_info=splits_info,  # Return the list of split info objects
                )

            except Exception as e:
                logger.error(f"Error accessing dataset data: {str(e)}", exc_info=True)
                return DatasetDataResponse(
                    error=str(e),
                    metadata=DatasetSimpleMetadata(
                        id=dataset_metadata.id,
                        name=dataset_metadata.name,
                        status=dataset_metadata.status,
                    ),
                    pagination=DatasetDataPagination(
                        total=0, page=page, page_size=page_size, total_pages=0
                    ),
                    data=[],
                    columns=[],
                    format="unknown",
                )

        # For basic or full detail level - return structure matching DatasetInfoResponse
        # Need to reconstruct the 'result' dict carefully to match the model
        info_response_data = {
            "id": dataset_metadata.id,
            "name": dataset_metadata.name,
            "status": dataset_metadata.status,
            "clustering_status": dataset_metadata.clustering_status,
            "is_clustered": dataset_metadata.is_clustered,
            "created_at": dataset_metadata.created_at,
        }
        if detail_level == "full":
            # Add optional fields safely
            for field in [
                "download_date",
                "message",
                "updated_at",
                "source",
                "identifier",
                "hf_dataset_name",
                "hf_config",
                "hf_split",
                "hf_revision",
            ]:
                if hasattr(dataset_metadata, field):
                    value = getattr(dataset_metadata, field)
                    if value is not None:  # Add only if not None
                        info_response_data[field] = value

        # Validate and return using the Pydantic model
        return DatasetInfoResponse(**info_response_data)

    except HTTPException as http_exc:
        raise http_exc  # Re-raise FastAPI/manual HTTP exceptions
    except Exception as e:
        logger.error(
            f"Error getting dataset info for ID {dataset_id}: {str(e)}", exc_info=True
        )
        # Return a generic error using one of the response models (e.g., DatasetInfoResponse)
        return DatasetInfoResponse(id=dataset_id, name="Error", error=str(e))


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
            status_info = {
                "status": dataset.status,
            }

            # Safely add all optional attributes with hasattr checks
            for field in [
                "download_date",
                "is_clustered",
                "clustering_status",
                "message",
                "error_message",
            ]:
                if hasattr(dataset, field):
                    value = getattr(dataset, field)
                    if value is not None:
                        # Use the field name as is, except for error_message which maps to message
                        key = "message" if field == "error_message" else field
                        status_info[key] = value

            return status_info
    except Exception as e:
        logger.error(f"Error getting dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
