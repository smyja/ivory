from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Optional, Any, Tuple
import logging
import os
import pandas as pd
import requests
import json
from models import (
    Base,
    DatasetRequest,
    DatasetMetadataResponse,
    DatasetMetadata,
    TextDB,
    Category,
    Level1Cluster,
    TextAssignment,
    ClusteringHistory,
    DatasetDetailResponse,
    CategoryResponse,
    Level1ClusterResponse,
    TextDBResponse,
    DownloadStatus,
    DownloadStatusUpdate,
    DownloadStatusEnum,
    ClusteringStatus,
)
from utils.database import (
    get_db,
    SessionLocal,
    download_and_save_dataset,
    verify_and_update_dataset_status,
    save_clustering_results,
    scan_existing_datasets,
    get_duckdb_connection,
)
from utils.clustering import cluster_texts
from sqlalchemy import func, desc
from datetime import datetime
from openai import OpenAI
from utils.huggingface_utils import get_dataset_configs

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants for clustering
BATCH_SIZE = 100  # Number of texts to process at once for embeddings
DISTANCE_THRESHOLD = 0.5  # Clustering distance threshold

# Model configurations
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENROUTER_LLM_MODEL = "anthropic/claude-3.7-sonnet"


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def generate_category_name(texts: List[str]) -> str:
    """Generate a category name for a cluster of texts using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY environment variable is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = (
        "Create a specific and concise category name (1-3 words) for the following texts. "
        "The category should be descriptive and focused on the main theme.\n\n"
        "Texts:\n" + "\n".join(texts)  # Removed limit
    )

    response = client.chat.completions.create(
        model=OPENROUTER_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in creating concise, specific category names.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,
        temperature=0.3,
        extra_headers={
            "HTTP-Referer": "https://github.com/ivory",
            "X-Title": "Ivory",
        },
    )
    return response.choices[0].message.content.strip()


# Comment out the unused generate_subcluster_name function
"""
async def generate_subcluster_name(texts: List[str]) -> str:
    # DEPRECATED: This function is no longer used with the new clustering approach
    pass
"""


@router.post("/download")
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
                DatasetMetadata.name == request.dataset_name,
                DatasetMetadata.subset == request.subset,
                DatasetMetadata.split == request.split,
            )
            .first()
        )

        if existing_dataset:
            if existing_dataset.status == DownloadStatus.COMPLETED:
                return {"message": "Dataset already exists", "id": existing_dataset.id}
            elif existing_dataset.status == DownloadStatus.IN_PROGRESS:
                return {
                    "message": "Dataset download already in progress",
                    "id": existing_dataset.id,
                }
            else:
                # If failed, delete the old record and try again
                db.delete(existing_dataset)
                db.commit()

        # Start the download in the background
        background_tasks.add_task(download_and_save_dataset, db, request)

        # Create a new metadata entry
        new_metadata = DatasetMetadata(
            name=request.dataset_name,
            subset=request.subset,
            split=request.split,
            status=DownloadStatus.IN_PROGRESS,
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)

        return {"message": "Download started", "id": new_metadata.id}
    except Exception as e:
        logger.error(f"Error starting dataset download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[DatasetMetadataResponse])
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


@router.get("/active_downloads")
async def list_active_downloads(db: Session = Depends(get_db)):
    return get_active_downloads(db)


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
                    detail=f"Dataset is not ready. Current status: {dataset_metadata.status}",  # Use the stored status directly
                )

            # Construct base path using only the dataset name
            dataset_path = os.path.join(
                "datasets", dataset_metadata.name.replace("/", "_")
            )
            # Removed subset logic

            # Find splits by listing directories in the dataset path
            if not os.path.exists(dataset_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset directory not found: {dataset_path}",
                )
            available_splits = [
                d
                for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))
            ]
            if not available_splits:
                raise HTTPException(
                    status_code=404,
                    detail=f"No splits found in dataset directory: {dataset_path}",
                )

            # Use the first available split
            split = available_splits[0]
            split_path = os.path.join(dataset_path, split)

            # Find parquet files
            parquet_files = [
                f for f in os.listdir(split_path) if f.endswith(".parquet")
            ]
            if not parquet_files:
                raise HTTPException(
                    status_code=404, detail=f"No parquet files found in split: {split}"
                )

            parquet_file = os.path.join(split_path, parquet_files[0])
            if not os.path.exists(parquet_file):
                raise HTTPException(status_code=404, detail="Dataset file not found")

            # Calculate offset for pagination
            offset = (page - 1) * page_size

            # Use DuckDB for data retrieval with pagination
            with get_duckdb_connection() as conn:
                # Get total count
                total_records = conn.execute(
                    f"SELECT COUNT(*) FROM '{parquet_file}'"
                ).fetchone()[0]

                # Get paginated data
                df = conn.execute(
                    f"SELECT * FROM '{parquet_file}' LIMIT {page_size} OFFSET {offset}"
                ).df()

                return {
                    "split": split,
                    "total_records": total_records,
                    "page": page,
                    "page_size": page_size,
                    "records": df.to_dict(orient="records"),
                }

        elif detail_level == "full":
            # Get detailed dataset information including splits and sample data
            dataset_path = os.path.join(
                "datasets", dataset_metadata.name.replace("/", "_")
            )
            # Removed subset logic

            if not os.path.exists(dataset_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset directory not found: {dataset_path}",
                )

            splits = [
                split
                for split in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, split))
            ]

            if not splits:
                raise HTTPException(
                    status_code=404,
                    detail=f"No splits found in dataset directory: {dataset_path}",
                )

            rows_info = {}
            with get_duckdb_connection() as conn:
                for split in splits:
                    split_path = os.path.join(dataset_path, split)
                    parquet_files = [
                        f for f in os.listdir(split_path) if f.endswith(".parquet")
                    ]

                    if not parquet_files:
                        continue

                    data_file = os.path.join(split_path, parquet_files[0])
                    try:
                        conn.execute(
                            f"CREATE OR REPLACE TABLE temp AS SELECT * FROM parquet_scan('{data_file}')"
                        )
                        columns = conn.execute("SELECT * FROM temp LIMIT 0").description
                        row_count = conn.execute(
                            "SELECT COUNT(*) FROM temp"
                        ).fetchone()[0]
                        sample_rows = (
                            conn.execute("SELECT * FROM temp LIMIT 5")
                            .fetchdf()
                            .to_dict(orient="records")
                        )
                        rows_info[split] = {
                            "columns": [col[0] for col in columns],
                            "row_count": row_count,
                            "sample_rows": sample_rows,
                        }
                    except Exception as e:
                        logger.error(
                            f"Error processing Parquet file {data_file}: {str(e)}"
                        )
                        rows_info[split] = {"error": str(e)}

            return {
                "id": dataset_metadata.id,
                "name": dataset_metadata.name,
                # "subset": dataset_metadata.subset, # Removed
                "status": dataset_metadata.status,
                "splits": rows_info,
            }
        else:
            # Basic information
            return {
                "id": dataset_metadata.id,
                "name": dataset_metadata.name,
                # "subset": dataset_metadata.subset, # Removed
                "status": dataset_metadata.status,
                # "download_date": dataset_metadata.download_date, # download_date removed from model
                "created_at": dataset_metadata.created_at,  # Use created_at instead
            }

    except Exception as e:
        logger.error(f"Error getting dataset information: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/verify")
async def verify_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Verify a dataset's integrity."""
    try:
        success, message = verify_and_update_dataset_status(db, dataset_id)
        if not success:
            raise HTTPException(status_code=400, detail=message)
        return {"message": message}
    except Exception as e:
        logger.error(f"Error verifying dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update_status")
async def update_download_status(
    dataset_id: int, status_update: DownloadStatusUpdate, db: Session = Depends(get_db)
):
    """Update the download status of a dataset."""
    # Find the dataset record
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Update the status using the enum value
    dataset_metadata.status = status_update.status
    db.commit()

    return {"message": "Status updated successfully"}


@router.get("/{dataset_id}/status")
async def get_dataset_status(
    dataset_id: int,
    status_type: Optional[str] = Query(
        None, description="Type of status to return (general/clustering)"
    ),
    db: Session = Depends(get_db),
):
    """Get the status of a dataset (general or clustering)."""
    try:
        dataset_metadata = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset_metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if status_type == "clustering":
            if dataset_metadata.clustering_status == "failed":
                raise HTTPException(
                    status_code=500,
                    detail="Clustering failed. Please try again.",
                )
            return {"status": dataset_metadata.clustering_status}
        else:
            return {
                "id": dataset_metadata.id,
                "dataset": dataset_metadata.name,
                "status": dataset_metadata.status,
                "download_date": dataset_metadata.download_date,
            }
    except Exception as e:
        logger.error(f"Error getting dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Comment out the old process_clustering endpoint
"""
@router.post("/{dataset_id}/cluster")
async def process_clustering(dataset_id: int, db: Session = Depends(get_db)):
    # DEPRECATED: Use trigger_clustering_endpoint instead
    # This endpoint uses the old clustering approach with Subcluster model
    # which has been replaced with Level1Cluster
    pass
"""


# Comment out the old clustering_status endpoint
"""
@router.get("/{dataset_id}/clustering_status")
async def get_clustering_status(dataset_id: int, db: Session = Depends(get_db)):
    # DEPRECATED: Use get_dataset_status with status_type=clustering instead
    pass
"""


# Comment out the old get_clusters endpoint
"""
@router.get("/{dataset_id}/clusters")
async def get_clusters(
    dataset_id: int, version: Optional[int] = None, db: Session = Depends(get_db)
):
    # DEPRECATED: Use get_dataset_details_endpoint instead 
    # This endpoint uses the old clustering approach with Subcluster model
    # which has been replaced with Level1Cluster
    pass
"""


@router.get("/clustering/history", response_model=List[Dict])
async def get_clustering_history(
    dataset_id: Optional[int] = None,
    clustering_version: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get clustering history, optionally filtered by dataset ID and version."""
    try:
        # Join with DatasetMetadata to get the name
        query = (
            db.query(ClusteringHistory, DatasetMetadata.name)
            .join(DatasetMetadata, ClusteringHistory.dataset_id == DatasetMetadata.id)
            .order_by(ClusteringHistory.created_at.desc())
        )
        if dataset_id is not None:
            query = query.filter(ClusteringHistory.dataset_id == dataset_id)
        if clustering_version is not None:
            query = query.filter(
                ClusteringHistory.clustering_version == clustering_version
            )

        history_results = query.all()
        # Construct the response including dataset_name and version
        return [
            {
                "id": h.id,
                "dataset_id": h.dataset_id,
                "dataset_name": dataset_name,  # Include dataset name
                "version": h.clustering_version,  # Include version
                "clustering_status": h.clustering_status,
                # titling_status removed as it's not in the model
                "created_at": h.created_at.isoformat() if h.created_at else None,
                "completed_at": h.completed_at.isoformat() if h.completed_at else None,
                "error_message": h.error_message,
                "details": h.details,
            }
            for h, dataset_name in history_results  # Unpack results
        ]
    except Exception as e:
        logger.error(f"Error getting clustering history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting clustering history: {e}"
        )


# Comment out the old get_subcluster_texts endpoint
"""
@router.get("/subclusters/{subcluster_id}/texts")
async def get_subcluster_texts(subcluster_id: int, db: Session = Depends(get_db)):
    # DEPRECATED: Use level1_clusters/{level1_cluster_id}/texts instead
    # This endpoint uses the old clustering approach with Subcluster model
    # which has been replaced with Level1Cluster
    pass
"""


@router.get("/{dataset_id}/clustering/versions", response_model=List[int])
async def get_clustering_versions(dataset_id: int, db: Session = Depends(get_db)):
    """Get available clustering versions for a dataset."""
    try:
        versions = (
            db.query(ClusteringHistory.clustering_version)
            .filter(ClusteringHistory.dataset_id == dataset_id)
            .filter(ClusteringHistory.clustering_status == "completed")
            .order_by(ClusteringHistory.clustering_version.desc())
            .all()
        )
        return [v[0] for v in versions]
    except Exception as e:
        logger.error(f"Error getting clustering versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting clustering versions: {e}"
        )


@router.post("", response_model=DatasetMetadataResponse)
async def create_dataset_endpoint(
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a new dataset, supporting HuggingFace and other sources."""
    try:
        # Check if dataset with same name already exists
        existing = (
            db.query(DatasetMetadata)
            .filter(DatasetMetadata.name == request.name)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset with name '{request.name}' already exists",
            )

        # Initialize new dataset metadata
        new_dataset = DatasetMetadata(
            name=request.name,
            description=request.description,
            source=request.source,
            identifier=request.identifier,  # Use the primary identifier field
            status="pending",
            verification_status="pending",
            clustering_status="pending",
            is_clustered=False,
        )

        # Handle different data sources
        if request.source.lower() == "huggingface":
            if not request.hf_dataset_name:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required field: hf_dataset_name for HuggingFace source",
                )

            # Set Hugging Face specific fields
            new_dataset.hf_dataset_name = request.hf_dataset_name
            new_dataset.hf_config = request.hf_config
            new_dataset.hf_split = request.hf_split
            new_dataset.hf_revision = request.hf_revision  # Store requested revision

            # Initial save to get an ID
            db.add(new_dataset)
            db.commit()
            db.refresh(new_dataset)

            # Start download in background
            background_tasks.add_task(
                process_huggingface_dataset,
                # Pass db session creator if needed, or rely on SessionLocal inside task
                new_dataset.id,
                request.hf_dataset_name,
                request.hf_config,
                request.hf_split,
                request.hf_revision,  # Pass revision
                request.hf_token,
                request.text_field,  # Pass text field hint
                request.label_field,  # Pass label field hint
                request.selected_columns,  # Pass selected columns
                request.limit_rows,
            )

            # Update status to downloading
            new_dataset.status = "downloading"
            db.commit()

            logger.info(
                f"Started HuggingFace dataset download: {request.hf_dataset_name} (Rev: {request.hf_revision})"
            )

        else:
            # Handle other data sources (placeholder for future implementation)
            raise HTTPException(
                status_code=400,
                detail=f"Data source '{request.source}' not supported yet",
            )

        # Return the created dataset metadata
        response_data = DatasetMetadataResponse.model_validate(new_dataset)
        # You might want to add other relevant fields from the request to the response
        return response_data

    except Exception as e:
        logger.exception(f"Error creating dataset: {e}")
        db.rollback()  # Ensure rollback on error
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/cluster", status_code=202)
async def trigger_clustering_endpoint(
    dataset_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Triggers the scalable clustering process in the background."""
    # 1. Find dataset
    dataset = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.clustering_status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Clustering is already in progress for this dataset.",
        )

    # 2. Determine the next version
    latest_version_result = (
        db.query(func.max(ClusteringHistory.clustering_version))
        .filter(ClusteringHistory.dataset_id == dataset_id)
        .scalar()
    )
    next_version = (latest_version_result or 0) + 1
    logger.info(f"Dataset {dataset_id}: Triggering clustering version {next_version}.")

    # 3. Fetch texts and create ID map
    logger.info(f"Dataset {dataset_id}: Fetching texts for clustering...")
    texts_with_ids = (
        db.query(TextDB.id, TextDB.text).filter(TextDB.dataset_id == dataset_id).all()
    )

    if not texts_with_ids:
        raise HTTPException(
            status_code=400, detail="Dataset contains no texts to cluster."
        )

    logger.info(f"Dataset {dataset_id}: Found {len(texts_with_ids)} texts.")

    # 4. Add the background task
    # Need to pass a *new* db session to the background task
    background_tasks.add_task(
        run_clustering_task,
        dataset_id=dataset_id,
        version=next_version,
        texts_with_ids=texts_with_ids,  # Pass texts and IDs
        db=SessionLocal(),  # Create a new session for the task
    )

    # 5. Update status immediately and return
    dataset.clustering_status = "processing"  # Mark as processing immediately
    db.commit()

    return {
        "message": f"Clustering version {next_version} started for dataset {dataset_id}"
    }


@router.get("/{dataset_id}/details")
async def get_dataset_details_endpoint(
    dataset_id: int, version: Optional[int] = None, db: Session = Depends(get_db)
):
    """Get dataset details, optionally including clustering results for a specific version."""
    dataset = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Find the latest successful version if none specified
    target_version = version
    if target_version is None:
        latest_history = (
            db.query(ClusteringHistory.clustering_version)
            .filter(ClusteringHistory.dataset_id == dataset.id)
            .filter(ClusteringHistory.clustering_status == "completed")
            .order_by(ClusteringHistory.clustering_version.desc())
            .first()
        )
        target_version = latest_history[0] if latest_history else None

    # Use model_validate instead of from_orm
    response_data = DatasetDetailResponse.model_validate(dataset)
    response_data.latest_version = (
        target_version  # Show which version we're potentially loading
    )
    response_data.categories = []  # Initialize categories list

    # Only attempt to load categories if we have a valid version
    if target_version is not None:
        logger.info(
            f"Loading clustering results for dataset {dataset_id}, version {target_version}"
        )

        # Calculate total texts for this specific dataset version
        # This requires querying TextAssignment based on Level1Cluster -> Category -> dataset_id and version
        total_texts_in_version = (
            db.query(func.count(TextAssignment.id))
            .join(Level1Cluster, TextAssignment.level1_cluster_id == Level1Cluster.id)
            .join(Category, Level1Cluster.category_id == Category.id)
            .filter(Category.dataset_id == dataset_id)
            .filter(Category.version == target_version)
            .scalar()
        ) or 0
        response_data.dataset_total_texts = total_texts_in_version
        logger.info(
            f"Total texts found for dataset {dataset_id} version {target_version}: {total_texts_in_version}"
        )

        # Query categories with eager loading
        categories = (
            db.query(Category)
            .options(
                joinedload(Category.level1_clusters).joinedload(
                    Level1Cluster.text_assignments
                )
                # .joinedload(TextAssignment.text) # Maybe remove text loading here if not needed directly
            )
            .filter(Category.dataset_id == dataset_id)
            .filter(Category.version == target_version)
            .order_by(Category.name)
            .all()
        )
        logger.info(
            f"Found {len(categories)} categories for dataset {dataset_id}, version {target_version}"
        )

        # Process each category
        for cat in categories:
            # Calculate total texts in this category
            total_texts_in_category = sum(
                len(l1.text_assignments) for l1 in cat.level1_clusters
            )

            # Create category response object
            cat_resp = CategoryResponse.model_validate(cat)
            cat_resp.category_text_count = (
                total_texts_in_category  # Assign category text count
            )
            cat_resp.level1_clusters = []

            # Process each level1_cluster in this category
            for l1 in sorted(cat.level1_clusters, key=lambda x: x.title):
                # Create level1_cluster response object
                l1_resp = Level1ClusterResponse.model_validate(l1)
                l1_resp.text_count = len(l1.text_assignments)

                # Add all texts from text_assignments (or keep limit if preferred)
                l1_resp.texts = [
                    TextDBResponse.model_validate(assign.text)
                    for assign in l1.text_assignments  # Removed [:50] limit as requested
                ]

                cat_resp.level1_clusters.append(l1_resp)

            response_data.categories.append(cat_resp)

    else:
        logger.info(
            f"No specific or latest completed version found for dataset {dataset_id}"
        )
        response_data.dataset_total_texts = 0  # Set total to 0 if no version

    logger.info(f"Response data before serialization: {response_data}")
    return response_data.model_dump()


# --- Helper Function for Background Task ---


async def run_clustering_task(
    dataset_id: int, version: int, texts_with_ids: List[Tuple[int, str]], db: Session
):
    """The actual clustering logic run in the background."""
    history_entry = None
    start_time = datetime.utcnow()
    try:
        # 1. Create initial history entry
        history_entry = ClusteringHistory(
            dataset_id=dataset_id,
            clustering_version=version,
            clustering_status="started",
            created_at=start_time,
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry)
        logger.info(f"Dataset {dataset_id} v{version}: Clustering task started.")

        # Update main metadata status
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).update(
            {
                "clustering_status": "processing",
                "is_clustered": False,  # Ensure false until completion
            }
        )
        db.commit()

        # Extract texts and create the content->ID map
        original_texts = [text for _, text in texts_with_ids]
        text_content_to_db_id_map = {text: id for id, text in texts_with_ids}

        # Update history status
        history_entry.clustering_status = (
            "clustering_l1"  # Or more granular if possible
        )
        db.commit()

        # 2. Call the scalable clustering function
        # cluster_texts now returns a tuple (final_categories, final_subclusters)
        # It no longer returns a dict with 'assignments' or 'category_names'
        clustering_results_tuple = await cluster_texts(
            texts=original_texts, db=db, dataset_id=dataset_id, version=version
        )

        # Check if the function indicated failure (e.g., by returning None or raising an exception handled above)
        # We rely on the exception handling within cluster_texts or the outer try/except here.
        # If it returns normally, we assume it saved results correctly.
        logger.info(
            f"Dataset {dataset_id} v{version}: cluster_texts function completed."
        )

        # Update history status (assuming saving is done within cluster_texts or save_clustering_results)
        history_entry.clustering_status = (
            "saving"  # Or potentially completed if saving is synchronous
        )
        db.commit()

        # 5. Update final status on success
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        history_entry.clustering_status = "completed"
        history_entry.completed_at = end_time
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).update(
            {"clustering_status": "completed", "is_clustered": True}
        )
        db.commit()
        logger.info(
            f"Dataset {dataset_id} v{version}: Clustering completed successfully in {duration:.2f}s."
        )

    except Exception as e:
        logger.error(
            f"Error during clustering task for dataset {dataset_id} v{version}: {e}",
            exc_info=True,
        )
        # Update status to failed
        # Check if history_entry was successfully created before accessing
        if history_entry:
            history_entry.clustering_status = "failed"
            history_entry.error_message = str(e)[:1000]  # Truncate error
            history_entry.completed_at = datetime.utcnow()
            # No need to commit here, will be committed with DatasetMetadata update

        # Update DatasetMetadata status
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).update(
            {"clustering_status": "failed", "is_clustered": False}
        )
        db.commit()  # Commit final failure status for both history and metadata

    finally:
        db.close()  # Ensure session is closed in background task


async def process_huggingface_dataset(
    db: Session,
    dataset_id: int,
    hf_dataset_name: str,
    hf_config: Optional[str] = None,
    hf_split: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    text_field_hint: Optional[str] = None,
    label_field_hint: Optional[str] = None,
    selected_columns: Optional[List[str]] = None,
    limit_rows: Optional[int] = None,
):
    """Process a Hugging Face dataset download in the background."""
    from ..utils.huggingface_utils import (
        download_hf_dataset,
        generate_field_mappings,
        save_dataset_to_parquet,
        insert_texts_from_dataframe,
    )

    try:
        # Get dataset metadata from database
        with SessionLocal() as local_db:
            dataset = (
                local_db.query(DatasetMetadata)
                .filter(DatasetMetadata.id == dataset_id)
                .first()
            )
            if not dataset:
                logger.error(f"Dataset with ID {dataset_id} not found")
                return

            # Update status to downloading
            dataset.status = "downloading"
            local_db.commit()

            # Check for text field
            if not text_field_hint:
                raise ValueError(
                    "Text field must be explicitly specified. Auto-detection is no longer supported."
                )

            # Download dataset from HuggingFace
            df, schema_info = download_hf_dataset(
                hf_dataset_name,
                config=hf_config,
                split=hf_split,
                revision=hf_revision,
                token=hf_token,
                limit_rows=limit_rows,
                selected_columns=selected_columns,
            )

            # Generate field mappings using explicitly provided fields
            try:
                field_mappings = generate_field_mappings(
                    df, text_field_hint, label_field_hint
                )
            except ValueError as e:
                raise ValueError(f"Field mapping error: {str(e)}")

            # Save dataset to disk
            parquet_path, schema_path = save_dataset_to_parquet(
                df, hf_dataset_name, hf_config, hf_split
            )

            # Update dataset metadata
            dataset.file_path = parquet_path
            dataset.dataset_schema = json.dumps(schema_info)
            dataset.field_mappings = json.dumps(field_mappings)
            dataset.status = "downloaded"
            dataset.verification_status = "valid"
            dataset.total_rows = schema_info["num_rows"]
            dataset.hf_revision = schema_info.get("revision_used")
            local_db.commit()

            # Insert texts into TextDB using the mapped text field
            try:
                text_count = insert_texts_from_dataframe(
                    local_db, df, dataset_id, field_mappings
                )
                logger.info(
                    f"Successfully processed HuggingFace dataset {hf_dataset_name}. Inserted/associated {text_count} texts."
                )
            except ValueError as e:
                raise ValueError(f"Text insertion error: {str(e)}")

            # Final status update
            dataset.status = "completed"
            local_db.commit()

    except Exception as e:
        logger.exception(f"Error processing HuggingFace dataset: {e}")

        # Update error status in DB
        with SessionLocal() as error_db:
            dataset = (
                error_db.query(DatasetMetadata)
                .filter(DatasetMetadata.id == dataset_id)
                .first()
            )
            if dataset:
                dataset.status = "failed"
                dataset.error_message = str(e)[:1000]
                error_db.commit()


@router.get("/huggingface/info")
async def get_huggingface_dataset_info(dataset_name: str, token: Optional[str] = None):
    """Get information about a Hugging Face dataset."""
    try:
        from ..utils.huggingface_utils import get_dataset_info

        info = get_dataset_info(dataset_name, token)
        return info
    except Exception as e:
        logger.exception(f"Error fetching HuggingFace dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface/configs")
async def get_huggingface_dataset_configs(
    dataset_name: str, token: Optional[str] = None
):
    """Get available configurations for a Hugging Face dataset."""
    try:

        configs = get_dataset_configs(dataset_name, token)
        return {"configs": configs}
    except Exception as e:
        logger.exception(f"Error fetching HuggingFace dataset configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface/splits")
async def get_huggingface_dataset_splits(
    dataset_name: str, config: Optional[str] = None, token: Optional[str] = None
):
    """Get available splits for a Hugging Face dataset configuration."""
    try:
        from utils.huggingface_utils import get_dataset_splits

        splits = get_dataset_splits(dataset_name, config, token)
        return {"splits": splits}
    except Exception as e:
        logger.exception(f"Error fetching HuggingFace dataset splits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface/features", response_model=Dict)
async def get_huggingface_dataset_features(
    dataset_name: str, config: Optional[str] = None, token: Optional[str] = None
):
    """Get features (schema) for a HuggingFace dataset configuration."""
    try:
        from utils.huggingface_utils import get_dataset_features

        features = get_dataset_features(dataset_name, config, token)
        return features
    except Exception as e:
        logger.error(f"Error fetching HuggingFace dataset features: {str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found on Hugging Face.",
            )

        max_error_length = 1000  # Limit the error message length
        short_error = str(e)[:max_error_length]
        if len(str(e)) > max_error_length:
            short_error += "... (truncated)"

        raise HTTPException(
            status_code=500, detail=f"Error fetching features: {short_error}"
        )
