from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import logging
import os
import pandas as pd
import requests
import json
from models import (
    DatasetRequest,
    DatasetMetadataResponse,
    DatasetMetadata,
    DownloadStatus,
    Category,
    Subcluster,
    TextCluster,
    TextDB,
    CategoryResponse,
    SubclusterResponse,
    ClusteringHistory,
    DatasetResponse,
)
from utils.database import (
    get_db,
    download_and_save_dataset,
    get_active_downloads,
    verify_and_update_dataset_status,
    get_duckdb_connection,
)
from openai import OpenAI


from utils.clustering import cluster_texts as clustering_utils_cluster_texts
from datetime import datetime
import duckdb
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils.scan_datasets import scan_datasets_folder
from sqlalchemy import func

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants for clustering
BATCH_SIZE = 100  # Number of texts to process at once for embeddings
DISTANCE_THRESHOLD = 0.5  # Clustering distance threshold

# Model configurations
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENROUTER_LLM_MODEL = "anthropic/claude-3-sonnet"


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


async def generate_subcluster_name(texts: List[str]) -> str:
    """Generate a subcluster name for a group of texts using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY environment variable is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = (
        "Create a specific and descriptive title (3-5 words) for the following group of texts. "
        "The title should be clear and focused on the shared theme.\n\n"
        "Texts:\n" + "\n".join(texts)  # Removed limit
    )

    response = client.chat.completions.create(
        model=OPENROUTER_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in creating specific, descriptive titles.",
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


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    """List all available datasets."""
    try:
        # First scan for any new datasets
        scan_datasets_folder()

        # Then fetch all datasets from the database
        datasets = db.query(DatasetMetadata).all()
        return [DatasetResponse.model_validate(dataset) for dataset in datasets]
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
            if dataset_metadata.status != DownloadStatus.COMPLETED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset is not ready. Current status: {dataset_metadata.status.value}",
                )

            dataset_path = os.path.join(
                "datasets", dataset_metadata.name.replace("/", "_")
            )
            if dataset_metadata.subset:
                dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

            # Find splits
            available_splits = [
                d
                for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))
            ]
            if not available_splits:
                raise HTTPException(
                    status_code=404, detail="No splits found for this dataset"
                )

            # Use the first split if none specified
            split = (
                dataset_metadata.split
                if dataset_metadata.split
                else available_splits[0]
            )
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
            if dataset_metadata.subset:
                dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

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
                "subset": dataset_metadata.subset,
                "status": dataset_metadata.status.value,
                "splits": rows_info,
            }
        else:
            # Basic information
            return {
                "id": dataset_metadata.id,
                "name": dataset_metadata.name,
                "subset": dataset_metadata.subset,
                "status": dataset_metadata.status.value,
                "download_date": dataset_metadata.download_date,
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
    dataset_id: int, status: DownloadStatus, db: Session = Depends(get_db)
):
    # Find the dataset record
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Update the status
    dataset_metadata.status = status
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
                "status": dataset_metadata.status.value,
                "download_date": dataset_metadata.download_date,
            }
    except Exception as e:
        logger.error(f"Error getting dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/cluster")
async def process_clustering(dataset_id: int, db: Session = Depends(get_db)):
    """Process clustering for a dataset."""
    try:
        # Get the latest version number for this dataset
        latest_version = (
            db.query(func.max(ClusteringHistory.clustering_version))
            .filter(ClusteringHistory.dataset_id == dataset_id)
            .scalar()
            or 0
        )
        new_version = latest_version + 1

        # Create clustering history entry
        history = ClusteringHistory(
            dataset_id=dataset_id,
            clustering_status="in_progress",
            titling_status="not_started",
            created_at=datetime.utcnow(),
            clustering_version=new_version,
        )
        db.add(history)
        db.commit()

        # Get dataset
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Update dataset status
        dataset.clustering_status = "in_progress"
        db.commit()

        try:
            # Get texts from the dataset
            with get_duckdb_connection() as conn:
                result = conn.execute(
                    """
                    SELECT text
                    FROM texts
                    WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchall()
                texts = [row[0] for row in result]

            if not texts:
                raise HTTPException(status_code=400, detail="No texts found in dataset")

            # Perform clustering
            categories, subclusters = await clustering_utils_cluster_texts(
                texts, db, dataset_id, new_version
            )

            # Update history
            history.clustering_status = "completed"
            history.titling_status = "completed"
            history.completed_at = datetime.utcnow()
            db.commit()

            # Update dataset status
            dataset.clustering_status = "completed"
            dataset.is_clustered = True
            db.commit()

            return {
                "message": "Clustering completed successfully",
                "categories": len(categories),
                "subclusters": len(subclusters),
                "version": new_version,
            }

        except Exception as e:
            # Update history with error
            history.clustering_status = "failed"
            history.titling_status = "failed"
            history.error_message = str(e)
            history.completed_at = datetime.utcnow()
            db.commit()

            # Update dataset status
            dataset.clustering_status = "failed"
            db.commit()

            logger.error(f"Error during clustering: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/clustering_status")
async def get_clustering_status(dataset_id: int, db: Session = Depends(get_db)):
    """Get the current clustering status for a dataset."""
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset_metadata.clustering_status == "failed":
        raise HTTPException(
            status_code=500,
            detail="Clustering failed. Please try again.",
        )

    return {"status": dataset_metadata.clustering_status}


@router.get("/{dataset_id}/clusters")
async def get_clusters(
    dataset_id: int, version: Optional[int] = None, db: Session = Depends(get_db)
):
    """Get clusters for a dataset."""
    try:
        # If version is not specified, get the latest version
        if version is None:
            latest_version = (
                db.query(func.max(ClusteringHistory.clustering_version))
                .filter(
                    ClusteringHistory.dataset_id == dataset_id,
                    ClusteringHistory.clustering_status == "completed",
                )
                .scalar()
            )

            if latest_version is None:
                return {"status": "not_started", "categories": []}

            version = latest_version

        # Check if clustering is in progress
        history = (
            db.query(ClusteringHistory)
            .filter(
                ClusteringHistory.dataset_id == dataset_id,
                ClusteringHistory.clustering_version == version,
            )
            .first()
        )

        if not history:
            return {"status": "not_started", "categories": []}

        if history.clustering_status == "in_progress":
            return {"status": "in_progress", "categories": []}

        if history.clustering_status == "failed":
            # Return a proper error response with 500 status code
            error_message = (
                history.error_message or "Clustering failed for this version"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Clustering failed for version {version}: {error_message}",
            )

        # Get categories and subclusters for this version
        categories = (
            db.query(Category)
            .filter(Category.dataset_id == dataset_id, Category.version == version)
            .all()
        )

        if not categories:
            # Check if there are any categories for this dataset at all
            any_categories = (
                db.query(Category).filter(Category.dataset_id == dataset_id).first()
            )

            if any_categories:
                # There are categories for this dataset, but not for this version
                available_versions = (
                    db.query(Category.version)
                    .filter(Category.dataset_id == dataset_id)
                    .distinct()
                    .all()
                )
                available_versions = [v[0] for v in available_versions]
                return {
                    "status": "completed",
                    "categories": [],
                    "message": f"No categories found for version {version}. Available versions: {available_versions}",
                }
            else:
                # No categories at all for this dataset
                return {
                    "status": "completed",
                    "categories": [],
                    "message": "No categories found for this dataset. Try running clustering first.",
                }

        # Get subclusters for each category
        result = []
        for category in categories:
            subclusters = (
                db.query(Subcluster)
                .filter(
                    Subcluster.category_id == category.id, Subcluster.version == version
                )
                .all()
            )

            # Get texts for each subcluster
            subcluster_data = []
            for subcluster in subclusters:
                texts = (
                    db.query(TextDB)
                    .join(TextCluster, TextDB.id == TextCluster.text_id)
                    .filter(TextCluster.subcluster_id == subcluster.id)
                    .all()
                )

                subcluster_data.append(
                    {
                        "id": subcluster.id,
                        "title": subcluster.title,
                        "row_count": subcluster.row_count,
                        "percentage": subcluster.percentage,
                        "texts": [{"id": text.id, "text": text.text} for text in texts],
                    }
                )

            result.append(
                {
                    "id": category.id,
                    "name": category.name,
                    "total_rows": category.total_rows,
                    "percentage": category.percentage,
                    "subclusters": subcluster_data,
                }
            )

        return {"status": "completed", "categories": result, "version": version}

    except Exception as e:
        logger.exception(f"Error getting clusters: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving clusters: {str(e)}"
        )


@router.get("/clustering/history")
async def get_clustering_history(db: Session = Depends(get_db)):
    """Get the clustering history for all datasets."""
    try:
        # Check if clustering_version column exists
        try:
            # Try to query with clustering_version
            history = (
                db.query(ClusteringHistory)
                .order_by(ClusteringHistory.created_at.desc())
                .all()
            )

            return [
                {
                    "id": entry.id,
                    "dataset_id": entry.dataset_id,
                    "dataset_name": entry.dataset.name if entry.dataset else "Unknown",
                    "clustering_status": entry.clustering_status,
                    "titling_status": entry.titling_status,
                    "created_at": entry.created_at.isoformat(),
                    "completed_at": (
                        entry.completed_at.isoformat() if entry.completed_at else None
                    ),
                    "error_message": entry.error_message,
                    "clustering_version": entry.clustering_version,
                }
                for entry in history
            ]
        except Exception as e:
            # If there's an error, it might be because the column doesn't exist
            logger.warning(f"Error querying with clustering_version: {str(e)}")

            # Fallback to query without clustering_version
            history = (
                db.query(ClusteringHistory)
                .order_by(ClusteringHistory.created_at.desc())
                .all()
            )

            return [
                {
                    "id": entry.id,
                    "dataset_id": entry.dataset_id,
                    "dataset_name": entry.dataset.name if entry.dataset else "Unknown",
                    "clustering_status": entry.clustering_status,
                    "titling_status": entry.titling_status,
                    "created_at": entry.created_at.isoformat(),
                    "completed_at": (
                        entry.completed_at.isoformat() if entry.completed_at else None
                    ),
                    "error_message": entry.error_message,
                    "clustering_version": 1,  # Default to version 1 for backward compatibility
                }
                for entry in history
            ]
    except Exception as e:
        logger.error(f"Error getting clustering history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subclusters/{subcluster_id}/texts")
async def get_subcluster_texts(subcluster_id: int, db: Session = Depends(get_db)):
    """Get all texts for a specific subcluster."""
    try:
        with get_duckdb_connection() as duckdb_conn:
            # Get texts for this subcluster
            texts_result = duckdb_conn.execute(
                """
                SELECT tc.text_id, t.text, tc.membership_score
                FROM text_clusters tc
                JOIN texts t ON tc.text_id = t.id
                WHERE tc.subcluster_id = ?
                ORDER BY tc.membership_score DESC
                """,
                [subcluster_id],
            ).fetchall()

            texts = [
                {
                    "text_id": text_id,
                    "text": text,
                    "membership_score": score,
                }
                for text_id, text, score in texts_result
            ]

            return {"texts": texts}

    except Exception as e:
        logger.exception(f"Error getting subcluster texts: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving subcluster texts: {str(e)}"
        )


@router.get("/{dataset_id}/clustering/versions")
async def get_clustering_versions(dataset_id: int, db: Session = Depends(get_db)):
    """Get all clustering versions for a dataset."""
    try:
        # Check if clustering_version column exists
        try:
            # Try to query with clustering_version
            versions = (
                db.query(ClusteringHistory)
                .filter(ClusteringHistory.dataset_id == dataset_id)
                .order_by(ClusteringHistory.created_at.desc())
                .all()
            )

            return [
                {
                    "id": version.id,
                    "version": version.clustering_version,
                    "status": version.clustering_status,
                    "created_at": version.created_at.isoformat(),
                    "completed_at": (
                        version.completed_at.isoformat()
                        if version.completed_at
                        else None
                    ),
                }
                for version in versions
            ]
        except Exception as e:
            # If there's an error, it might be because the column doesn't exist
            logger.warning(f"Error querying with clustering_version: {str(e)}")

            # Fallback to query without clustering_version
            versions = (
                db.query(ClusteringHistory)
                .filter(ClusteringHistory.dataset_id == dataset_id)
                .order_by(ClusteringHistory.created_at.desc())
                .all()
            )

            return [
                {
                    "id": version.id,
                    "version": 1,  # Default to version 1 for backward compatibility
                    "status": version.clustering_status,
                    "created_at": version.created_at.isoformat(),
                    "completed_at": (
                        version.completed_at.isoformat()
                        if version.completed_at
                        else None
                    ),
                }
                for version in versions
            ]
    except Exception as e:
        logger.error(f"Error getting clustering versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
