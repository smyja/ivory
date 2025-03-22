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

# Import as different name to avoid conflicts
from utils.clustering import cluster_texts as clustering_utils_cluster_texts
from datetime import datetime
import duckdb
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils.scan_datasets import scan_datasets_folder

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
async def get_dataset_info(dataset_id: int, db: Session = Depends(get_db)):
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset with id {dataset_id} not found in database",
        )

    logger.info(
        f"Found dataset metadata for id {dataset_id}: {dataset_metadata.__dict__}"
    )

    # Improved path construction
    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    if dataset_metadata.subset:
        dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

    logger.info(f"Constructed dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404, detail=f"Dataset directory not found: {dataset_path}"
        )

    # More flexible split handling
    if dataset_metadata.split:
        split_path = os.path.join(dataset_path, dataset_metadata.split)
        if os.path.exists(split_path):
            splits = [dataset_metadata.split]
        else:
            splits = [
                split
                for split in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, split))
            ]
    else:
        splits = [
            split
            for split in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, split))
        ]

    logger.info(f"Found splits: {splits}")

    if not splits:
        raise HTTPException(
            status_code=404,
            detail=f"No splits found in dataset directory: {dataset_path}",
        )

    rows_info = {}
    with get_duckdb_connection() as conn:
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            logger.info(f"Processing split: {split_path}")
            parquet_files = [
                f for f in os.listdir(split_path) if f.endswith(".parquet")
            ]

            if not parquet_files:
                logger.warning(
                    f"No Parquet files found in split directory: {split_path}"
                )
                continue

            data_file = os.path.join(split_path, parquet_files[0])
            logger.info(f"Processing Parquet file: {data_file}")

            try:
                conn.execute(
                    f"CREATE OR REPLACE TABLE temp AS SELECT * FROM parquet_scan('{data_file}')"
                )
                columns = conn.execute("SELECT * FROM temp LIMIT 0").description
                row_count = conn.execute("SELECT COUNT(*) FROM temp").fetchone()[0]
                # Get all rows instead of just 5
                sample_rows = (
                    conn.execute("SELECT * FROM temp")
                    .fetchdf()
                    .to_dict(orient="records")
                )
                rows_info[split] = {
                    "columns": [col[0] for col in columns],
                    "row_count": row_count,
                    "sample_rows": sample_rows,
                }
            except Exception as e:
                logger.error(f"Error processing Parquet file {data_file}: {str(e)}")
                rows_info[split] = {"error": str(e)}

    if not rows_info:
        raise HTTPException(
            status_code=404, detail="No valid Parquet files found in any split"
        )

    return {
        "id": dataset_metadata.id,
        "name": dataset_metadata.name,
        "subset": dataset_metadata.subset,
        "status": dataset_metadata.status,
        "splits": rows_info,
    }


@router.get("/{dataset_id}/records")
async def get_dataset_records(
    dataset_id: int,
    split: Optional[str] = Query(None, description="Dataset split to query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    if dataset_metadata.subset:
        dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

    # If split is not provided, find the first available split
    if split is None:
        available_splits = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        if not available_splits:
            raise HTTPException(
                status_code=404, detail="No splits found for this dataset"
            )
        split = available_splits[0]

    split_path = os.path.join(dataset_path, split)
    if not os.path.exists(split_path):
        raise HTTPException(
            status_code=404, detail=f"Split directory not found: {split_path}"
        )

    # Find the parquet file in the split directory
    parquet_files = [f for f in os.listdir(split_path) if f.endswith(".parquet")]
    if not parquet_files:
        raise HTTPException(
            status_code=404,
            detail=f"No parquet files found in split directory: {split_path}",
        )

    data_file = os.path.join(split_path, parquet_files[0])
    if not os.path.exists(data_file):
        raise HTTPException(
            status_code=404, detail=f"Dataset file not found: {data_file}"
        )

    try:
        with get_duckdb_connection() as conn:
            # Get total count
            total_records = conn.execute(
                f"SELECT COUNT(*) FROM parquet_scan('{data_file}')"
            ).fetchone()[0]

            # Calculate offset
            offset = (page - 1) * page_size

            # Fetch paginated records
            records = conn.execute(
                f"SELECT * FROM parquet_scan('{data_file}') LIMIT {page_size} OFFSET {offset}"
            ).fetchdf()

            return {
                "split": split,
                "total_records": total_records,
                "page": page,
                "page_size": page_size,
                "records": records.to_dict(orient="records"),
            }
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")


@router.get("/{dataset_id}/query")
async def query_dataset(
    dataset_id: int,
    query: str = Query(..., description="SQL-like query string"),
    split: Optional[str] = Query(None, description="Dataset split to query"),
    db: Session = Depends(get_db),
):
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
    if dataset_metadata.subset:
        dataset_path = os.path.join(dataset_path, dataset_metadata.subset)

    # If split is not provided, find the first available split
    if split is None:
        available_splits = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        if not available_splits:
            raise HTTPException(
                status_code=404, detail="No splits found for this dataset"
            )
        split = available_splits[0]

    split_path = os.path.join(dataset_path, split)
    if not os.path.exists(split_path):
        raise HTTPException(
            status_code=404, detail=f"Split directory not found: {split_path}"
        )

    # Find the parquet file in the split directory
    parquet_files = [f for f in os.listdir(split_path) if f.endswith(".parquet")]
    if not parquet_files:
        raise HTTPException(
            status_code=404,
            detail=f"No parquet files found in split directory: {split_path}",
        )

    data_file = os.path.join(split_path, parquet_files[0])
    if not os.path.exists(data_file):
        raise HTTPException(
            status_code=404, detail=f"Dataset file not found: {data_file}"
        )

    try:
        with get_duckdb_connection() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS dataset_{dataset_id} AS SELECT * FROM parquet_scan('{data_file}')"
            )
            result_df = conn.execute(query).fetchdf()

        return {
            "split": split,
            "total_records": len(result_df),
            "records": result_df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error querying dataset: {str(e)}")


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
def get_dataset_status(dataset_id: int, db: Session = Depends(get_db)):
    """Get the status of a dataset download."""
    try:
        dataset_metadata = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset_metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {
            "id": dataset_metadata.id,
            "dataset": dataset_metadata.name,
            "status": dataset_metadata.status.value,
            "download_date": dataset_metadata.download_date,
        }
    except Exception as e:
        logger.error(f"Error getting dataset status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/data")
def get_dataset_data(dataset_id: int, db: Session = Depends(get_db)):
    """Get the data from a downloaded dataset."""
    try:
        dataset_metadata = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset_metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if dataset_metadata.status != DownloadStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset is not ready. Current status: {dataset_metadata.status.value}",
            )

        dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
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
            dataset_metadata.split if dataset_metadata.split else available_splits[0]
        )
        split_path = os.path.join(dataset_path, split)

        # Find parquet files
        parquet_files = [f for f in os.listdir(split_path) if f.endswith(".parquet")]
        if not parquet_files:
            raise HTTPException(
                status_code=404, detail=f"No parquet files found in split: {split}"
            )

        parquet_file = os.path.join(split_path, parquet_files[0])
        if not os.path.exists(parquet_file):
            raise HTTPException(status_code=404, detail="Dataset file not found")

        # Use the new connection manager for DuckDB
        with get_duckdb_connection() as conn:
            df = conn.execute(f"SELECT * FROM '{parquet_file}'").df()
            return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error getting dataset data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/cluster")
async def process_clustering(dataset_id: int, db: Session = Depends(get_db)):
    """Process clustering for a dataset."""
    try:
        # Create clustering history entry
        history = ClusteringHistory(
            dataset_id=dataset_id,
            clustering_status="in_progress",
            titling_status="not_started",
            created_at=datetime.utcnow(),
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
                texts, db, dataset_id
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
async def get_clusters(dataset_id: int, db: Session = Depends(get_db)):
    """Get the clustering results for a dataset."""
    try:
        # Check if clustering is in progress
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"Dataset with id {dataset_id} not found"
            )

        # If clustering hasn't been done yet, return not_started status
        if not dataset.is_clustered:
            return {"status": "not_started"}

        # Query for categories
        categories = []
        total_texts = 0

        try:
            with get_duckdb_connection() as duckdb_conn:
                # Get total texts count
                total_texts = duckdb_conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM texts
                    WHERE dataset_id = ?
                    """,
                    [dataset_id],
                ).fetchone()[0]

                # Get categories
                categories_result = duckdb_conn.execute(
                    """
                    SELECT c.id, c.name, c.total_rows, c.percentage
                    FROM categories c
                    WHERE c.dataset_id = ?
                    ORDER BY c.total_rows DESC
                    """,
                    [dataset_id],
                ).fetchall()

                for cat_id, name, total_rows, percentage in categories_result:
                    # Get subclusters for this category
                    subclusters_result = duckdb_conn.execute(
                        """
                        SELECT s.id, s.title, s.row_count, s.percentage
                        FROM subclusters s
                        WHERE s.category_id = ?
                        ORDER BY s.row_count DESC
                        """,
                        [cat_id],
                    ).fetchall()

                    subclusters = []
                    for sub_id, title, row_count, sub_percentage in subclusters_result:
                        # Get texts for this subcluster
                        texts_result = duckdb_conn.execute(
                            """
                            SELECT tc.text_id, t.text, tc.membership_score
                            FROM text_clusters tc
                            JOIN texts t ON tc.text_id = t.id
                            WHERE tc.subcluster_id = ?
                            ORDER BY tc.membership_score DESC
                            """,
                            [sub_id],
                        ).fetchall()

                        texts = [
                            {
                                "text_id": text_id,
                                "text": text,
                                "membership_score": score,
                            }
                            for text_id, text, score in texts_result
                        ]

                        subclusters.append(
                            {
                                "id": sub_id,
                                "title": title,
                                "row_count": row_count,
                                "percentage": sub_percentage,
                                "texts": texts,
                            }
                        )

                    categories.append(
                        {
                            "id": cat_id,
                            "name": name,
                            "total_rows": total_rows,
                            "percentage": percentage,
                            "subclusters": subclusters,
                        }
                    )

        except Exception as e:
            logger.exception(f"Error executing DuckDB queries: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error while fetching clusters: {str(e)}",
            )

        return {
            "status": "completed",
            "total_texts": total_texts,
            "categories": categories,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error getting clusters: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving clusters: {str(e)}"
        )


@router.get("/clustering/history")
async def get_clustering_history(db: Session = Depends(get_db)):
    """Get the clustering history for all datasets."""
    try:
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
