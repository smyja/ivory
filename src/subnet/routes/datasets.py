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
)
from utils.database import (
    get_db,
    download_and_save_dataset,
    get_active_downloads,
    verify_and_update_dataset_status,
    get_duckdb_connection,
)
from utils.clustering import cluster_texts
from datetime import datetime
import duckdb
import numpy as np
from sklearn.cluster import AgglomerativeClustering

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants for clustering
BATCH_SIZE = 100  # Number of texts to process at once for embeddings
DISTANCE_THRESHOLD = 0.5  # Clustering distance threshold
TOGETHER_EMBEDDING_MODEL = (
    "togethercomputer/m2-bert-80M-8k-retrieval"  # Model for embeddings
)
TOGETHER_LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Model for text generation


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using Together API."""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise Exception("TOGETHER_API_KEY environment variable is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.together.xyz/v1/embeddings",
            headers=headers,
            json={"model": TOGETHER_EMBEDDING_MODEL, "input": texts},
        )

        if response.status_code != 200:
            raise Exception(f"Error from Together API: {response.text}")

        result = response.json()
        return [item["embedding"] for item in result["data"]]
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise


async def generate_category_name(texts: List[str]) -> str:
    """Generate a category name for a cluster of texts using Together API."""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise Exception("TOGETHER_API_KEY environment variable is not set")

        prompt = f"""Given these texts, provide a short (1-3 words) category name that best describes them:

{texts[:5]}

Provide ONLY the category name, nothing else."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json={
                "model": TOGETHER_LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides short, descriptive category names.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 10,
                "temperature": 0.3,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Error from Together API: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error generating category name: {str(e)}")
        return "Untitled Category"


async def generate_subcluster_name(texts: List[str]) -> str:
    """Generate a subcluster name for a group of texts using Together API."""
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise Exception("TOGETHER_API_KEY environment variable is not set")

        prompt = f"""Given these texts, provide a short (3-5 words) descriptive title that captures their common theme:

{texts[:5]}

Provide ONLY the title, nothing else."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json={
                "model": TOGETHER_LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides short, descriptive titles.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 20,
                "temperature": 0.3,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Error from Together API: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error generating subcluster name: {str(e)}")
        return "Untitled Subcluster"


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


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    """List all datasets."""
    try:
        datasets = db.query(DatasetMetadata).all()
        return [
            {
                "id": dataset.id,
                "name": dataset.name,
                "subset": dataset.subset,
                "split": dataset.split,
                "status": dataset.status.value,
                "download_date": dataset.download_date,
                "clustering_status": dataset.clustering_status,
            }
            for dataset in datasets
        ]
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
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
async def process_clustering(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Process clustering for a dataset."""
    try:
        # Get dataset metadata
        dataset_metadata = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset_metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Clean up any existing clustering data for this dataset
        db.query(Category).filter(Category.dataset_id == dataset_id).delete()
        db.commit()

        # Check if dataset directory exists
        dataset_path = os.path.join("datasets", dataset_metadata.name.replace("/", "_"))
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset directory not found")

        # Check for available splits
        available_splits = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        if not available_splits:
            raise HTTPException(
                status_code=404, detail="No splits found in dataset directory"
            )

        # Use the first available split
        split_dir = os.path.join(dataset_path, available_splits[0])
        parquet_files = [f for f in os.listdir(split_dir) if f.endswith(".parquet")]
        if not parquet_files:
            raise HTTPException(
                status_code=404, detail="No Parquet files found in split directory"
            )

        # Use the first Parquet file
        parquet_path = os.path.join(split_dir, parquet_files[0])
        if not os.path.exists(parquet_path):
            raise HTTPException(status_code=404, detail="Parquet file not found")

        # Create table from Parquet file if it doesn't exist
        table_name = f"dataset_{dataset_id}"
        conn = duckdb.connect("datasets.db")
        try:
            # Check if table exists
            result = conn.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            ).fetchone()
            if not result:
                conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')"
                )
                logger.info(f"Created table {table_name} from Parquet file")
            else:
                logger.info(f"Table {table_name} already exists")
        finally:
            conn.close()

        # Get texts from the dataset
        texts = []
        conn = duckdb.connect("datasets.db")
        try:
            # Get the first text column
            columns = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").description
            text_column = next(
                (col[0] for col in columns if isinstance(col[1], str)), None
            )
            if not text_column:
                raise HTTPException(
                    status_code=400, detail="No text column found in dataset"
                )

            # Get texts from the column
            result = conn.execute(f"SELECT {text_column} FROM {table_name}").fetchall()
            texts = [row[0] for row in result if row[0] is not None]
        finally:
            conn.close()

        if not texts:
            raise HTTPException(status_code=400, detail="No texts found in dataset")

        # Update dataset status
        dataset_metadata.clustering_status = "in_progress"
        db.commit()

        # Create new clustering history entry
        history_entry = ClusteringHistory(
            dataset_id=dataset_id,
            clustering_status="in_progress",
            titling_status="not_started",
            created_at=datetime.utcnow(),
        )
        db.add(history_entry)
        db.commit()

        # Perform clustering
        categories, subclusters = await cluster_texts(texts, db, dataset_id)

        # Update dataset status and history
        dataset_metadata.clustering_status = "completed"
        dataset_metadata.is_clustered = True
        history_entry.clustering_status = "completed"
        history_entry.completed_at = datetime.utcnow()
        db.commit()

        return {
            "message": "Clustering completed successfully",
            "categories": [CategoryResponse.model_validate(cat) for cat in categories],
            "subclusters": [
                SubclusterResponse.model_validate(sub) for sub in subclusters
            ],
        }

    except Exception as e:
        logger.exception(f"Error during clustering: {str(e)}")
        if dataset_metadata:
            dataset_metadata.clustering_status = "failed"
            if history_entry:
                history_entry.clustering_status = "failed"
                history_entry.error_message = str(e)
                history_entry.completed_at = datetime.utcnow()
            db.commit()
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
    # First check if the dataset exists
    dataset_metadata = (
        db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    )
    if not dataset_metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if clustering is in progress
    if dataset_metadata.clustering_status == "in_progress":
        return {
            "status": "in_progress",
            "message": "Clustering is in progress. Please try again later.",
        }

    # Check if clustering failed
    if dataset_metadata.clustering_status == "failed":
        return {"status": "failed", "message": "Clustering failed. Please try again."}

    # Get categories
    categories = db.query(Category).filter(Category.dataset_id == dataset_id).all()

    if not categories:
        return {
            "status": "not_started",
            "message": "Clustering has not been started yet.",
        }

    return {
        "status": "completed",
        "categories": [
            {
                "id": category.id,
                "name": category.name,
                "total_rows": category.total_rows,
                "percentage": category.percentage,
                "subclusters": [
                    {
                        "id": subcluster.id,
                        "title": subcluster.title,
                        "row_count": subcluster.row_count,
                        "percentage": subcluster.percentage,
                        "texts": [
                            {
                                "id": tc.text_id,
                                "text": tc.text.text,
                                "membership_score": tc.membership_score,
                            }
                            for tc in subcluster.texts
                        ],
                    }
                    for subcluster in category.subclusters
                ],
            }
            for category in categories
        ],
    }


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


async def cluster_texts(texts: List[str], db: Session, dataset_id: int):
    """Cluster texts using OpenAI embeddings."""
    try:
        # Get embeddings for all texts
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_embeddings = await get_embeddings(batch)
            embeddings.extend(batch_embeddings)

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average",
        )
        clustering.fit(embeddings_array)

        # Create a dictionary to store unique categories
        categories_dict = {}
        subclusters_dict = {}

        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])

        # Create categories and subclusters
        for cluster_id, cluster_texts in clusters.items():
            # Generate category name using GPT
            category_name = await generate_category_name(cluster_texts)

            # Check if category name already exists
            if category_name not in categories_dict:
                category = Category(
                    name=category_name,
                    dataset_id=dataset_id,
                    total_rows=len(cluster_texts),
                    percentage=len(cluster_texts) / len(texts) * 100,
                )
                db.add(category)
                db.flush()  # Get the category ID
                categories_dict[category_name] = category
            else:
                category = categories_dict[category_name]
                # Update row count and percentage
                category.total_rows += len(cluster_texts)
                category.percentage = category.total_rows / len(texts) * 100

            # Generate subcluster name
            subcluster_name = await generate_subcluster_name(cluster_texts)

            # Create unique key for subcluster
            subcluster_key = f"{category_name}:{subcluster_name}"

            if subcluster_key not in subclusters_dict:
                subcluster = Subcluster(
                    title=subcluster_name,
                    category_id=category.id,
                    row_count=len(cluster_texts),
                    percentage=len(cluster_texts) / len(texts) * 100,
                )
                db.add(subcluster)
                subclusters_dict[subcluster_key] = subcluster
            else:
                subcluster = subclusters_dict[subcluster_key]
                # Update row count and percentage
                subcluster.row_count += len(cluster_texts)
                subcluster.percentage = subcluster.row_count / len(texts) * 100

        db.commit()
        return list(categories_dict.values()), list(subclusters_dict.values())

    except Exception as e:
        logger.exception(f"Error in cluster_texts: {str(e)}")
        raise
