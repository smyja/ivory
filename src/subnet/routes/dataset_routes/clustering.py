from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime
import json

from models import (
    DatasetMetadata,
    ClusteringHistory,
    ClusteringStatus,
    TextDB,
    Category,
    Level1Cluster,
    TextAssignment,
    CategoryResponse,
    Level1ClusterResponse,
    TextDBResponse,
    DatasetDetailResponse,
)
from utils.database import (
    get_db,
    save_clustering_results,
)
from utils.clustering import cluster_texts
from .utils import get_embeddings, generate_category_name, BATCH_SIZE

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/history", response_model=List[Dict])
async def get_clustering_history(
    dataset_id: Optional[int] = None,
    clustering_version: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get clustering history for a dataset or specific version."""
    try:
        query = db.query(ClusteringHistory)

        if dataset_id:
            query = query.filter(ClusteringHistory.dataset_id == dataset_id)

        if clustering_version:
            query = query.filter(
                ClusteringHistory.clustering_version == clustering_version
            )

        history_entries = query.order_by(
            ClusteringHistory.clustering_version.desc()
        ).all()

        result = []
        for entry in history_entries:
            result.append(
                {
                    "dataset_id": entry.dataset_id,
                    "version": entry.clustering_version,
                    "status": entry.clustering_status,
                    "message": entry.message,
                    "started_at": entry.started_at,
                    "completed_at": entry.completed_at,
                    "num_categories": entry.num_categories,
                    "num_clusters": entry.num_clusters,
                    "num_texts": entry.num_texts,
                }
            )

        return result
    except Exception as e:
        logger.error(f"Error getting clustering history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/versions", response_model=List[int])
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
        logger.error(f"Error getting clustering versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{dataset_id}/cluster", status_code=202)
async def trigger_clustering_endpoint(
    dataset_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Trigger clustering for a dataset."""
    try:
        # Check if dataset exists
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get the next version number
        latest_version = (
            db.query(ClusteringHistory.clustering_version)
            .filter(ClusteringHistory.dataset_id == dataset_id)
            .order_by(ClusteringHistory.clustering_version.desc())
            .first()
        )
        next_version = 1 if not latest_version else latest_version[0] + 1

        # Get all texts from the dataset
        texts_with_ids = (
            db.query(TextDB.id, TextDB.text)
            .filter(TextDB.dataset_id == dataset_id)
            .all()
        )

        if not texts_with_ids:
            raise HTTPException(status_code=400, detail="No texts found in dataset")

        # Create clustering history entry
        history_entry = ClusteringHistory(
            dataset_id=dataset_id,
            clustering_version=next_version,
            clustering_status="in_progress",
            started_at=datetime.now(),
            num_texts=len(texts_with_ids),
        )
        db.add(history_entry)
        db.commit()

        # Update dataset status
        dataset.clustering_status = "in_progress"
        db.commit()

        # Start clustering in the background
        background_tasks.add_task(
            run_clustering_task, dataset_id, next_version, texts_with_ids, db
        )

        return {
            "message": "Clustering started",
            "dataset_id": dataset_id,
            "version": next_version,
        }
    except Exception as e:
        logger.error(f"Error triggering clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/details")
async def get_dataset_details_endpoint(
    dataset_id: int, version: Optional[int] = None, db: Session = Depends(get_db)
):
    """Get detailed dataset information including categories and clusters."""
    try:
        # Check if dataset exists
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # If version not specified, use the latest
        if not version:
            latest_version = (
                db.query(ClusteringHistory.clustering_version)
                .filter(ClusteringHistory.dataset_id == dataset_id)
                .filter(ClusteringHistory.clustering_status == "completed")
                .order_by(ClusteringHistory.clustering_version.desc())
                .first()
            )
            if not latest_version:
                raise HTTPException(
                    status_code=404,
                    detail="No completed clustering found for this dataset",
                )
            version = latest_version[0]

        # Get categories
        categories = (
            db.query(Category)
            .filter(Category.dataset_id == dataset_id)
            .filter(Category.clustering_version == version)
            .all()
        )

        # Get clusters
        clusters = (
            db.query(Level1Cluster)
            .filter(Level1Cluster.dataset_id == dataset_id)
            .filter(Level1Cluster.clustering_version == version)
            .all()
        )

        # Get clustering history
        history = (
            db.query(ClusteringHistory)
            .filter(ClusteringHistory.dataset_id == dataset_id)
            .filter(ClusteringHistory.clustering_version == version)
            .first()
        )

        if not history or not categories or not clusters:
            raise HTTPException(status_code=404, detail="Clustering data not found")

        # Convert to response model
        category_responses = [CategoryResponse.model_validate(c) for c in categories]
        cluster_responses = [Level1ClusterResponse.model_validate(c) for c in clusters]

        # Return detailed response
        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            subset=dataset.subset,
            split=dataset.split,
            download_date=dataset.download_date,
            status=dataset.status,
            clustering_status=dataset.clustering_status,
            is_clustered=dataset.is_clustered,
            categories=category_responses,
            clusters=cluster_responses,
            clustering_version=version,
            history=history,
        )
    except Exception as e:
        logger.error(f"Error getting dataset details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_clustering_task(
    dataset_id: int, version: int, texts_with_ids: List[Tuple[int, str]], db: Session
):
    """Run the clustering task in the background."""
    try:
        # Extract texts for embedding
        text_ids = [t[0] for t in texts_with_ids]
        texts = [t[1] for t in texts_with_ids]

        # Process in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_embeddings = await get_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Run clustering algorithm
        categories, clusters, assignments = await cluster_texts(
            text_ids, texts, all_embeddings
        )

        # Generate category names
        for category in categories:
            # Get a sample of texts from this category
            category_text_ids = set(
                assignment["text_id"]
                for assignment in assignments
                if assignment["category_id"] == category["id"]
            )

            # Find the actual texts
            category_texts = [
                text for text_id, text in texts_with_ids if text_id in category_text_ids
            ][
                :5
            ]  # Limit to 5 samples

            # Generate a name
            if category_texts:
                category_name = await generate_category_name(category_texts)
                category["name"] = category_name

        # Save results to database
        await save_clustering_results(
            db, dataset_id, version, categories, clusters, assignments
        )

        # Update dataset status
        session = db()
        try:
            # Update clustering history
            history = (
                session.query(ClusteringHistory)
                .filter(ClusteringHistory.dataset_id == dataset_id)
                .filter(ClusteringHistory.clustering_version == version)
                .first()
            )
            if history:
                history.clustering_status = "completed"
                history.completed_at = datetime.now()
                history.num_categories = len(categories)
                history.num_clusters = len(clusters)

                # Update dataset metadata
                dataset = (
                    session.query(DatasetMetadata)
                    .filter(DatasetMetadata.id == dataset_id)
                    .first()
                )
                if dataset:
                    dataset.clustering_status = "completed"
                    dataset.is_clustered = True

                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

        logger.info(f"Clustering completed for dataset {dataset_id} version {version}")
    except Exception as e:
        logger.error(f"Error in clustering task: {str(e)}")
        # Update error status
        session = db()
        try:
            history = (
                session.query(ClusteringHistory)
                .filter(ClusteringHistory.dataset_id == dataset_id)
                .filter(ClusteringHistory.clustering_version == version)
                .first()
            )
            if history:
                history.clustering_status = "failed"
                history.message = str(e)

                # Update dataset metadata
                dataset = (
                    session.query(DatasetMetadata)
                    .filter(DatasetMetadata.id == dataset_id)
                    .first()
                )
                if dataset:
                    dataset.clustering_status = "failed"
                    dataset.message = str(e)

                session.commit()
        except Exception as inner_e:
            session.rollback()
            logger.error(f"Error updating status: {str(inner_e)}")
        finally:
            session.close()
