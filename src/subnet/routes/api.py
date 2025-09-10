from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Union, Any
from models import (
    TextDB,
    CategoryResponse,
    Category,
    Level1Cluster,
    Level1ClusterResponse,
    TextAssignment,
    TextDBResponse,
)
from utils.clustering import cluster_texts
from utils.database import get_db
from sqlalchemy.orm import Session, joinedload
import logging
import os

router = APIRouter()
logger = logging.getLogger(__name__)


def _orm_reads_disabled() -> bool:
    val = os.environ.get("IVORY_DISABLE_ORM_READS", "0").lower()
    return val in {"1", "true", "yes", "on"}


def _ensure_orm_reads_enabled():
    if _orm_reads_disabled():
        raise HTTPException(status_code=410, detail="ORM-backed read endpoints are disabled. Use /query APIs.")


@router.post(
    "/cluster",
    response_model=Dict[
        str, Union[List[CategoryResponse], List[Level1ClusterResponse]]
    ],
)
async def cluster_texts_endpoint(
    texts: List[str],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    dataset_id: int = 0,  # Use 0 as a default for standalone clustering
):
    """
    Cluster texts into hierarchical categories and level-1 clusters.
    Returns both categories and their level-1 clusters.
    """
    try:
        categories, level1_clusters_or_any = await cluster_texts(texts, db, dataset_id)

        level1_clusters = (
            level1_clusters_or_any if isinstance(level1_clusters_or_any, list) else []
        )

        return {
            "categories": [CategoryResponse.model_validate(cat) for cat in categories],
            "level1_clusters": [
                Level1ClusterResponse.model_validate(l1) for l1 in level1_clusters
            ],
        }
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories", response_model=List[CategoryResponse])
async def list_categories(db: Session = Depends(get_db)):
    """List all categories."""
    try:
        _ensure_orm_reads_enabled()
        categories = db.query(Category).all()
        return [CategoryResponse.model_validate(cat) for cat in categories]
    except Exception as e:
        logger.exception(f"Error listing categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/categories/{category_id}/level1_clusters",
    response_model=List[Level1ClusterResponse],
)
async def list_level1_clusters(category_id: int, db: Session = Depends(get_db)):
    """List all Level 1 clusters for a specific category."""
    try:
        _ensure_orm_reads_enabled()
        level1_clusters = (
            db.query(Level1Cluster)
            .filter(Level1Cluster.category_id == category_id)
            .all()
        )
        if not level1_clusters:
            category_exists = (
                db.query(Category.id).filter(Category.id == category_id).first()
            )
            if not category_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Category {category_id} not found",
                )
            logger.info(f"No Level 1 clusters found for category {category_id}")
            return []

        return [Level1ClusterResponse.model_validate(l1) for l1 in level1_clusters]
    except Exception as e:
        logger.exception(f"Error listing level 1 clusters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/level1_clusters/{level1_cluster_id}/texts", response_model=Dict[str, Any])
async def list_level1_cluster_texts(
    level1_cluster_id: int, db: Session = Depends(get_db)
):
    """List all texts assigned to a specific Level 1 cluster."""
    try:
        _ensure_orm_reads_enabled()
        logger.info(f"Fetching texts for Level 1 cluster {level1_cluster_id}")

        level1_cluster = (
            db.query(Level1Cluster)
            .filter(Level1Cluster.id == level1_cluster_id)
            .first()
        )

        if not level1_cluster:
            logger.error(f"Level 1 cluster {level1_cluster_id} not found")
            raise HTTPException(
                status_code=404, detail=f"Level 1 cluster {level1_cluster_id} not found"
            )

        logger.info(f"Found Level 1 cluster: {level1_cluster.title}")

        assignments = (
            db.query(TextAssignment, TextDB)
            .join(TextDB, TextAssignment.text_id == TextDB.id)
            .filter(TextAssignment.level1_cluster_id == level1_cluster_id)
            .order_by(TextAssignment.l1_probability.desc())
            .all()
        )

        logger.info(
            f"Retrieved {len(assignments)} text assignments for Level 1 cluster {level1_cluster_id}"
        )

        texts_data = [
            {
                "id": text.id,
                "text": text.text,
                "assignment_id": assignment.id,
                "l1_probability": assignment.l1_probability,
                "l2_probability": assignment.l2_probability,
            }
            for assignment, text in assignments
        ]

        return {
            "level1_cluster": {
                "id": level1_cluster.id,
                "l1_cluster_id": level1_cluster.l1_cluster_id,
                "title": level1_cluster.title,
                "category_id": level1_cluster.category_id,
                "version": level1_cluster.version,
            },
            "texts": texts_data,
        }
    except Exception as e:
        logger.exception(
            f"Error listing texts for Level 1 cluster {level1_cluster_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
