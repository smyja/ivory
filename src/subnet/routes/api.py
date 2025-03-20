from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict
from models import (
    TextDB,
    CategoryResponse,
    SubclusterResponse,
    Category,
    Subcluster,
    TextCluster,
)
from utils.clustering import cluster_texts
from utils.database import get_db
from sqlalchemy.orm import Session
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cluster", response_model=Dict[str, List])
async def cluster_texts_endpoint(
    texts: List[str],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    dataset_id: int = 0,  # Use 0 as a default for standalone clustering
):
    """
    Cluster texts into hierarchical categories and subclusters.
    Returns both categories and their subclusters.
    """
    try:
        categories, subclusters = await cluster_texts(texts, db, dataset_id)
        return {
            "categories": [CategoryResponse.model_validate(cat) for cat in categories],
            "subclusters": [
                SubclusterResponse.model_validate(sub) for sub in subclusters
            ],
        }
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories", response_model=List[CategoryResponse])
async def list_categories(db: Session = Depends(get_db)):
    """List all categories."""
    try:
        categories = db.query(Category).all()
        return [CategoryResponse.model_validate(cat) for cat in categories]
    except Exception as e:
        logger.exception(f"Error listing categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/categories/{category_id}/subclusters", response_model=List[SubclusterResponse]
)
async def list_subclusters(category_id: int, db: Session = Depends(get_db)):
    """List all subclusters for a specific category."""
    try:
        subclusters = (
            db.query(Subcluster).filter(Subcluster.category_id == category_id).all()
        )
        if not subclusters:
            raise HTTPException(
                status_code=404,
                detail=f"No subclusters found for category {category_id}",
            )
        return [SubclusterResponse.model_validate(sub) for sub in subclusters]
    except Exception as e:
        logger.exception(f"Error listing subclusters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subclusters/{subcluster_id}/texts")
async def list_subcluster_texts(subcluster_id: int, db: Session = Depends(get_db)):
    """List all texts in a specific subcluster."""
    try:
        subcluster = db.query(Subcluster).filter(Subcluster.id == subcluster_id).first()
        if not subcluster:
            raise HTTPException(
                status_code=404, detail=f"Subcluster {subcluster_id} not found"
            )

        # Query texts with their membership scores
        texts_with_scores = (
            db.query(TextDB, TextCluster.membership_score)
            .join(TextCluster, TextDB.id == TextCluster.text_id)
            .filter(TextCluster.subcluster_id == subcluster_id)
            .order_by(TextCluster.membership_score.desc())
            .all()
        )

        return {
            "subcluster": SubclusterResponse.model_validate(subcluster),
            "texts": [
                {
                    "id": t.id,
                    "text": t.text,
                    "membership_score": score,
                }
                for t, score in texts_with_scores
            ],
        }
    except Exception as e:
        logger.exception(f"Error listing texts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
