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
from utils.database import get_db, get_duckdb_connection
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
async def list_subcluster_texts(subcluster_id: int):
    """List all texts in a specific subcluster."""
    try:
        logger.info(f"Fetching texts for subcluster {subcluster_id}")

        with get_duckdb_connection() as conn:
            # First check if subcluster exists
            subcluster = conn.execute(
                "SELECT * FROM subclusters WHERE id = ?", [subcluster_id]
            ).fetchone()

            if not subcluster:
                logger.error(f"Subcluster {subcluster_id} not found")
                raise HTTPException(
                    status_code=404, detail=f"Subcluster {subcluster_id} not found"
                )

            logger.info(f"Found subcluster: {subcluster}")

            # Count how many text_clusters entries exist for this subcluster
            count = conn.execute(
                "SELECT COUNT(*) FROM text_clusters WHERE subcluster_id = ?",
                [subcluster_id],
            ).fetchone()[0]

            logger.info(
                f"Found {count} text_clusters entries for subcluster {subcluster_id}"
            )

            # Query texts with their membership scores
            texts_with_scores = conn.execute(
                """
                SELECT t.id, t.text, tc.membership_score
                FROM texts t
                JOIN text_clusters tc ON t.id = tc.text_id
                WHERE tc.subcluster_id = ?
                ORDER BY tc.membership_score DESC
            """,
                [subcluster_id],
            ).fetchall()

            logger.info(
                f"Retrieved {len(texts_with_scores)} texts for subcluster {subcluster_id}"
            )

            return {
                "subcluster": {
                    "id": subcluster[0],  # id
                    "title": subcluster[2],  # title
                    "row_count": subcluster[3],  # row_count
                    "percentage": subcluster[4],  # percentage
                },
                "texts": [
                    {"id": t[0], "text": t[1], "membership_score": t[2]}
                    for t in texts_with_scores
                ],
            }
    except Exception as e:
        logger.exception(
            f"Error listing texts for subcluster {subcluster_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
