from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict
from models import (
    QuestionList,
    CategoryResponse,
    SubclusterResponse,
    Category,
    Subcluster,
    QuestionDB,
    QuestionCluster,
)
from utils.clustering import process_questions
from utils.database import get_db
from sqlalchemy.orm import Session
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cluster", response_model=Dict[str, List])
async def cluster_questions(
    question_list: QuestionList,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Cluster questions into hierarchical categories and subclusters.
    Returns both categories and their subclusters.
    """
    try:
        categories, subclusters = await process_questions(question_list.questions, db)
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


@router.get("/subclusters/{subcluster_id}/questions")
async def list_subcluster_questions(subcluster_id: int, db: Session = Depends(get_db)):
    """List all questions in a specific subcluster."""
    try:
        subcluster = db.query(Subcluster).filter(Subcluster.id == subcluster_id).first()
        if not subcluster:
            raise HTTPException(
                status_code=404, detail=f"Subcluster {subcluster_id} not found"
            )

        # Query questions with their membership scores
        questions_with_scores = (
            db.query(QuestionDB, QuestionCluster.membership_score)
            .join(QuestionCluster, QuestionDB.id == QuestionCluster.question_id)
            .filter(QuestionCluster.subcluster_id == subcluster_id)
            .order_by(QuestionCluster.membership_score.desc())
            .all()
        )

        return {
            "subcluster": SubclusterResponse.model_validate(subcluster),
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "answer": q.answer,
                    "membership_score": score,
                }
                for q, score in questions_with_scores
            ],
        }
    except Exception as e:
        logger.exception(f"Error listing questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
