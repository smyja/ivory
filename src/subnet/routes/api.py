from fastapi import APIRouter, HTTPException
from models import QuestionList, ClusteredQuestion
from utils.clustering import process_questions
from typing import List
import logging


router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/cluster", response_model=List[ClusteredQuestion])
async def cluster_questions(question_list: QuestionList):
    try:
        results = await process_questions(question_list.questions)
        logger.info(f"Clustering completed for all {len(results)} questions")
        return results
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))