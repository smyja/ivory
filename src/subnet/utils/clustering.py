from models import Question, ClusteredQuestion
from utils.titling import generate_title_together
from utils.vectorizer import vectorize_questions
import hdbscan
import numpy as np
from typing import List
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

async def process_questions(questions: List[Question], batch_size: int = 1000) -> List[ClusteredQuestion]:
    batches = [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, batches))
    
    return [item for batch_result in results for item in batch_result]

def process_batch(questions: List[Question]) -> List[ClusteredQuestion]:
    try:
        question_texts = [q.question for q in questions]
        
        logger.info(f"Processing batch of {len(question_texts)} questions")

        X = vectorize_questions(question_texts)
        
        logger.info(f"Vectorized questions. Shape: {X.shape}")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5)
        cluster_labels = clusterer.fit_predict(X)
        
        logger.info(f"Clustering completed. Unique labels: {set(cluster_labels)}")

        cluster_titles = generate_cluster_titles(question_texts, cluster_labels)
        
        logger.info(f"Generated cluster titles: {cluster_titles}")

        results = [
            ClusteredQuestion(
                question=q.question,
                answer=q.answer,
                cluster=int(label),
                cluster_title=cluster_titles.get(label, "Unclustered")
            )
            for q, label in zip(questions, cluster_labels)
        ]
        
        logger.info(f"Successfully prepared results for {len(results)} questions")
        
        return results
    except Exception as e:
        logger.exception(f"An error occurred in batch processing: {str(e)}")
        return []

def generate_cluster_titles(questions: List[str], labels: np.ndarray) -> dict:
    cluster_titles = {}
    for label in set(labels):
        if label != -1:
            cluster_docs = np.array(questions)[labels == label]
            cluster_titles[label] = generate_title_together(cluster_docs.tolist())
        else:
            cluster_titles[label] = "Unclustered"
    return cluster_titles