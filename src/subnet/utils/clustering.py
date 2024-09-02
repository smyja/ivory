from typing import List, Dict
from models import Question, ClusteredQuestion
from utils.vectorizer import vectorize_questions
from utils.titling import generate_semantic_title, derive_overarching_category
import hdbscan
import numpy as np
import logging
import asyncio

logger = logging.getLogger(__name__)

async def cluster_questions(questions: List[Question]) -> List[ClusteredQuestion]:
    try:
        question_texts = [q.question for q in questions]
        logger.info(f"Clustering {len(question_texts)} questions")

        X = vectorize_questions(question_texts)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5)
        cluster_labels = clusterer.fit_predict(X)

        unique_clusters = set(cluster_labels)
        cluster_titles: Dict[int, str] = {}
        cluster_categories: Dict[int, str] = {}

        for cluster in unique_clusters:
            if cluster != -1:
                cluster_questions = [q for q, label in zip(question_texts, cluster_labels) if label == cluster]
                cluster_titles[cluster] = await generate_semantic_title(cluster_questions)
                
                # Instead of passing just one title, collect all titles in this cluster
                all_titles_in_cluster = [cluster_titles[c] for c in cluster_labels if c == cluster]
                cluster_categories[cluster] = await derive_overarching_category(all_titles_in_cluster)
            else:
                cluster_titles[cluster] = "Miscellaneous Questions"
                cluster_categories[cluster] = "Uncategorized"

        clustered_questions = []
        for q, label in zip(questions, cluster_labels):
            clustered_questions.append(
                ClusteredQuestion(
                    question=q.question,
                    answer=q.answer,
                    cluster=int(label),
                    cluster_title=cluster_titles[label],
                    category=cluster_categories[label]  # Use the category specific to this cluster
                )
            )

        logger.info(f"Clustering completed. Found {len(unique_clusters)} clusters.")
        for cluster in unique_clusters:
            logger.info(f"Cluster {cluster}: Title - {cluster_titles[cluster]}, Category - {cluster_categories[cluster]}")

        return clustered_questions
    except Exception as e:
        logger.exception(f"An error occurred during clustering: {str(e)}")
        raise


async def process_questions(questions: List[Question]) -> List[ClusteredQuestion]:
    return await cluster_questions(questions)