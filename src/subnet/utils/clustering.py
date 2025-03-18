from typing import List, Dict, Tuple
from models import (
    Question,
    ClusteredQuestion,
    Category,
    Subcluster,
    QuestionCluster,
    QuestionDB,
)
from utils.vectorizer import vectorize_questions
from utils.titling import generate_semantic_title, derive_overarching_category
import hdbscan
import numpy as np
import logging
import asyncio
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def cluster_questions(
    questions: List[Question], db: Session
) -> Tuple[List[Category], List[Subcluster]]:
    """
    Perform hierarchical clustering on questions:
    1. First level: Main categories (broader topics)
    2. Second level: Subclusters (specific topics within categories)
    """
    try:
        # First, store all questions in the database
        db_questions = []
        for q in questions:
            db_question = QuestionDB(question=q.question, answer=q.answer)
            db.add(db_question)
            db_questions.append(db_question)
        db.flush()  # Get IDs for all questions

        question_texts = [q.question for q in questions]
        logger.info(f"Clustering {len(question_texts)} questions")

        # Special case: if there's only 1 question
        if len(questions) == 1:
            logger.info(
                "Only one question provided, creating single category and subcluster"
            )
            single_question = db_questions[0]
            title = await generate_semantic_title([single_question.question])

            # Create category
            category = Category(name=title, total_rows=1, percentage=100.0)
            db.add(category)
            db.flush()  # Get the category ID

            # Create subcluster
            subcluster = Subcluster(
                category_id=category.id, title=title, row_count=1, percentage=100.0
            )
            db.add(subcluster)
            db.flush()

            # Create question cluster membership
            question_cluster = QuestionCluster(
                question_id=single_question.id,
                subcluster_id=subcluster.id,
                membership_score=1.0,
            )
            db.add(question_cluster)

            return [category], [subcluster]

        # Vectorize questions for clustering
        X = vectorize_questions(question_texts)

        # First level: Create main categories
        main_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(
                3, len(questions) // 10
            ),  # Adjust size based on dataset
            min_samples=2,
            cluster_selection_epsilon=0.7,
            metric="euclidean",
        )
        main_labels = main_clusterer.fit_predict(X)

        categories = []
        subclusters = []
        unique_main_clusters = set(main_labels)

        # Handle unclustered points in main clustering
        if -1 in unique_main_clusters:
            unclustered_indices = np.where(main_labels == -1)[0]
            # Assign unclustered points to nearest cluster or create a misc category
            if len(unclustered_indices) < len(questions):
                # Assign to nearest cluster
                for idx in unclustered_indices:
                    distances = main_clusterer.probabilities_[idx]
                    nearest_cluster = np.argmax(distances)
                    main_labels[idx] = nearest_cluster
            else:
                # All points are unclustered, create a misc category
                main_labels = np.zeros_like(main_labels)
                unique_main_clusters = {0}

        # Process each main category
        for category_id in unique_main_clusters:
            category_indices = np.where(main_labels == category_id)[0]
            category_questions = [db_questions[i] for i in category_indices]

            # Generate category title
            category_texts = [q.question for q in category_questions]
            category_title = await generate_semantic_title(category_texts)

            # Create category
            category = Category(
                name=category_title,
                total_rows=len(category_questions),
                percentage=len(category_questions) / len(questions) * 100,
            )
            db.add(category)
            db.flush()  # Get the category ID
            categories.append(category)

            # Second level: Create subclusters within category
            sub_X = vectorize_questions(category_texts)
            sub_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, len(category_questions) // 5),
                min_samples=1,
                cluster_selection_epsilon=0.5,
                metric="euclidean",
            )
            sub_labels = sub_clusterer.fit_predict(sub_X)

            # Handle unclustered points in subclustering
            if -1 in set(sub_labels):
                unclustered_indices = np.where(sub_labels == -1)[0]
                if len(unclustered_indices) < len(category_questions):
                    # Assign to nearest subcluster
                    for idx in unclustered_indices:
                        distances = sub_clusterer.probabilities_[idx]
                        nearest_cluster = np.argmax(distances)
                        sub_labels[idx] = nearest_cluster
                else:
                    # All points are unclustered, create a single subcluster
                    sub_labels = np.zeros_like(sub_labels)

            # Process each subcluster
            for sub_id in set(sub_labels):
                sub_indices = np.where(sub_labels == sub_id)[0]
                sub_questions = [category_questions[i] for i in sub_indices]

                # Generate subcluster title
                sub_texts = [q.question for q in sub_questions]
                subcluster_title = await generate_semantic_title(sub_texts)

                # Create subcluster
                subcluster = Subcluster(
                    category_id=category.id,
                    title=subcluster_title,
                    row_count=len(sub_questions),
                    percentage=len(sub_questions) / len(questions) * 100,
                )
                db.add(subcluster)
                db.flush()  # Get the subcluster ID
                subclusters.append(subcluster)

                # Create question cluster memberships
                for q_idx, question in enumerate(sub_questions):
                    membership_score = sub_clusterer.probabilities_[
                        sub_indices[q_idx]
                    ].max()
                    question_cluster = QuestionCluster(
                        question_id=question.id,
                        subcluster_id=subcluster.id,
                        membership_score=float(membership_score),
                    )
                    db.add(question_cluster)

        db.commit()
        logger.info(
            f"Clustering completed. Found {len(categories)} categories and {len(subclusters)} subclusters"
        )
        return categories, subclusters

    except Exception as e:
        db.rollback()
        logger.exception(f"An error occurred during clustering: {str(e)}")
        raise


async def process_questions(
    questions: List[Question], db: Session
) -> Tuple[List[Category], List[Subcluster]]:
    """Process questions and return hierarchical clustering results."""
    return await cluster_questions(questions, db)
