from typing import List, Dict, Tuple
from models import (
    TextDB,
    Category,
    Subcluster,
    TextCluster,
)
from utils.vectorizer import vectorize_texts
from utils.titling import generate_semantic_title, derive_overarching_category
import hdbscan
import numpy as np
import logging
import asyncio
from sqlalchemy.orm import Session
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


async def cluster_texts(
    texts: List[str], db: Session, dataset_id: int
) -> Tuple[List[Category], List[Subcluster]]:
    """
    Cluster texts into hierarchical categories and subclusters.
    Returns both categories and their subclusters.
    """
    try:
        # Convert texts to TextDB objects and add to database
        text_objects = [TextDB(text=text) for text in texts]
        db.add_all(text_objects)
        db.flush()  # Get IDs for the new texts

        # Vectorize texts for clustering
        vectors = await vectorize_texts(texts)

        # Perform clustering
        n_clusters = min(5, len(texts))  # Limit to 5 clusters or less
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)

        # Create categories and subclusters
        categories = []
        subclusters = []
        total_texts = len(texts)

        # Dictionary to store cluster title mappings
        cluster_titles = {}

        # First generate all subcluster titles
        for cluster_idx in range(n_clusters):
            # Get texts in this cluster
            cluster_texts_data = [
                text
                for text, label in zip(texts, cluster_labels)
                if label == cluster_idx
            ]

            if not cluster_texts_data:
                continue

            # Generate a semantic title for this subcluster
            subcluster_title = await generate_semantic_title(cluster_texts_data)
            cluster_titles[cluster_idx] = subcluster_title
            logger.info(
                f"Generated title for cluster {cluster_idx}: {subcluster_title}"
            )

        # Generate overarching categories for clusters
        unique_titles = list(cluster_titles.values())
        category_name = await derive_overarching_category(unique_titles)
        logger.info(f"Generated overarching category: {category_name}")

        # Now create the database entries
        for cluster_idx in range(n_clusters):
            # Get texts in this cluster with scores
            cluster_texts_with_scores = [
                (text, label, score)
                for text, label, score in zip(
                    texts, cluster_labels, kmeans.transform(vectors).min(axis=1)
                )
                if label == cluster_idx
            ]

            if not cluster_texts_with_scores:
                continue

            # Check if we have a semantic title for this cluster
            cluster_title = cluster_titles.get(
                cluster_idx, f"Subcluster {cluster_idx + 1}"
            )

            # Create category
            category = Category(
                dataset_id=dataset_id,
                name=category_name,
                total_rows=len(cluster_texts_with_scores),
                percentage=len(cluster_texts_with_scores) / total_texts * 100,
            )
            db.add(category)
            db.flush()
            categories.append(category)

            # Create subcluster
            subcluster = Subcluster(
                category_id=category.id,
                title=cluster_title,
                row_count=len(cluster_texts_with_scores),
                percentage=len(cluster_texts_with_scores) / total_texts * 100,
            )
            db.add(subcluster)
            db.flush()
            subclusters.append(subcluster)

            # Add text memberships
            for text, _, score in cluster_texts_with_scores:
                text_obj = next(t for t in text_objects if t.text == text)
                text_cluster = TextCluster(
                    text_id=text_obj.id,
                    subcluster_id=subcluster.id,
                    membership_score=float(score),
                )
                db.add(text_cluster)

        db.commit()
        logger.info(
            f"Successfully created {len(categories)} categories and {len(subclusters)} subclusters"
        )
        return categories, subclusters

    except Exception as e:
        db.rollback()
        logger.exception(f"Error during clustering: {str(e)}")
        raise
