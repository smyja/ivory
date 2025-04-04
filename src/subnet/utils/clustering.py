from typing import List, Dict, Tuple, Optional, ContextManager
from contextlib import contextmanager
from models import (
    TextDB,
    Category,
    Subcluster,
    TextCluster,
    DatasetMetadata,
)
from utils.vectorizer import vectorize_texts
from utils.titling import (
    generate_semantic_title,
    derive_overarching_category,
)
import hdbscan
import numpy as np
import logging
import asyncio
from sqlalchemy.orm import Session
from sklearn.cluster import AgglomerativeClustering
from utils.database import get_duckdb_connection
import gc
import os
import time

logger = logging.getLogger(__name__)

# Constants
MIN_CLUSTER_SIZE = 5  # Reduced for more granular clusters
MIN_SAMPLES = 3  # Minimum samples for HDBSCAN
CLUSTER_SELECTION_EPS = 0.05  # Epsilon for cluster selection
BATCH_SIZE = 512  # Batch size for processing
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


@contextmanager
def retry_on_lock(
    max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY
) -> ContextManager:
    """Context manager that retries operations on database lock conflicts."""
    attempt = 0
    while True:
        try:
            with get_duckdb_connection() as conn:
                yield conn
                break
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            if "Conflicting lock" in str(e):
                logger.warning(
                    f"Database locked, retrying in {delay} seconds (attempt {attempt}/{max_retries})"
                )
                time.sleep(delay)
            else:
                raise


def cleanup_dataset_data(dataset_id: int) -> None:
    """Clean up existing data for a dataset with retry logic."""
    try:
        with retry_on_lock() as cleanup_conn:
            # First delete text_clusters entries
            cleanup_conn.execute(
                """
                DELETE FROM text_clusters tc
                WHERE tc.subcluster_id IN (
                    SELECT s.id 
                    FROM subclusters s
                    JOIN categories c ON s.category_id = c.id
                    WHERE c.dataset_id = ?
                )
                """,
                [dataset_id],
            )

            # Then delete subclusters
            cleanup_conn.execute(
                """
                DELETE FROM subclusters 
                WHERE category_id IN (
                    SELECT id 
                    FROM categories 
                    WHERE dataset_id = ?
                )
                """,
                [dataset_id],
            )

            # Then delete categories
            cleanup_conn.execute(
                """
                DELETE FROM categories 
                WHERE dataset_id = ?
                """,
                [dataset_id],
            )

            # Finally clean up orphaned texts
            cleanup_conn.execute(
                """
                DELETE FROM texts
                WHERE dataset_id = ?
                AND id NOT IN (
                    SELECT DISTINCT text_id
                    FROM text_clusters
                )
                """,
                [dataset_id],
            )
    except Exception as e:
        logger.error(f"Failed to cleanup dataset data: {str(e)}")
        raise


async def cluster_texts(
    texts: List[str], db: Session, dataset_id: int
) -> Tuple[List[Category], List[Subcluster]]:
    """
    Cluster texts into hierarchical categories and subclusters using HDBSCAN.
    Returns both categories and their subclusters.
    """
    try:
        # Clean up existing data
        cleanup_dataset_data(dataset_id)

        # Insert or update texts in the database
        try:
            with retry_on_lock() as insert_conn:
                # First check if dataset_id column exists
                columns = insert_conn.execute("DESCRIBE texts").fetchall()
                column_names = [col[0] for col in columns]

                if "dataset_id" not in column_names:
                    insert_conn.execute(
                        "ALTER TABLE texts ADD COLUMN dataset_id INTEGER"
                    )

                # Insert texts with ON CONFLICT DO NOTHING to handle duplicates
                for text in texts:
                    insert_conn.execute(
                        """
                        INSERT INTO texts (text, dataset_id)
                        SELECT ?, ?
                        WHERE NOT EXISTS (
                            SELECT 1 FROM texts 
                            WHERE text = ? AND dataset_id = ?
                        )
                        """,
                        [text, dataset_id, text, dataset_id],
                    )
        except Exception as e:
            logger.error(f"Failed to insert texts: {str(e)}")
            raise

        # Get embeddings for all texts
        embeddings = await vectorize_texts(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # First level clustering for major categories using HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            cluster_selection_epsilon=CLUSTER_SELECTION_EPS,
            cluster_selection_method="leaf",
            prediction_data=True,
        )
        clusters = clusterer.fit_predict(embeddings_array)
        probabilities = clusterer.probabilities_

        # Handle noise points by assigning them to nearest clusters
        noise_mask = clusters == -1
        if np.any(noise_mask):
            noise_vectors = embeddings_array[noise_mask]
            if len(noise_vectors) > 0 and len(noise_vectors) < len(clusters):
                noise_labels = hdbscan.membership_vector(clusterer, noise_vectors)
                if noise_labels.ndim < 2:
                    noise_labels = noise_labels.reshape(-1, 1)
                clusters[noise_mask] = np.argmax(noise_labels, axis=1)
                probabilities[noise_mask] = np.max(noise_labels, axis=1)

        # Get unique clusters
        unique_clusters = np.unique(clusters)
        total_texts = len(texts)

        # Process each cluster
        for cluster_id in unique_clusters:
            # Get texts in this cluster
            cluster_mask = clusters == cluster_id
            cluster_texts = [text for i, text in enumerate(texts) if cluster_mask[i]]
            cluster_probs = probabilities[cluster_mask]

            # Generate category name
            category_name = await derive_overarching_category(cluster_texts)

            # Ensure category name is not too long (max 2 words)
            if category_name and len(category_name.split()) > 2:
                words = category_name.split()
                if any(w.lower() in ["and", "or", "the", "a", "an"] for w in words):
                    words = [
                        w
                        for w in words
                        if w.lower() not in ["and", "or", "the", "a", "an"]
                    ]
                category_name = " ".join(words[:2])

            # Create category
            with get_duckdb_connection() as category_conn:
                category_conn.execute(
                    """
                    INSERT INTO categories (dataset_id, name, total_rows, percentage)
                    VALUES (?, ?, ?, ?)
                    RETURNING id
                    """,
                    [
                        dataset_id,
                        category_name,
                        len(cluster_texts),
                        len(cluster_texts) / total_texts * 100,
                    ],
                )
                category_id = category_conn.fetchone()[0]

            # For small clusters, create a single subcluster
            if len(cluster_texts) < MIN_CLUSTER_SIZE * 2:
                subcluster_title = await generate_semantic_title(cluster_texts)
                with get_duckdb_connection() as subcluster_conn:
                    subcluster_conn.execute(
                        """
                        INSERT INTO subclusters (category_id, title, row_count, percentage)
                        VALUES (?, ?, ?, ?)
                        RETURNING id
                        """,
                        [
                            category_id,
                            subcluster_title,
                            len(cluster_texts),
                            100.0,
                        ],
                    )
                    subcluster_id = subcluster_conn.fetchone()[0]

                # Insert text cluster relationships with probabilities
                with get_duckdb_connection() as text_cluster_conn:
                    for text, prob in zip(cluster_texts, cluster_probs):
                        text_cluster_conn.execute(
                            """
                            INSERT INTO text_clusters (text_id, subcluster_id, membership_score)
                            SELECT t.id, ?, ?
                            FROM texts t
                            WHERE t.dataset_id = ? AND t.text = ?
                            ON CONFLICT (text_id, subcluster_id) 
                            DO UPDATE SET membership_score = ?
                            """,
                            [subcluster_id, float(prob), dataset_id, text, float(prob)],
                        )
                continue

            # For larger clusters, perform subclustering
            cluster_embeddings = embeddings_array[cluster_mask]
            subclusterer = hdbscan.HDBSCAN(
                min_cluster_size=MIN_SAMPLES,
                min_samples=MIN_SAMPLES - 1,
                cluster_selection_epsilon=CLUSTER_SELECTION_EPS,
                cluster_selection_method="leaf",
                prediction_data=True,
            )
            subclusters = subclusterer.fit_predict(cluster_embeddings)
            subcluster_probs = subclusterer.probabilities_

            # Handle noise points in subclusters
            subcluster_noise_mask = subclusters == -1
            if np.any(subcluster_noise_mask):
                subcluster_noise_vectors = cluster_embeddings[subcluster_noise_mask]
                if len(subcluster_noise_vectors) > 0 and len(
                    subcluster_noise_vectors
                ) < len(subclusters):
                    subcluster_noise_labels = hdbscan.membership_vector(
                        subclusterer, subcluster_noise_vectors
                    )
                    if subcluster_noise_labels.ndim < 2:
                        subcluster_noise_labels = subcluster_noise_labels.reshape(-1, 1)
                    subclusters[subcluster_noise_mask] = np.argmax(
                        subcluster_noise_labels, axis=1
                    )
                    subcluster_probs[subcluster_noise_mask] = np.max(
                        subcluster_noise_labels, axis=1
                    )

            # Process each subcluster
            unique_subclusters = np.unique(subclusters)
            for subcluster_id in unique_subclusters:
                subcluster_mask = subclusters == subcluster_id
                subcluster_texts = [
                    text for i, text in enumerate(cluster_texts) if subcluster_mask[i]
                ]
                subcluster_probs_filtered = subcluster_probs[subcluster_mask]

                # Generate subcluster title
                subcluster_title = await generate_semantic_title(subcluster_texts)

                # Create subcluster
                with get_duckdb_connection() as subcluster_conn:
                    subcluster_conn.execute(
                        """
                        INSERT INTO subclusters (category_id, title, row_count, percentage)
                        VALUES (?, ?, ?, ?)
                        RETURNING id
                        """,
                        [
                            category_id,
                            subcluster_title,
                            len(subcluster_texts),
                            len(subcluster_texts) / len(cluster_texts) * 100,
                        ],
                    )
                    subcluster_id = subcluster_conn.fetchone()[0]

                # Insert text cluster relationships with probabilities
                with get_duckdb_connection() as text_cluster_conn:
                    for text, prob in zip(subcluster_texts, subcluster_probs_filtered):
                        text_cluster_conn.execute(
                            """
                            INSERT INTO text_clusters (text_id, subcluster_id, membership_score)
                            SELECT t.id, ?, ?
                            FROM texts t
                            WHERE t.dataset_id = ? AND t.text = ?
                            ON CONFLICT (text_id, subcluster_id) 
                            DO UPDATE SET membership_score = ?
                            """,
                            [subcluster_id, float(prob), dataset_id, text, float(prob)],
                        )

        # Update dataset clustering status
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if dataset:
            dataset.is_clustered = True
            db.commit()

        # Get final results
        categories = db.query(Category).filter(Category.dataset_id == dataset_id).all()
        subclusters = (
            db.query(Subcluster)
            .filter(Subcluster.category_id.in_([c.id for c in categories]))
            .all()
        )

        return categories, subclusters

    except Exception as e:
        logger.exception(f"Error during clustering: {str(e)}")
        raise
