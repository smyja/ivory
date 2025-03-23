from typing import List, Dict, Tuple
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
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.database import get_duckdb_connection

logger = logging.getLogger(__name__)

# Constants
DISTANCE_THRESHOLD = 0.7  # Increased to create more general categories
MIN_CLUSTER_SIZE = 2  # Minimum size for a cluster


async def cluster_texts(
    texts: List[str], db: Session, dataset_id: int
) -> Tuple[List[Category], List[Subcluster]]:
    """
    Cluster texts into hierarchical categories and subclusters.
    Returns both categories and their subclusters.
    """
    try:
        # Clean up existing data for this dataset
        with get_duckdb_connection() as cleanup_conn:
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

        # Insert or update texts in the database
        with get_duckdb_connection() as insert_conn:
            # First check if dataset_id column exists
            columns = insert_conn.execute("DESCRIBE texts").fetchall()
            column_names = [col[0] for col in columns]

            if "dataset_id" not in column_names:
                insert_conn.execute("ALTER TABLE texts ADD COLUMN dataset_id INTEGER")

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

        # Get embeddings for all texts
        embeddings = await vectorize_texts(texts)

        # First level clustering for major categories (more general)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average",  # Changed to average for more balanced clusters
        )
        clusters = clustering.fit_predict(embeddings)

        # Get unique clusters and merge small clusters
        unique_clusters = np.unique(clusters)
        cluster_sizes = [sum(clusters == c) for c in unique_clusters]

        # If we have too many small clusters, try to merge them
        if (
            len([s for s in cluster_sizes if s < MIN_CLUSTER_SIZE])
            > len(cluster_sizes) / 2
        ):
            # Reduce threshold to create larger clusters
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD
                * 1.2,  # Increase threshold to merge more
                metric="cosine",
                linkage="average",
            )
            clusters = clustering.fit_predict(embeddings)
            unique_clusters = np.unique(clusters)

        total_texts = len(texts)

        # Process each cluster
        for cluster_id in unique_clusters:
            # Get texts in this cluster
            cluster_texts = [
                text for i, text in enumerate(texts) if clusters[i] == cluster_id
            ]
            cluster_embeddings = [
                emb for i, emb in enumerate(embeddings) if clusters[i] == cluster_id
            ]

            # Generate category name and create category first
            # For single-text clusters, still generate a proper category name
            category_name = await derive_overarching_category(cluster_texts)

            # Ensure category name is not too long (max 2 words)
            if category_name and len(category_name.split()) > 2:
                words = category_name.split()
                # Try to keep the most meaningful words
                if any(w.lower() in ["and", "or", "the", "a", "an"] for w in words):
                    # Remove common words first
                    words = [
                        w
                        for w in words
                        if w.lower() not in ["and", "or", "the", "a", "an"]
                    ]
                category_name = " ".join(words[:2])

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

            # For small clusters, still create a subcluster but with a more specific title
            if len(cluster_texts) < MIN_CLUSTER_SIZE:
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
                            100.0,  # 100% since it's the only subcluster
                        ],
                    )
                    subcluster_id = subcluster_conn.fetchone()[0]

                # Insert text cluster relationships
                with get_duckdb_connection() as text_cluster_conn:
                    for text in cluster_texts:
                        text_cluster_conn.execute(
                            """
                            INSERT INTO text_clusters (text_id, subcluster_id, membership_score)
                            SELECT t.id, ?, 1.0
                            FROM texts t
                            WHERE t.dataset_id = ? AND t.text = ?
                            ON CONFLICT (text_id, subcluster_id) 
                            DO UPDATE SET membership_score = 1.0
                            """,
                            [subcluster_id, dataset_id, text],
                        )
                continue

            # Perform subclustering with a tighter threshold for more specific groups
            subclustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=DISTANCE_THRESHOLD
                * 0.6,  # More granular subclusters
                metric="cosine",
                linkage="complete",  # Use complete linkage for tighter subclusters
            )
            subclusters = subclustering.fit_predict(cluster_embeddings)

            # Process each subcluster
            unique_subclusters = np.unique(subclusters)
            for subcluster_id in unique_subclusters:
                # Get texts in this subcluster
                subcluster_texts = [
                    text
                    for i, text in enumerate(cluster_texts)
                    if subclusters[i] == subcluster_id
                ]

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

                # Insert text cluster relationships
                with get_duckdb_connection() as text_cluster_conn:
                    for text in subcluster_texts:
                        text_cluster_conn.execute(
                            """
                            INSERT INTO text_clusters (text_id, subcluster_id, membership_score)
                            SELECT t.id, ?, 1.0
                            FROM texts t
                            WHERE t.dataset_id = ? AND t.text = ?
                            ON CONFLICT (text_id, subcluster_id) 
                            DO UPDATE SET membership_score = 1.0
                            """,
                            [subcluster_id, dataset_id, text],
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
