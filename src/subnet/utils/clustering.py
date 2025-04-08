from typing import List, Dict, Tuple, Optional, ContextManager, Any
from contextlib import contextmanager
from models import (
    Base,
    DatasetMetadata,
    TextDB,
    Category,
    Level1Cluster,
    TextAssignment,
    ClusteringHistory,
    Subcluster,
    TextCluster,
)
from utils.vectorizer import vectorize_texts
from utils.titling import (
    generate_semantic_title,
    derive_overarching_category,
)
from sqlalchemy.orm import Session, joinedload
import hdbscan
import numpy as np
import logging
import asyncio
from utils.database import get_duckdb_connection
import gc
import os
import time
from sklearn.preprocessing import normalize
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# Constants - Adjusted parameters for better clustering quality
MIN_CLUSTER_SIZE = 5  # Keep min size for level 1
MIN_SAMPLES = 4  # Keep min samples for level 1
CLUSTER_SELECTION_EPS = 0.05  # Keep epsilon as is for now
BATCH_SIZE = 512
MAX_RETRIES = 3
RETRY_DELAY = 1
MIN_NOISE_PROBABILITY_THRESHOLD = 0.3  # Reverted back to 0.3 to include more points
MIN_SUBCLUSTER_SIZE = 3  # Keep constant for minimum subcluster size
MIN_SUBCLUSTER_SAMPLES = 2  # Keep constant for minimum subcluster samples
SIMILARITY_THRESHOLD = 0.7  # Keep constant for post-processing similarity check (post-processing is disabled)
MIN_CATEGORY_CLUSTER_SIZE = 5  # Increased from 3 to 5 (for Level 2 Title Clustering)
MIN_CATEGORY_SAMPLES = 4  # Increased from 2 to 4 (L2: min_cluster_size - 1)


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


async def create_single_level1_cluster(
    category_db_id: int,
    texts_in_category: List[str],
    l2_probs_for_texts: np.ndarray,
    dataset_id: int,
    version: int,
):
    """Creates a single Level1Cluster for a category (e.g., if too small or subclustering failed)."""
    fallback_title = "General"  # Simple fallback title
    level1_cluster_db_id = None
    try:
        with retry_on_lock() as conn:
            # Insert into level1_clusters
            conn.execute(
                """
                INSERT INTO level1_clusters (category_id, version, l1_cluster_id, title)
                VALUES (?, ?, ?, ?)
                RETURNING id
                """,
                [
                    category_db_id,
                    version,
                    -2,  # Use -2 to signify a fallback cluster ID
                    fallback_title,
                ],
            )
            result = conn.fetchone()
            if result:
                level1_cluster_db_id = result[0]
                logger.info(
                    f"Inserted single fallback Level1Cluster ID {level1_cluster_db_id} for category {category_db_id}, version {version}."
                )

                # Insert into text_assignments
                with retry_on_lock() as text_assign_conn:
                    for i, text in enumerate(texts_in_category):
                        # Use L2 probability as L1 probability here, L2 prob might be 0?
                        l1_prob = (
                            float(l2_probs_for_texts[i])
                            if i < len(l2_probs_for_texts)
                            else 0.0
                        )
                        l2_prob = (
                            float(l2_probs_for_texts[i])
                            if i < len(l2_probs_for_texts)
                            else 0.0
                        )

                        # First, check if this text already has an assignment
                        text_assign_conn.execute(
                            """
                            INSERT INTO text_assignments (text_id, version, level1_cluster_id, l1_probability, l2_probability)
                            SELECT t.id, ?, ?, ?, ?
                            FROM texts t
                            WHERE t.dataset_id = ? AND t.text = ?
                            AND NOT EXISTS (
                                SELECT 1 FROM text_assignments ta 
                                WHERE ta.text_id = t.id
                            )
                            """,
                            [
                                version,
                                level1_cluster_db_id,
                                l1_prob,
                                l2_prob,
                                dataset_id,
                                text,
                            ],
                        )
                    logger.info(
                        f"Linked {len(texts_in_category)} texts to fallback Level1Cluster {level1_cluster_db_id}."
                    )
            else:
                raise Exception("Fallback Level1Cluster insertion failed.")
    except Exception as e_single:
        logger.error(
            f"Failed to create single fallback Level1Cluster for category {category_db_id}: {e_single}"
        )


async def cluster_texts(
    texts: List[str], db: Session, dataset_id: int, version: int = 1
) -> Tuple[List[Category], List[Any]]:
    """
    Cluster texts into categories and subclusters.

    Args:
        texts: List of texts to cluster
        db: Database session
        dataset_id: ID of the dataset
        version: Version number for this clustering attempt

    Returns:
        Tuple of (categories, level1_clusters)
    """
    final_categories = []
    final_level1_clusters = []
    embeddings_array = None
    clusters = None
    subclusters_dict = {}

    try:
        # Insert or update texts
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
            logger.error(f"Failed to insert texts for dataset {dataset_id}: {str(e)}")
            raise  # Re-raise critical error

        # Get embeddings for all texts
        embeddings = await vectorize_texts(texts)
        if embeddings is None or len(embeddings) == 0:
            logger.warning(
                f"No embeddings generated for dataset {dataset_id}. Skipping clustering."
            )
            return [], []

        embeddings_array = np.array(embeddings, dtype=np.float32)
        del embeddings  # Free memory
        gc.collect()

        # Ensure minimum samples check uses the new MIN_SAMPLES
        if embeddings_array.shape[0] < MIN_SAMPLES:
            logger.warning(
                f"Dataset {dataset_id} has insufficient samples ({embeddings_array.shape[0]}, need {MIN_SAMPLES}) for clustering. Skipping."
            )
            return [], []

        # Normalize embeddings for effective use of Euclidean distance (approximates cosine)
        logger.info(f"Normalizing {embeddings_array.shape[0]} embeddings.")
        embeddings_array = normalize(embeddings_array, axis=1, norm="l2")
        gc.collect()

        # First level clustering for major categories using HDBSCAN
        logger.info(
            f"Starting level 1 clustering for dataset {dataset_id} with {embeddings_array.shape[0]} items (min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES})."
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,  # Use updated constant
            min_samples=MIN_SAMPLES,  # Use updated constant
            cluster_selection_epsilon=CLUSTER_SELECTION_EPS,
            metric="euclidean",  # Stays euclidean, but works like cosine due to normalization
            cluster_selection_method="leaf",
            prediction_data=True,
        )
        clusters = clusterer.fit_predict(embeddings_array)
        probabilities = clusterer.probabilities_
        gc.collect()
        logger.info(
            f"Level 1 clustering complete for dataset {dataset_id}. Found {len(np.unique(clusters[clusters != -1]))} clusters."
        )

        # Handle noise points (-1 labels) by assigning them to the nearest cluster *if confidence is high enough*
        noise_mask = clusters == -1
        num_noise = np.sum(noise_mask)
        assigned_noise_count = 0  # Keep track of assigned points
        if num_noise > 0 and num_noise < len(clusters):
            logger.info(
                f"Assigning {num_noise} noise points to nearest clusters for dataset {dataset_id} (threshold={MIN_NOISE_PROBABILITY_THRESHOLD})."
            )
            noise_vectors = embeddings_array[noise_mask]
            try:
                soft_clusters = hdbscan.membership_vector(clusterer, noise_vectors)
                if soft_clusters.ndim < 2:
                    soft_clusters = soft_clusters.reshape(-1, 1)

                if (
                    soft_clusters.shape[1] > 0
                ):  # Ensure there are actual clusters to assign to
                    noise_labels = np.argmax(soft_clusters, axis=1)
                    noise_probs = np.max(soft_clusters, axis=1)

                    # Get original indices of noise points
                    original_noise_indices = np.where(noise_mask)[0]

                    # Assign only if probability exceeds threshold
                    for i in range(num_noise):
                        if noise_probs[i] >= MIN_NOISE_PROBABILITY_THRESHOLD:
                            original_idx = original_noise_indices[i]
                            clusters[original_idx] = noise_labels[i]
                            probabilities[original_idx] = noise_probs[
                                i
                            ]  # Update probability as well
                            assigned_noise_count += 1
                    logger.info(
                        f"Successfully assigned {assigned_noise_count} / {num_noise} noise points for dataset {dataset_id}."
                    )
                else:
                    logger.warning(
                        f"membership_vector returned no cluster probabilities for noise points in dataset {dataset_id}. Skipping noise assignment."
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to assign noise points for dataset {dataset_id} using membership_vector: {e}. Skipping noise assignment."
                )
            finally:
                del noise_vectors, soft_clusters  # Clean up memory
                gc.collect()
        elif num_noise == len(clusters):
            logger.warning(
                f"All points ({num_noise}) were classified as noise in level 1 clustering for dataset {dataset_id}. Cannot proceed."
            )
            return [], []

        del noise_mask  # Clean up memory
        del clusterer  # Free HDBSCAN object memory
        gc.collect()

        # Get unique cluster IDs (excluding potential remaining -1 if noise assignment failed)
        unique_clusters = np.unique(clusters[clusters != -1])
        total_texts = len(texts)
        logger.info(
            f"Processing {len(unique_clusters)} unique L2 categories for dataset {dataset_id}."
        )

        # Process each L2 category cluster
        for l2_cluster_id_hdbscan in unique_clusters:
            cluster_mask = clusters == l2_cluster_id_hdbscan
            cluster_indices = np.where(cluster_mask)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            # Get L2 probabilities for texts in this category
            l2_probabilities_for_category = probabilities[cluster_mask]

            if not cluster_texts:  # Skip empty L2 clusters
                logger.warning(
                    f"Skipping empty L2 category cluster {l2_cluster_id_hdbscan} for dataset {dataset_id}."
                )
                continue

            # Generate category name using the semantic function
            category_name = await derive_overarching_category(cluster_texts)

            # Limit category name length (simple approach)
            # if category_name: # <-- Comment out this block
            #     category_name = " ".join(
            #         category_name.split()[:3]
            #     )  # Limit to first 3 words
            # else:
            #     category_name = f"Category {l2_cluster_id_hdbscan}"  # Fallback name

            # Use fallback name only if LLM fails
            if not category_name:
                category_name = f"Category {l2_cluster_id_hdbscan}"

            # Create category in DB
            category_db_id = None
            try:
                with retry_on_lock() as category_conn:
                    category_conn.execute(
                        """
                        INSERT INTO categories (dataset_id, name, version, l2_cluster_id)
                        VALUES (?, ?, ?, ?)
                        RETURNING id
                        """,
                        [
                            dataset_id,
                            category_name,
                            version,
                            int(l2_cluster_id_hdbscan),  # Save L2 HDBSCAN ID
                        ],
                    )
                    result = category_conn.fetchone()
                    if result:
                        category_db_id = result[0]
                        logger.info(
                            f"Inserted category ID {category_db_id} with name '{category_name}' for L2 cluster {l2_cluster_id_hdbscan}, dataset {dataset_id}, version {version}."
                        )
                    else:
                        raise Exception("Category insertion did not return an ID.")
            except Exception as e:
                logger.error(
                    f"Failed to insert category '{category_name}' for L2 cluster {l2_cluster_id_hdbscan}, dataset {dataset_id}: {e}"
                )
                continue  # Skip processing this L2 category if insertion fails

            # Level 1 Clustering logic within each L2 category
            cluster_embeddings = embeddings_array[cluster_mask]  # Already normalized

            # Determine if L1 clustering is needed
            should_l1_cluster = len(
                cluster_texts
            ) >= MIN_CLUSTER_SIZE * 2 and cluster_embeddings.shape[0] >= max(
                MIN_SAMPLES - 1, 1
            )

            if should_l1_cluster:
                logger.info(
                    f"Starting Level 1 clustering for category {category_db_id} (L2 cluster {l2_cluster_id_hdbscan}), dataset {dataset_id} with {cluster_embeddings.shape[0]} items."
                )
                try:
                    # Use the defined constants for L1 clustering parameters (previously subcluster)
                    l1_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=MIN_SUBCLUSTER_SIZE,  # Rename constants later if desired
                        min_samples=MIN_SUBCLUSTER_SAMPLES,  # Rename constants later if desired
                        cluster_selection_epsilon=CLUSTER_SELECTION_EPS * 0.8,
                        metric="euclidean",
                        cluster_selection_method="leaf",
                        prediction_data=True,
                    )
                    # These are the L1 cluster assignments *within* the L2 category
                    l1_clusters_local = l1_clusterer.fit_predict(cluster_embeddings)
                    # These are the L1 probabilities *within* the L2 category
                    l1_probabilities_local = l1_clusterer.probabilities_

                    # Store L1 results - maybe not needed anymore?
                    # subclusters_dict[category_db_id] = l1_clusters_local

                    gc.collect()
                    logger.info(
                        f"Level 1 clustering complete for category {category_db_id}. Found {len(np.unique(l1_clusters_local[l1_clusters_local != -1]))} L1 clusters."
                    )

                    # Check if ANY L1 clusters were found before proceeding
                    unique_l1_clusters = np.unique(
                        l1_clusters_local[l1_clusters_local != -1]
                    )
                    if len(unique_l1_clusters) == 0:
                        logger.warning(
                            f"All points ({len(l1_clusters_local)}) were noise in L1 clustering for category {category_db_id}. Creating single fallback Level1Cluster."
                        )
                        # Use L2 probabilities as fallback probabilities
                        await create_single_level1_cluster(  # Call renamed helper
                            category_db_id,
                            cluster_texts,  # Texts for the whole L2 category
                            l2_probabilities_for_category,  # L2 probabilities for these texts
                            dataset_id,
                            version,
                        )
                        del l1_clusterer  # Free memory here after fallback
                        gc.collect()
                        continue  # Skip rest of L1 cluster processing loop for this category
                    # --- End check ---

                    del l1_clusterer  # Free memory earlier if successful
                    gc.collect()

                    logger.info(
                        f"Processing {len(unique_l1_clusters)} unique L1 clusters for category {category_db_id}, dataset {dataset_id}."
                    )

                    # Process each *found* L1 cluster
                    for l1_id_local in unique_l1_clusters:
                        l1_cluster_mask_local = l1_clusters_local == l1_id_local
                        l1_cluster_indices_local = np.where(l1_cluster_mask_local)[0]
                        # Map local L1 indices back to original text list indices
                        original_indices_for_l1 = cluster_indices[
                            l1_cluster_indices_local
                        ]
                        l1_cluster_texts = [texts[i] for i in original_indices_for_l1]
                        # Get L1 probabilities corresponding to the L1 cluster texts
                        l1_probs_filtered = l1_probabilities_local[
                            l1_cluster_mask_local
                        ]
                        # Get L2 probabilities corresponding to the L1 cluster texts
                        l2_probs_filtered = l2_probabilities_for_category[
                            l1_cluster_indices_local
                        ]

                        if not l1_cluster_texts:
                            logger.warning(
                                f"Skipping empty L1 cluster {l1_id_local} for category {category_db_id}, dataset {dataset_id}."
                            )
                            continue

                        # Generate Level1Cluster title semantically
                        level1_title = await generate_semantic_title(l1_cluster_texts)
                        if level1_title:
                            level1_title = " ".join(
                                level1_title.split()[:5]
                            )  # Limit title length
                        else:
                            level1_title = f"Cluster {l1_id_local}"  # Fallback name

                        # Create Level1Cluster in DB
                        level1_cluster_db_id = None
                        try:
                            # --- Insert Level1Cluster using retry_on_lock (keep for this) ---
                            with retry_on_lock() as l1_conn:
                                # Insert into level1_clusters table
                                l1_conn.execute(
                                    """
                                    INSERT INTO level1_clusters (category_id, version, l1_cluster_id, title)
                                    VALUES (?, ?, ?, ?)
                                    RETURNING id
                                    """,
                                    [
                                        category_db_id,
                                        version,
                                        int(l1_id_local),  # L1 HDBSCAN ID
                                        level1_title,
                                    ],
                                )
                                l1_result = l1_conn.fetchone()
                                if l1_result:
                                    level1_cluster_db_id = l1_result[0]
                                    logger.info(
                                        f"Inserted Level1Cluster ID {level1_cluster_db_id} with title '{level1_title}' for category {category_db_id}, dataset {dataset_id}, version {version}."
                                    )
                                else:
                                    raise Exception(
                                        "Level1Cluster insertion did not return an ID."
                                    )

                            # --- Insert text-assignments using main SQLAlchemy session 'db' ---
                            # First, delete any existing assignments for these texts in this version
                            for text_content in l1_cluster_texts:
                                # Find the text_id by querying the TextDB table directly
                                text_query = sql_text(
                                    """
                                    SELECT id FROM texts 
                                    WHERE dataset_id = :dataset_id AND text = :text_content
                                    """
                                )
                                text_result = db.execute(
                                    text_query,
                                    {
                                        "dataset_id": dataset_id,
                                        "text_content": text_content,
                                    },
                                ).fetchone()

                                if text_result:
                                    text_id = text_result[0]
                                    # Delete existing assignment for this text_id and version
                                    delete_query = sql_text(
                                        """
                                        DELETE FROM text_assignments
                                        WHERE text_id = :text_id AND version = :version
                                        """
                                    )
                                    db.execute(
                                        delete_query,
                                        {"text_id": text_id, "version": version},
                                    )

                            # Now insert new assignments
                            for i, text_content in enumerate(l1_cluster_texts):
                                # Ensure probabilities are float
                                l1_prob = (
                                    float(l1_probs_filtered[i])
                                    if i < len(l1_probs_filtered)
                                    else 0.0
                                )
                                l2_prob = (
                                    float(l2_probs_filtered[i])
                                    if i < len(l2_probs_filtered)
                                    else 0.0
                                )
                                # Simple insert without ON CONFLICT
                                insert_sql = sql_text(
                                    """
                                    INSERT INTO text_assignments (text_id, version, level1_cluster_id, l1_probability, l2_probability)
                                    SELECT t.id, :version, :l1_cluster_db_id, :l1_prob, :l2_prob
                                    FROM texts t
                                    WHERE t.dataset_id = :dataset_id AND t.text = :text_content
                                    """
                                )
                                db.execute(
                                    insert_sql,
                                    {
                                        "version": version,
                                        "l1_cluster_db_id": level1_cluster_db_id,
                                        "l1_prob": l1_prob,
                                        "l2_prob": l2_prob,
                                        "dataset_id": dataset_id,
                                        "text_content": text_content,
                                    },
                                )

                            # Commit after processing each L1 cluster to ensure assignments are saved
                            try:
                                db.commit()
                                logger.info(
                                    f"Committed assignments for {len(l1_cluster_texts)} texts for L1 cluster {level1_cluster_db_id}"
                                )
                            except Exception as commit_error:
                                logger.error(
                                    f"Error committing assignments for L1 cluster {level1_cluster_db_id}: {commit_error}"
                                )
                                db.rollback()
                                raise

                        except Exception as e_l1:
                            logger.error(
                                f"Failed to process Level1Cluster '{level1_title}' (local L1 ID {l1_id_local}) for category {category_db_id}, dataset {dataset_id}: {e_l1}"
                            )
                            db.rollback()  # Rollback if insert fails
                            # Continue to next L1 cluster even if one fails

                    # ---- Handle remaining L1 clustering noise ----
                    l1_noise_mask_local = l1_clusters_local == -1
                    num_l1_noise = np.sum(l1_noise_mask_local)

                    if num_l1_noise > 0:
                        logger.info(
                            f"Handling {num_l1_noise} remaining L1 noise points for category {category_db_id} by creating/using miscellaneous Level1Cluster."
                        )
                        l1_noise_indices_local = np.where(l1_noise_mask_local)[0]
                        # Map local L1 noise indices back to original text list indices
                        original_l1_noise_indices = cluster_indices[
                            l1_noise_indices_local
                        ]
                        l1_noise_texts = [texts[i] for i in original_l1_noise_indices]
                        # Get L2 probabilities corresponding to the L1 noise texts
                        l2_noise_probs = l2_probabilities_for_category[
                            l1_noise_indices_local
                        ]

                        if l1_noise_texts:
                            misc_l1_title = (
                                f"Miscellaneous {category_name}"  # Renamed variable
                            )
                            misc_l1_cluster_db_id = None  # Renamed variable
                            try:
                                # --- Insert misc Level1Cluster using retry_on_lock (keep for this) ---
                                with retry_on_lock() as misc_conn:
                                    # Insert misc Level1Cluster
                                    misc_conn.execute(
                                        """
                                        INSERT INTO level1_clusters (category_id, version, l1_cluster_id, title)
                                        VALUES (?, ?, ?, ?)
                                        RETURNING id
                                        """,
                                        [
                                            category_db_id,
                                            version,
                                            -1,  # Use -1 for L1 noise cluster ID
                                            misc_l1_title,
                                        ],
                                    )
                                    result = misc_conn.fetchone()
                                    if result:
                                        misc_l1_cluster_db_id = result[0]
                                        logger.info(
                                            f"Inserted miscellaneous Level1Cluster ID {misc_l1_cluster_db_id} for category {category_db_id}, version {version}."
                                        )
                                    else:
                                        raise Exception(
                                            "Misc Level1Cluster insertion failed."
                                        )

                                # --- Insert text-assignments for L1 noise using main SQLAlchemy session 'db' ---
                                # First, delete any existing assignments for these noise texts in this version
                                for text_content in l1_noise_texts:
                                    # Find the text_id by querying the TextDB table directly
                                    text_query = sql_text(
                                        """
                                        SELECT id FROM texts 
                                        WHERE dataset_id = :dataset_id AND text = :text_content
                                        """
                                    )
                                    text_result = db.execute(
                                        text_query,
                                        {
                                            "dataset_id": dataset_id,
                                            "text_content": text_content,
                                        },
                                    ).fetchone()

                                    if text_result:
                                        text_id = text_result[0]
                                        # Delete existing assignment for this text_id and version
                                        delete_query = sql_text(
                                            """
                                            DELETE FROM text_assignments
                                            WHERE text_id = :text_id AND version = :version
                                            """
                                        )
                                        db.execute(
                                            delete_query,
                                            {"text_id": text_id, "version": version},
                                        )

                                # Now insert noise assignments
                                for i, text_content in enumerate(l1_noise_texts):
                                    # Use L2 probability as both L1 and L2 prob here
                                    l1_prob = (
                                        float(l2_noise_probs[i])
                                        if i < len(l2_noise_probs)
                                        else 0.0
                                    )
                                    l2_prob = (
                                        float(l2_noise_probs[i])
                                        if i < len(l2_noise_probs)
                                        else 0.0
                                    )
                                    # Simple insert without ON CONFLICT
                                    insert_sql_noise = sql_text(
                                        """
                                        INSERT INTO text_assignments (text_id, version, level1_cluster_id, l1_probability, l2_probability)
                                        SELECT t.id, :version, :l1_cluster_db_id, :l1_prob, :l2_prob
                                        FROM texts t
                                        WHERE t.dataset_id = :dataset_id AND t.text = :text_content
                                        """
                                    )
                                    db.execute(
                                        insert_sql_noise,
                                        {
                                            "version": version,
                                            "l1_cluster_db_id": misc_l1_cluster_db_id,
                                            "l1_prob": l1_prob,
                                            "l2_prob": l2_prob,
                                            "dataset_id": dataset_id,
                                            "text_content": text_content,
                                        },
                                    )

                                # Commit after processing noise to ensure assignments are saved
                                try:
                                    db.commit()
                                    logger.info(
                                        f"Committed assignments for {len(l1_noise_texts)} noise texts to misc L1 cluster {misc_l1_cluster_db_id}"
                                    )
                                except Exception as commit_error:
                                    logger.error(
                                        f"Error committing noise assignments: {commit_error}"
                                    )
                                    db.rollback()
                                    raise
                            except Exception as e_misc_l1:  # Renamed variable
                                logger.error(
                                    f"Failed to create or link miscellaneous Level1Cluster for category {category_db_id}: {e_misc_l1}"
                                )
                                db.rollback()  # Rollback if insert fails
                    # ---- END: Handle L1 noise ----

                except Exception as e_l1_clustering:  # Renamed variable
                    logger.error(
                        f"Failed Level 1 clustering for category {category_db_id} (L2 cluster {l2_cluster_id_hdbscan}), dataset {dataset_id}: {e_l1_clustering}. Creating single fallback Level1Cluster."
                    )
                    # Fallback: Create a single Level1Cluster if L1 clustering fails
                    await create_single_level1_cluster(  # Call renamed helper
                        category_db_id,
                        cluster_texts,
                        l2_probabilities_for_category,  # Pass L2 probs
                        dataset_id,
                        version,
                    )

            else:
                # If L2 category cluster is too small or failed checks for L1 clustering, create one Level1Cluster
                logger.info(
                    f"Creating single Level1Cluster for category {category_db_id} (L2 cluster {l2_cluster_id_hdbscan}), dataset {dataset_id} as L1 clustering was skipped."
                )
                await create_single_level1_cluster(  # Call renamed helper
                    category_db_id,
                    cluster_texts,
                    l2_probabilities_for_category,
                    dataset_id,
                    version,
                )

            del cluster_embeddings  # Clean up memory
            gc.collect()
            logger.info(
                f"Finished processing L2 category cluster {l2_cluster_id_hdbscan} for dataset {dataset_id}."
            )

        del embeddings_array, clusters, probabilities  # Free large numpy arrays
        gc.collect()
        logger.info(
            f"Finished all category and cluster processing for dataset {dataset_id}."
        )

        # Final DB updates and return using the main SQLAlchemy session `db`
        try:
            logger.info(
                f"Updating dataset status and fetching final results for dataset {dataset_id}."
            )
            # Update dataset clustering status using the original SQLAlchemy session
            dataset = (
                db.query(DatasetMetadata)
                .filter(DatasetMetadata.id == dataset_id)
                .first()
            )
            if dataset:
                dataset.is_clustered = True
                # dataset.clustering_status = 'completed' # Status is updated in the background task runner
                db.commit()
                logger.info(f"Committed is_clustered=True for dataset {dataset_id}.")
            else:
                logger.warning(
                    f"DatasetMetadata with id {dataset_id} not found in SQLAlchemy session. Cannot update status."
                )
                db.rollback()  # Rollback if dataset not found

            # Get final results using the original SQLAlchemy session
            final_categories = (
                db.query(Category)
                .options(joinedload(Category.level1_clusters))  # Eager load L1 clusters
                .filter(Category.dataset_id == dataset_id, Category.version == version)
                .all()
            )
            # No need to query Level1Cluster separately if eager loaded
            # final_level1_clusters = []
            # if final_categories:
            #     category_ids = [c.id for c in final_categories]
            #     final_level1_clusters = (
            #         db.query(Level1Cluster)
            #         .filter(Level1Cluster.category_id.in_(category_ids), Level1Cluster.version == version)
            #         .all()
            #     )

            logger.info(
                f"Successfully fetched {len(final_categories)} categories (with L1 clusters eager loaded) for dataset {dataset_id}, version {version}."
            )

            # The background task runner expects (categories, subclusters), but subclusters are now Level1Clusters
            # For now, return categories only, assuming the runner handles this.
            # Alternatively, adjust the runner `run_clustering_task`
            return final_categories, []  # Return empty list for second element for now

        except Exception as e_final:
            logger.error(
                f"Failed during final DB update/fetch for dataset {dataset_id}: {e_final}"
            )
            db.rollback()
            raise

    except Exception as e:
        logger.error(f"Error in clustering process for dataset {dataset_id}: {str(e)}")
        # Attempt to clean up DB entries for this version if clustering failed mid-way
        try:
            # Delete TextAssignments first
            db.query(TextAssignment).filter(
                TextAssignment.version == version,
                TextAssignment.text.has(TextDB.dataset_id == dataset_id),
            ).delete(synchronize_session=False)
            # Delete Level1Clusters
            db.query(Level1Cluster).filter(
                Level1Cluster.version == version,
                Level1Cluster.category.has(Category.dataset_id == dataset_id),
            ).delete(synchronize_session=False)
            # Delete Categories
            db.query(Category).filter(
                Category.version == version, Category.dataset_id == dataset_id
            ).delete(synchronize_session=False)
            # Delete orphaned Texts? Maybe not safe.
            db.commit()
            logger.info(
                f"Cleaned up partial results for failed clustering version {version} of dataset {dataset_id}."
            )
        except Exception as cleanup_e:
            logger.error(
                f"Failed to cleanup partial results for version {version} dataset {dataset_id}: {cleanup_e}"
            )
            db.rollback()
        raise
