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
from utils.database import get_duckdb_connection
import gc
import os
import time
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Constants - Relaxed parameters to allow cluster formation
MIN_CLUSTER_SIZE = 5  # Reduced min size for level 1
MIN_SAMPLES = 3  # Reduced min samples for level 1
CLUSTER_SELECTION_EPS = 0.05  # Keep epsilon as is for now
BATCH_SIZE = 512
MAX_RETRIES = 3
RETRY_DELAY = 1
MIN_NOISE_PROBABILITY_THRESHOLD = (
    0.3  # Increased minimum confidence to assign a noise point
)


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


async def create_single_subcluster(
    category_db_id: int,
    texts: List[str],
    probs: np.ndarray,
    dataset_id: int,
    version: int,
):
    """Helper function to create a single subcluster when subclustering is skipped or fails."""
    if not texts:
        return

    # Generate title semantically
    subcluster_title = await generate_semantic_title(texts)
    if subcluster_title:
        subcluster_title = " ".join(subcluster_title.split()[:5])  # Limit title length

    try:
        with retry_on_lock() as subcluster_conn:
            subcluster_conn.execute(
                """
                INSERT INTO subclusters (category_id, title, row_count, percentage, version)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """,
                [
                    category_db_id,
                    subcluster_title or "General",  # Fallback title
                    len(texts),
                    100.0,  # Takes 100% of the category
                    version,
                ],
            )
            subcluster_db_id = subcluster_conn.fetchone()[0]

            # Insert text-subcluster relationships
            with retry_on_lock() as text_cluster_conn:
                for text, prob in zip(texts, probs):
                    # Ensure prob is a standard Python float
                    membership_score = (
                        float(prob)
                        if isinstance(prob, (np.float32, np.float64))
                        else prob
                    )
                    text_cluster_conn.execute(
                        """
                        INSERT INTO text_clusters (text_id, subcluster_id, membership_score)
                        SELECT t.id, ?, ?
                        FROM texts t
                        WHERE t.dataset_id = ? AND t.text = ?
                        ON CONFLICT (text_id, subcluster_id) 
                        DO UPDATE SET membership_score = ?
                        """,
                        [
                            subcluster_db_id,
                            membership_score,
                            dataset_id,
                            text,
                            membership_score,
                        ],
                    )
    except Exception as e:
        logger.error(
            f"Failed to create single subcluster for category {category_db_id}: {e}"
        )


async def cluster_texts(
    texts: List[str], db: Session, dataset_id: int, version: int = 1
) -> Tuple[List[Category], List[Subcluster]]:
    """
    Cluster texts into categories and subclusters.

    Args:
        texts: List of texts to cluster
        db: Database session
        dataset_id: ID of the dataset
        version: Version number for this clustering attempt

    Returns:
        Tuple of (categories, subclusters)
    """
    final_categories = []
    final_subclusters = []
    try:
        # Clean up existing data
        cleanup_dataset_data(dataset_id)

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
            f"Processing {len(unique_clusters)} unique clusters for dataset {dataset_id}."
        )

        # Process each cluster
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_probs = probabilities[cluster_mask]

            if not cluster_texts:  # Skip empty clusters if any somehow occur
                logger.warning(
                    f"Skipping empty cluster {cluster_id} for dataset {dataset_id}."
                )
                continue

            # Generate category name using the semantic function
            category_name = await derive_overarching_category(cluster_texts)

            # Limit category name length (simple approach)
            if category_name:
                category_name = " ".join(
                    category_name.split()[:3]
                )  # Limit to first 3 words
            else:
                category_name = f"Category {cluster_id}"  # Fallback name

            # Create category in DB
            category_db_id = None
            try:
                with retry_on_lock() as category_conn:
                    category_conn.execute(
                        """
                        INSERT INTO categories (dataset_id, name, total_rows, percentage, version)
                        VALUES (?, ?, ?, ?, ?)
                        RETURNING id
                        """,
                        [
                            dataset_id,
                            category_name,
                            len(cluster_texts),
                            (
                                len(cluster_texts) / total_texts * 100
                                if total_texts > 0
                                else 0
                            ),
                            version,
                        ],
                    )
                    result = category_conn.fetchone()
                    if result:
                        category_db_id = result[0]
                    else:
                        raise Exception("Category insertion did not return an ID.")
            except Exception as e:
                logger.error(
                    f"Failed to insert category '{category_name}' for cluster {cluster_id}, dataset {dataset_id}: {e}"
                )
                continue  # Skip processing this cluster if category insertion fails

            # Subclustering logic within each category
            cluster_embeddings = embeddings_array[cluster_mask]  # Already normalized

            # Determine if subclustering is needed
            # Note: Subclustering still uses smaller min_samples/min_cluster_size defined inline
            should_subcluster = len(
                cluster_texts
            ) >= MIN_CLUSTER_SIZE * 2 and cluster_embeddings.shape[  # Check against level 1 size
                0
            ] >= max(
                MIN_SAMPLES - 1, 1
            )  # Check against level 2 min_samples

            if should_subcluster:
                logger.info(
                    f"Starting level 2 subclustering for category {category_db_id} (cluster {cluster_id}), dataset {dataset_id} with {cluster_embeddings.shape[0]} items."
                )
                try:
                    # Subclustering uses smaller parameters defined inline here
                    subclusterer = hdbscan.HDBSCAN(
                        min_cluster_size=max(
                            max(MIN_SAMPLES - 1, 1),
                            2,  # Keep subcluster min_size small (e.g., 2 or original MIN_SAMPLES-1)
                        ),
                        min_samples=max(
                            MIN_SAMPLES - 1, 1
                        ),  # Keep subcluster min_samples small
                        cluster_selection_epsilon=CLUSTER_SELECTION_EPS * 0.8,
                        metric="euclidean",  # Still euclidean on normalized data
                        cluster_selection_method="leaf",
                        prediction_data=True,
                    )
                    subclusters = subclusterer.fit_predict(cluster_embeddings)
                    subcluster_probs = subclusterer.probabilities_
                    gc.collect()
                    logger.info(
                        f"Level 2 subclustering complete for category {category_db_id}. Found {len(np.unique(subclusters[subclusters != -1]))} subclusters."
                    )

                    # Handle noise in subclusters with threshold
                    subcluster_noise_mask = subclusters == -1
                    num_sub_noise = np.sum(subcluster_noise_mask)
                    assigned_sub_noise_count = 0
                    if num_sub_noise > 0 and num_sub_noise < len(subclusters):
                        logger.info(
                            f"Assigning {num_sub_noise} noise points in subcluster for category {category_db_id}, dataset {dataset_id} (threshold={MIN_NOISE_PROBABILITY_THRESHOLD})."
                        )
                        noise_sub_vectors = cluster_embeddings[subcluster_noise_mask]
                        try:
                            soft_sub_clusters = hdbscan.membership_vector(
                                subclusterer, noise_sub_vectors
                            )
                            if soft_sub_clusters.ndim < 2:
                                soft_sub_clusters = soft_sub_clusters.reshape(-1, 1)

                            if soft_sub_clusters.shape[1] > 0:
                                subcluster_noise_labels = np.argmax(
                                    soft_sub_clusters, axis=1
                                )
                                subcluster_noise_probs = np.max(
                                    soft_sub_clusters, axis=1
                                )

                                original_sub_noise_indices = np.where(
                                    subcluster_noise_mask
                                )[0]

                                for i in range(num_sub_noise):
                                    if (
                                        subcluster_noise_probs[i]
                                        >= MIN_NOISE_PROBABILITY_THRESHOLD
                                    ):
                                        original_idx = original_sub_noise_indices[i]
                                        subclusters[original_idx] = (
                                            subcluster_noise_labels[i]
                                        )
                                        subcluster_probs[original_idx] = (
                                            subcluster_noise_probs[i]
                                        )
                                        assigned_sub_noise_count += 1
                                logger.info(
                                    f"Successfully assigned {assigned_sub_noise_count} / {num_sub_noise} subcluster noise points for category {category_db_id}."
                                )
                            else:
                                logger.warning(
                                    f"membership_vector returned no cluster probabilities for subcluster noise points in category {category_db_id}. Skipping assignment."
                                )

                        except Exception as e_sub_noise:
                            logger.warning(
                                f"Failed to assign subcluster noise for category {category_db_id}, dataset {dataset_id}: {e_sub_noise}"
                            )
                        finally:
                            del noise_sub_vectors, soft_sub_clusters
                            gc.collect()
                    elif num_sub_noise == len(subclusters):
                        logger.warning(
                            f"All points ({num_sub_noise}) were noise in subclustering for category {category_db_id}. Creating single subcluster."
                        )
                        await create_single_subcluster(
                            category_db_id,
                            cluster_texts,
                            cluster_probs,
                            dataset_id,
                            version,
                        )
                        del subclusterer  # Free memory
                        gc.collect()
                        continue  # Skip rest of subcluster processing loop

                    del subcluster_noise_mask  # Free memory
                    del subclusterer  # Free HDBSCAN object memory
                    gc.collect()

                    unique_subclusters = np.unique(subclusters[subclusters != -1])
                    logger.info(
                        f"Processing {len(unique_subclusters)} unique subclusters for category {category_db_id}, dataset {dataset_id}."
                    )

                    # Process each subcluster
                    for (
                        sub_id_local
                    ) in unique_subclusters:  # Use a different name to avoid confusion
                        subcluster_mask = subclusters == sub_id_local
                        subcluster_indices = np.where(subcluster_mask)[0]
                        # Map local indices back to original text list indices
                        original_indices = cluster_indices[subcluster_indices]
                        subcluster_texts = [texts[i] for i in original_indices]
                        # Get probabilities corresponding to the subcluster texts
                        subcluster_probs_filtered = subcluster_probs[subcluster_mask]

                        if not subcluster_texts:
                            logger.warning(
                                f"Skipping empty subcluster {sub_id_local} for category {category_db_id}, dataset {dataset_id}."
                            )
                            continue

                        # Generate subcluster title semantically
                        subcluster_title = await generate_semantic_title(
                            subcluster_texts
                        )
                        if subcluster_title:
                            subcluster_title = " ".join(
                                subcluster_title.split()[:5]
                            )  # Limit title length
                        else:
                            subcluster_title = (
                                f"Subcluster {sub_id_local}"  # Fallback name
                            )

                        # Create subcluster in DB
                        subcluster_db_id = None
                        try:
                            with retry_on_lock() as subcluster_conn:
                                subcluster_conn.execute(
                                    """
                                    INSERT INTO subclusters (category_id, title, row_count, percentage, version)
                                    VALUES (?, ?, ?, ?, ?)
                                    RETURNING id
                                    """,
                                    [
                                        category_db_id,
                                        subcluster_title,
                                        len(subcluster_texts),
                                        (
                                            len(subcluster_texts)
                                            / len(cluster_texts)
                                            * 100
                                            if len(cluster_texts) > 0
                                            else 0
                                        ),
                                        version,
                                    ],
                                )
                                sub_result = subcluster_conn.fetchone()
                                if sub_result:
                                    subcluster_db_id = sub_result[0]
                                else:
                                    raise Exception(
                                        "Subcluster insertion did not return an ID."
                                    )

                                # Insert text-subcluster relationships
                                with retry_on_lock() as text_cluster_conn:
                                    for text, prob in zip(
                                        subcluster_texts, subcluster_probs_filtered
                                    ):
                                        membership_score = (
                                            float(prob)
                                            if isinstance(
                                                prob, (np.float32, np.float64)
                                            )
                                            else prob
                                        )
                                        text_cluster_conn.execute(
                                            """
                                            INSERT INTO text_clusters (text_id, subcluster_id, membership_score)
                                            SELECT t.id, ?, ?
                                            FROM texts t
                                            WHERE t.dataset_id = ? AND t.text = ?
                                            ON CONFLICT (text_id, subcluster_id) 
                                            DO UPDATE SET membership_score = ?
                                            """,
                                            [
                                                subcluster_db_id,
                                                membership_score,
                                                dataset_id,
                                                text,
                                                membership_score,
                                            ],
                                        )
                        except Exception as e_sub:
                            logger.error(
                                f"Failed to process subcluster '{subcluster_title}' (local ID {sub_id_local}) for category {category_db_id}, dataset {dataset_id}: {e_sub}"
                            )
                            # Continue to next subcluster even if one fails

                except Exception as e_cluster:
                    logger.error(
                        f"Failed to subcluster category {category_db_id} (cluster {cluster_id}), dataset {dataset_id}: {e_cluster}. Creating single fallback subcluster."
                    )
                    # Fallback: Create a single subcluster for the whole category if subclustering fails
                    await create_single_subcluster(
                        category_db_id,
                        cluster_texts,
                        cluster_probs,
                        dataset_id,
                        version,
                    )

            else:
                # If cluster is too small or failed conditions for subclustering, create one subcluster
                logger.info(
                    f"Creating single subcluster for category {category_db_id} (cluster {cluster_id}), dataset {dataset_id} as it's too small or failed checks."
                )
                await create_single_subcluster(
                    category_db_id, cluster_texts, cluster_probs, dataset_id, version
                )

            del cluster_embeddings  # Clean up memory
            gc.collect()
            logger.info(
                f"Finished processing cluster {cluster_id} for dataset {dataset_id}."
            )

        del embeddings_array, clusters, probabilities  # Free large numpy arrays
        gc.collect()
        logger.info(f"Finished all cluster processing for dataset {dataset_id}.")

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
                db.commit()
                logger.info(f"Committed is_clustered=True for dataset {dataset_id}.")
            else:
                logger.warning(
                    f"DatasetMetadata with id {dataset_id} not found in SQLAlchemy session. Cannot update status."
                )
                db.rollback()  # Rollback if dataset not found

            # Get final results using the original SQLAlchemy session
            final_categories = (
                db.query(Category).filter(Category.dataset_id == dataset_id).all()
            )
            if final_categories:
                final_subclusters = (
                    db.query(Subcluster)
                    .filter(
                        Subcluster.category_id.in_([c.id for c in final_categories])
                    )
                    .all()
                )
            else:
                final_subclusters = []

            logger.info(
                f"Successfully fetched {len(final_categories)} categories and {len(final_subclusters)} subclusters for dataset {dataset_id}."
            )
            return final_categories, final_subclusters

        except Exception as e_final:
            logger.error(
                f"Failed during final DB update/fetch for dataset {dataset_id}: {e_final}"
            )
            db.rollback()  # Rollback potential partial commit
            # Fallback: Try fetching directly via DuckDB connection as results might be committed partially
            logger.warning(
                f"Attempting fallback fetch for dataset {dataset_id} via direct DuckDB connection."
            )
            try:
                with retry_on_lock() as final_conn:
                    cats_raw = final_conn.execute(
                        "SELECT id, name, total_rows, percentage FROM categories WHERE dataset_id = ?",
                        [dataset_id],
                    ).fetchall()
                    cat_ids = [c[0] for c in cats_raw]
                    # Basic conversion - assumes Category/Subcluster are simple data holders
                    final_categories = [
                        Category(
                            id=c[0],
                            name=c[1],
                            total_rows=c[2],
                            percentage=c[3],
                            dataset_id=dataset_id,
                        )
                        for c in cats_raw
                    ]

                    if cat_ids:
                        subs_raw = final_conn.execute(
                            f"SELECT id, category_id, title, row_count, percentage FROM subclusters WHERE category_id IN ({','.join(['?']*len(cat_ids))})",
                            cat_ids,
                        ).fetchall()
                        final_subclusters = [
                            Subcluster(
                                id=s[0],
                                category_id=s[1],
                                title=s[2],
                                row_count=s[3],
                                percentage=s[4],
                            )
                            for s in subs_raw
                        ]
                    else:
                        final_subclusters = []
                    logger.info(
                        f"Fallback fetch successful for dataset {dataset_id}: got {len(final_categories)} categories, {len(final_subclusters)} subclusters."
                    )
                    return final_categories, final_subclusters
            except Exception as e_fallback:
                logger.error(
                    f"Fallback fetch failed for dataset {dataset_id}: {e_fallback}"
                )
                return [], []  # Return empty if fallback also fails

    except Exception as e:
        logger.exception(
            f"CRITICAL error during clustering for dataset {dataset_id}: {str(e)}"
        )
        # Ensure DB session is rolled back in case of critical failure before returning
        try:
            db.rollback()
            logger.info(
                f"Rolled back SQLAlchemy session due to critical error for dataset {dataset_id}."
            )
        except Exception as e_rollback:
            logger.error(
                f"Failed to rollback SQLAlchemy session for dataset {dataset_id}: {e_rollback}"
            )
        # Return empty lists as the process failed critically
        return [], []
