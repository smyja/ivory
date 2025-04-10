import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
import requests
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from models import DatasetMetadata, TextDB
from utils.database import engine
from huggingface_hub import HfApi
from datasets.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)


def get_dataset_info(dataset_name: str, token: Optional[str] = None) -> Dict:
    """Get information about a Hugging Face dataset using HfApi."""
    api = HfApi()
    try:
        info = api.dataset_info(dataset_name, token=token)
        # Convert to dict for easier serialization if needed
        return info.__dict__
    except DatasetNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' not found on Hugging Face")
    except Exception as e:
        logger.error(f"Error fetching dataset info for {dataset_name} via HfApi: {e}")
        raise Exception(f"Error fetching dataset info: {e}")


def get_dataset_configs(dataset_name: str, token: Optional[str] = None) -> List[str]:
    """Get available configurations for a Hugging Face dataset."""
    try:
        # Use the official datasets library function
        configs = get_dataset_config_names(dataset_name, token=token)
        return configs
    except Exception as e:
        logger.error(f"Error getting configs for {dataset_name}: {e}")
        # Fallback or raise? Returning empty list for now.
        return []


def get_dataset_splits(
    dataset_name: str, config: Optional[str] = None, token: Optional[str] = None
) -> List[str]:
    """Get available splits for a Hugging Face dataset configuration."""
    try:
        # Use the official datasets library function
        splits = get_dataset_split_names(dataset_name, config, token=token)
        return splits
    except Exception as e:
        logger.error(f"Error getting splits for {dataset_name}/{config}: {e}")
        # Fallback or raise? Returning empty list for now.
        return []


def download_hf_dataset(
    dataset_name: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    limit_rows: Optional[int] = None,
    selected_columns: Optional[List[str]] = None,
    cache_dir: Optional[str] = "datasets/cache",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Download a dataset (or specific columns) from Hugging Face using a specific revision.

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    logger.info(
        f"Downloading {dataset_name} (config={config}, split={split}, revision={revision}, limit={limit_rows}, cols={selected_columns})"
    )

    try:
        dataset = load_dataset(
            dataset_name,
            name=config,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Get dataset info and features (schema)
        info = dataset.info
        features = dataset.features
        schema_dict = (
            features.to_dict() if hasattr(features, "to_dict") else str(features)
        )

        # Handle limiting rows
        total_rows_available = len(dataset)
        if limit_rows is not None and limit_rows > 0:
            logger.info(f"Limiting download to {limit_rows} rows.")
            actual_limit = min(limit_rows, total_rows_available)
            if actual_limit < total_rows_available:
                dataset = dataset.select(range(actual_limit))
        else:
            logger.info(
                f"No row limit specified, downloading all {total_rows_available} rows."
            )

        # Convert the potentially sliced dataset to DataFrame
        df = dataset.to_pandas()

        # Select specific columns if requested
        if selected_columns:
            # Ensure all requested columns exist, filter out those that don't
            valid_columns = [col for col in selected_columns if col in df.columns]
            missing_columns = set(selected_columns) - set(valid_columns)
            if missing_columns:
                logger.warning(
                    f"Requested columns not found and will be ignored: {missing_columns}"
                )
            if not valid_columns:
                raise ValueError("No valid columns selected or found in the dataset.")
            logger.info(f"Selecting columns: {valid_columns}")
            df = df[valid_columns]

        # Extract schema information based on the FINAL DataFrame (potentially column-filtered)
        schema_info = {
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "num_rows": len(df),
            "total_rows_available": total_rows_available,
            "features": schema_dict,
            "revision_used": info.version if info and info.version else revision,
            "description": info.description if info else None,
            "citation": info.citation if info else None,
            "homepage": info.homepage if info else None,
            "license": info.license if info else None,
        }

        return df, schema_info

    except Exception as e:
        logger.error(
            f"Error downloading dataset '{dataset_name}' (revision: {revision}): {e}"
        )
        raise


def generate_field_mappings(
    df: pd.DataFrame,
    text_field_hint: Optional[str] = None,
    label_field_hint: Optional[str] = None,
) -> Dict:
    """
    Generate mappings between dataset fields and our system.
    Uses the explicitly provided field mappings from the frontend.
    """
    # Validate the text field (required)
    if not text_field_hint or text_field_hint not in df.columns:
        logger.error(
            f"Text field '{text_field_hint}' not found in columns: {list(df.columns)}"
        )
        raise ValueError(
            f"The specified text field '{text_field_hint}' is not present in the dataset. "
            "Please select a valid text field from the available columns."
        )

    # Validate the label field (optional)
    if label_field_hint and label_field_hint not in df.columns:
        logger.warning(
            f"Specified label field '{label_field_hint}' not found in columns: {list(df.columns)}"
        )
        label_field_hint = None

    # Identify other fields
    other_fields = [
        col for col in df.columns if col not in [text_field_hint, label_field_hint]
    ]

    mappings = {
        "text_field": text_field_hint,
        "label_field": label_field_hint,  # Will be None if not found/specified
        "other_fields": other_fields,
    }
    logger.info(f"Using field mappings: {mappings}")
    return mappings


def save_dataset_to_parquet(
    df: pd.DataFrame,
    dataset_name: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> Tuple[str, str]:  # Return paths for parquet and schema
    """
    Save a dataset DataFrame to a Parquet file and its schema to JSON.

    Returns:
        Tuple (Path to parquet file, Path to schema file)
    """
    # Create directory structure
    safe_name = dataset_name.replace("/", "__")  # Use double underscore for safety
    base_path = os.path.join("datasets", safe_name)

    if config:
        # Sanitize config name as well if needed, though less likely to have slashes
        safe_config = config.replace("/", "__")
        base_path = os.path.join(base_path, safe_config)

    if split:
        safe_split = split.replace("/", "__")
        base_path = os.path.join(base_path, safe_split)

    os.makedirs(base_path, exist_ok=True)

    # Save to Parquet
    parquet_path = os.path.join(base_path, "data.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved dataset ({len(df)} rows) to: {parquet_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to Parquet at {parquet_path}: {e}")
        raise

    # Save schema information (basic structure)
    schema = {
        "columns": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "num_rows": len(df),
    }

    schema_path = os.path.join(base_path, "schema.json")
    try:
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info(f"Saved schema information to: {schema_path}")
    except Exception as e:
        logger.error(f"Error saving schema JSON to {schema_path}: {e}")
        # Continue even if schema saving fails, parquet is more important

    return parquet_path, schema_path


def insert_texts_from_dataframe(
    db: Session, df: pd.DataFrame, dataset_id: int, field_mappings: Dict
) -> int:
    """
    Insert texts from a DataFrame into the TextDB table using the
    explicitly specified text field from field_mappings.

    Returns:
        Number of successfully inserted texts (after deduplication)
    """
    text_field = field_mappings.get("text_field")

    # Check if the designated text field exists in the final DataFrame
    if not text_field or text_field not in df.columns:
        logger.error(
            f"Text field '{text_field}' specified in mappings not found in the DataFrame columns: {list(df.columns)}. Cannot insert texts."
        )
        raise ValueError(
            f"Text field '{text_field}' is required for insertion but was not found in the dataset. "
            "Please ensure you select a valid text field."
        )

    # Extract potential text content, convert to string, handle NaNs
    texts_to_insert = df[text_field].dropna().astype(str).tolist()

    if not texts_to_insert:
        logger.warning("No valid text records found in the specified text column.")
        return 0

    # Deduplicate texts before preparing records
    unique_texts = list(set(texts_to_insert))
    if len(unique_texts) < len(texts_to_insert):
        logger.info(
            f"Removed {len(texts_to_insert) - len(unique_texts)} duplicate texts before insertion."
        )

    # Prepare records for bulk insert
    text_records = [
        {"dataset_id": dataset_id, "text": text_content}
        for text_content in unique_texts
    ]

    if not text_records:
        logger.warning("No unique text records to insert after deduplication.")
        return 0

    inserted_count = 0
    try:
        # Query existing texts for this dataset to avoid violating unique constraint
        existing_texts_query = db.query(TextDB.text).filter(
            TextDB.dataset_id == dataset_id
        )
        existing_texts = {row[0] for row in existing_texts_query.all()}
        logger.info(
            f"Found {len(existing_texts)} existing texts for dataset {dataset_id}. Filtering new texts."
        )

        new_text_records = [
            record for record in text_records if record["text"] not in existing_texts
        ]

        if not new_text_records:
            logger.info(
                f"No new unique texts to insert for dataset {dataset_id}. All texts already exist."
            )
            # Return count of unique texts from the source that are now associated
            return len(unique_texts)

        logger.info(
            f"Attempting to bulk insert {len(new_text_records)} new unique texts for dataset {dataset_id}."
        )
        db.bulk_insert_mappings(TextDB, new_text_records)
        db.commit()  # Commit the insertion
        inserted_count = len(new_text_records)
        logger.info(f"Successfully inserted {inserted_count} new unique texts.")

        # Return total unique texts from source now associated with the dataset
        return len(unique_texts)

    except Exception as e:
        logger.error(
            f"Error bulk inserting texts for dataset {dataset_id}: {e}", exc_info=True
        )
        db.rollback()
        raise  # Re-raise after rollback


def get_dataset_features(
    dataset_name: str, config: Optional[str] = None, token: Optional[str] = None
) -> Dict[str, Any]:
    """Get the features (schema) of a Hugging Face dataset configuration without downloading."""
    try:
        from datasets import (
            load_dataset_builder,
        )  # Use the builder to get info without full download

        builder = load_dataset_builder(dataset_name, name=config, token=token)
        # Access features from the builder's info
        features = builder.info.features
        if features:
            # Convert features to a serializable dictionary format
            feature_dict = (
                features.to_dict() if hasattr(features, "to_dict") else str(features)
            )
            # Extract basic column names and types if possible
            columns = {}
            if isinstance(features, dict):
                for name, feature_type in features.items():
                    columns[name] = str(feature_type)  # Simplified type representation
            elif hasattr(features, "keys"):  # Handle datasets.Features object
                for name in features.keys():
                    columns[name] = str(features[name])

            return {"features": feature_dict, "columns": columns}
        else:
            return {"features": {}, "columns": {}}
    except Exception as e:
        logger.error(f"Error getting features for {dataset_name}/{config}: {e}")
        raise Exception(f"Could not retrieve features for dataset: {e}")
