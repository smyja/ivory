import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datasets import (
    load_dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    Dataset,
    DatasetDict,
)
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
    If 'split' is None, attempts to download and concatenate all available splits.

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    log_split = split if split else "all"
    logger.info(
        f"Downloading {dataset_name} (config={config}, split={log_split}, revision={revision}, limit={limit_rows}, cols={selected_columns})"
    )

    try:
        # Load dataset - 'split' parameter determines if we get Dataset or DatasetDict
        loaded_data = load_dataset(
            dataset_name,
            name=config,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        info = None
        features = None
        all_dfs = []
        total_rows_available = 0
        splits_loaded = []

        # --- Handle single split vs multiple splits ---
        if isinstance(loaded_data, DatasetDict):
            logger.info(f"Loaded DatasetDict with splits: {list(loaded_data.keys())}")
            # Combine all splits
            for split_name, dataset_split in loaded_data.items():
                splits_loaded.append(split_name)
                if not info:  # Get info/features from the first split
                    info = dataset_split.info
                    features = dataset_split.features

                split_rows = len(dataset_split)
                total_rows_available += split_rows
                logger.debug(f"Processing split '{split_name}' with {split_rows} rows.")
                split_df = dataset_split.to_pandas()
                split_df["_hf_split"] = split_name  # Add split identifier column
                all_dfs.append(split_df)

            if not all_dfs:
                raise ValueError("DatasetDict was empty or contained no valid splits.")

            # Concatenate DataFrames from all splits
            df = pd.concat(all_dfs, ignore_index=True)
            logger.info(
                f"Concatenated {len(splits_loaded)} splits into single DataFrame with {len(df)} rows (before limit)."
            )

        elif isinstance(loaded_data, Dataset):
            logger.info("Loaded single Dataset split.")
            # Single split loaded directly
            splits_loaded.append(split)  # Store the requested split name
            info = loaded_data.info
            features = loaded_data.features
            total_rows_available = len(loaded_data)
            df = loaded_data.to_pandas()
            df["_hf_split"] = split  # Add split identifier column
        else:
            raise TypeError(
                f"load_dataset returned unexpected type: {type(loaded_data)}"
            )
        # --- End split handling ---

        schema_dict = (
            features.to_dict()
            if features and hasattr(features, "to_dict")
            else str(features)
        )

        # Handle limiting rows AFTER combining splits (if applicable)
        if limit_rows is not None and limit_rows > 0:
            actual_limit = min(limit_rows, len(df))
            if actual_limit < len(df):
                logger.info(
                    f"Limiting combined DataFrame rows from {len(df)} to {actual_limit}."
                )
                df = df.iloc[:actual_limit]
            else:
                logger.info(
                    f"Row limit ({limit_rows}) is >= total rows ({len(df)}), using all rows."
                )
        else:
            logger.info(
                f"No row limit specified or limit is 0, using all {len(df)} rows."
            )

        # Select specific columns if requested (after potentially adding _hf_split)
        if selected_columns:
            # Ensure _hf_split is included if it exists, if user didn't explicitly select it
            if "_hf_split" in df.columns and "_hf_split" not in selected_columns:
                selected_columns_internal = selected_columns + ["_hf_split"]
            else:
                selected_columns_internal = selected_columns

            valid_columns = [
                col for col in selected_columns_internal if col in df.columns
            ]
            missing_columns = set(selected_columns_internal) - set(valid_columns)
            if missing_columns:
                logger.warning(
                    f"Requested columns not found and will be ignored: {missing_columns}"
                )
            if not valid_columns:
                raise ValueError("No valid columns selected or found in the dataset.")

            # Ensure _hf_split is kept if it exists
            if "_hf_split" in df.columns and "_hf_split" not in valid_columns:
                valid_columns.append("_hf_split")

            logger.info(f"Selecting final columns: {valid_columns}")
            df = df[valid_columns]

        # Extract schema information based on the FINAL DataFrame
        final_columns = list(df.columns)
        schema_info = {
            "columns": final_columns,
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "num_rows": len(df),  # Rows actually in the final DataFrame
            "total_rows_available": total_rows_available,  # Rows before limiting
            "features": schema_dict,  # Original HF features dict
            "splits_loaded": splits_loaded,  # List of splits included
            "revision_used": info.version if info and info.version else revision,
            "description": info.description if info else None,
            "citation": info.citation if info else None,
            "homepage": info.homepage if info else None,
            "license": info.license if info else None,
        }
        # Ensure _hf_split type is captured if present
        if "_hf_split" in df.columns:
            schema_info["column_types"]["_hf_split"] = str(df["_hf_split"].dtype)

        logger.info(
            f"Returning DataFrame with {len(df)} rows and {len(final_columns)} columns. Splits loaded: {splits_loaded}"
        )
        return df, schema_info

    except Exception as e:
        logger.error(
            f"Error downloading dataset '{dataset_name}' (config: {config}, split: {log_split}, revision: {revision}): {e}"
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
    Adjusts path if 'split' is None (meaning all splits were combined).

    Returns:
        Tuple (Path to parquet file, Path to schema file)
    """
    # Create directory structure
    safe_name = dataset_name.replace("/", "__")
    base_path = os.path.join("datasets", safe_name)

    if config:
        safe_config = config.replace("/", "__")
        base_path = os.path.join(base_path, safe_config)

    # Only add split directory if a specific split was requested
    if split:
        safe_split = split.replace("/", "__")
        base_path = os.path.join(base_path, safe_split)
    # If split is None, we save directly under the config (or dataset name) directory

    os.makedirs(base_path, exist_ok=True)

    # Save to Parquet (add stable row id if missing)
    parquet_path = os.path.join(base_path, "data.parquet")
    try:
        try:
            import hashlib
            import json as _json
            import pandas as _pd  # for type hints in lambda

            if "__row_id" not in df.columns:
                cols = sorted([c for c in df.columns if c != "__row_id"])

                def _row_hash(series: _pd.Series) -> str:
                    parts = []
                    for col in cols:
                        val = series.get(col, None)
                        try:
                            parts.append(_json.dumps(val, sort_keys=True, ensure_ascii=False, default=str))
                        except Exception:
                            parts.append(str(val))
                    base = "\u241F".join(parts)
                    return hashlib.sha1(base.encode("utf-8")).hexdigest()

                df["__row_id"] = df.apply(_row_hash, axis=1)
        except Exception:
            # Best-effort: continue without row ids
            pass

        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved dataset ({len(df)} rows) to: {parquet_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to Parquet at {parquet_path}: {e}")
        raise

    # Save schema information (basic structure reflecting the final DataFrame)
    schema = {
        "columns": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "num_rows": len(df),
    }
    # Ensure _hf_split type is captured if present
    if "_hf_split" in df.columns:
        schema["column_types"]["_hf_split"] = str(df["_hf_split"].dtype)

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
