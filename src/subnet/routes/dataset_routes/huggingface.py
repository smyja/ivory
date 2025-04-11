from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime

from models import (
    DatasetMetadata,
    ClusteringStatus,
    TextDB,
)
from utils.database import get_db
from utils.huggingface_utils import (
    get_dataset_configs,
    get_dataset_splits,
    get_dataset_features,
    download_hf_dataset,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/info")
async def get_huggingface_dataset_info(dataset_name: str, token: Optional[str] = None):
    """Get information about a Hugging Face dataset."""
    try:
        configs = get_dataset_configs(dataset_name, token)
        return {"dataset": dataset_name, "configs": configs}
    except Exception as e:
        logger.error(f"Error getting HF dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def get_huggingface_dataset_configs(
    dataset_name: str, token: Optional[str] = None
):
    """Get available configurations for a Hugging Face dataset."""
    try:
        configs = get_dataset_configs(dataset_name, token)
        return configs
    except Exception as e:
        logger.error(f"Error getting HF dataset configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/splits")
async def get_huggingface_dataset_splits(
    dataset_name: str, config: Optional[str] = None, token: Optional[str] = None
):
    """Get available splits for a Hugging Face dataset configuration."""
    try:
        splits = get_dataset_splits(dataset_name, config, token)
        return splits
    except Exception as e:
        logger.error(f"Error getting HF dataset splits: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features", response_model=Dict)
async def get_huggingface_dataset_features(
    dataset_name: str, config: Optional[str] = None, token: Optional[str] = None
):
    """Get feature information for a Hugging Face dataset."""
    try:
        features = get_dataset_features(dataset_name, config, token)
        return features
    except Exception as e:
        logger.error(f"Error getting HF dataset features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_huggingface_dataset(
    db: Session,
    dataset_id: int,
    hf_dataset_name: str,
    hf_config: Optional[str] = None,
    hf_split: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    text_field_hint: Optional[str] = None,
    label_field_hint: Optional[str] = None,
    selected_columns: Optional[List[str]] = None,
    limit_rows: Optional[int] = None,
):
    """Process and save a HuggingFace dataset."""
    try:
        # Download the dataset
        df, schema_info = download_hf_dataset(
            dataset_name=hf_dataset_name,
            config=hf_config,
            split=hf_split,
            revision=hf_revision,
            token=hf_token,
            limit_rows=limit_rows,
            selected_columns=selected_columns,
        )

        # Try to auto-detect text field if not specified
        text_field = None
        if text_field_hint:
            # Try to find a field matching the hint
            matching_fields = [
                col for col in df.columns if text_field_hint.lower() in col.lower()
            ]
            if matching_fields:
                text_field = matching_fields[0]

        # If no text field found yet, look for common text field names
        if not text_field:
            common_text_fields = ["text", "content", "sentence", "document", "abstract"]
            for field in common_text_fields:
                matching_fields = [
                    col for col in df.columns if field.lower() in col.lower()
                ]
                if matching_fields:
                    text_field = matching_fields[0]
                    break

        # If still no text field, use the first string column
        if not text_field:
            for col in df.columns:
                if (
                    df[col].dtype == "object"
                    and df[col].apply(lambda x: isinstance(x, str)).all()
                ):
                    text_field = col
                    break

        if not text_field:
            raise ValueError("Could not determine text field in dataset")

        # Try to auto-detect label field if specified
        label_field = None
        if label_field_hint:
            matching_fields = [
                col for col in df.columns if label_field_hint.lower() in col.lower()
            ]
            if matching_fields:
                label_field = matching_fields[0]

        # Save texts to database
        for index, row in df.iterrows():
            text = str(row[text_field])
            label = (
                str(row[label_field]) if label_field and label_field in row else None
            )

            # Get additional metadata
            metadata = {}
            for col in df.columns:
                if col != text_field and col != label_field:
                    metadata[col] = str(row[col])

            # Create TextDB entry
            text_entry = TextDB(
                dataset_id=dataset_id,
                text=text,
                label=label,
                metadata=metadata,
            )
            db.add(text_entry)

        db.commit()

        # Update dataset metadata
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if dataset:
            dataset.status = "completed"
            dataset.download_date = datetime.now()
            dataset.clustering_status = ClusteringStatus.NOT_STARTED
            dataset.metadata = {
                "hf_dataset": hf_dataset_name,
                "hf_config": hf_config,
                "hf_split": hf_split,
                "hf_revision": hf_revision,
                "schema": schema_info,
                "text_field": text_field,
                "label_field": label_field,
                "num_rows": len(df),
            }
            db.commit()

        return {"message": "Dataset processed successfully", "id": dataset_id}
    except Exception as e:
        logger.error(f"Error processing HF dataset: {str(e)}")
        # Update dataset status
        dataset = (
            db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
        )
        if dataset:
            dataset.status = "failed"
            dataset.message = str(e)
            db.commit()
        raise e
