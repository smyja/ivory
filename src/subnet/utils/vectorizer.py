import numpy as np
import logging
import os
from typing import List
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.ai import get_embed_client, get_embed_model

# Retry configuration
MAX_RETRIES = 3


@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60))
async def _embed_batch(create_call, model: str, batch: List[str]):
    response = await asyncio.to_thread(create_call, model=model, input=batch)
    return [item.embedding for item in response.data]

logger = logging.getLogger(__name__)


async def vectorize_texts(texts: List[str]) -> np.ndarray:
    """Vectorize texts using Together embeddings if configured, otherwise OpenAI."""
    try:
        client = get_embed_client()
        embed_model = get_embed_model()
        create_call = client.embeddings.create

        # Process texts in batches to avoid overwhelming the API
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await _embed_batch(create_call, embed_model, batch)
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array and ensure it's float32 for consistency
        return np.array(all_embeddings, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error vectorizing texts: {str(e)}")
        raise
