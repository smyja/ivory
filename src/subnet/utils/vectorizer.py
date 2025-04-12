import numpy as np
import logging
import os
from typing import List
from openai import OpenAI
import asyncio

logger = logging.getLogger(__name__)


async def vectorize_texts(texts: List[str]) -> np.ndarray:
    """Convert a list of texts to their vector representations using OpenAI's embedding API."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        client = OpenAI(
            api_key=api_key,
        )

        # Process texts in batches to avoid overwhelming the API
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Use asyncio.to_thread to make the synchronous API call asynchronous
            response = await asyncio.to_thread(
                client.embeddings.create,
                model="text-embedding-3-small",  # OpenAI's latest embedding model
                input=batch,
            )

            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array and ensure it's float32 for consistency
        return np.array(all_embeddings, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error vectorizing texts: {str(e)}")
        raise

    return embeddings
