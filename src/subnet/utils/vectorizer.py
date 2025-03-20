import numpy as np
import logging
import requests
import os
from typing import List

logger = logging.getLogger(__name__)


async def vectorize_texts(texts: List[str]) -> np.ndarray:
    """Convert a list of texts to their vector representations using Together AI's embedding API."""
    try:
        url = "https://api.together.xyz/v1/embeddings"
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError("TOGETHER_API_KEY environment variable is not set.")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }

        # Process texts in batches to avoid overwhelming the API
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": "WhereIsAI/UAE-Large-V1", "input": batch}

            # Use httpx or aiohttp for async requests
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(
                            f"API request failed with status {response.status}: {await response.text()}"
                        )

                    # Extract embeddings from the nested response structure
                    response_data = await response.json()
                    batch_embeddings = [
                        item["embedding"] for item in response_data["data"]
                    ]
                    all_embeddings.extend(batch_embeddings)

        # Convert to numpy array and ensure it's float32 for consistency
        return np.array(all_embeddings, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error vectorizing texts: {str(e)}")
        raise


async def test_vectorizer():
    """Test the vectorizer with sample texts."""
    # Sample texts for testing
    test_texts = [
        "Our solar system orbits the Milky Way galaxy at about 515,000 mph",
        "The Earth is the third planet from the Sun",
        "Mars is known as the Red Planet",
    ]

    try:
        # Get embeddings
        embeddings = await vectorize_texts(test_texts)

        # Print shape and basic statistics
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Number of texts: {len(test_texts)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Mean value: {np.mean(embeddings):.4f}")
        print(f"Std value: {np.std(embeddings):.4f}")

        # Verify data type
        print(f"Data type: {embeddings.dtype}")

        # Calculate cosine similarity between first two texts
        if len(embeddings) >= 2:
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"Cosine similarity between first two texts: {similarity:.4f}")

        return True

    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_vectorizer())
