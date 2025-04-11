from typing import List
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Constants for clustering
BATCH_SIZE = 100  # Number of texts to process at once for embeddings
DISTANCE_THRESHOLD = 0.5  # Clustering distance threshold

# Model configurations
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENROUTER_LLM_MODEL = "anthropic/claude-3.7-sonnet"


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def generate_category_name(texts: List[str]) -> str:
    """Generate a category name for a cluster of texts using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY environment variable is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = (
        "Create a specific and concise category name (1-3 words) for the following texts. "
        "The category should be descriptive and focused on the main theme.\n\n"
        "Texts:\n" + "\n".join(texts)
    )

    response = client.chat.completions.create(
        model=OPENROUTER_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in creating concise, specific category names.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
