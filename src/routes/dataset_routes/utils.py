from typing import List
import os
import logging
from utils.ai import (
    get_embed_client,
    get_llm_client,
    get_embed_model,
    get_llm_model,
)

logger = logging.getLogger(__name__)

# Constants for clustering
BATCH_SIZE = 100  # Number of texts to process at once for embeddings
DISTANCE_THRESHOLD = 0.5  # Clustering distance threshold

# Default model configurations
TOGETHER_EMBED_MODEL = os.getenv(
    "TOGETHER_EMBED_MODEL", "togethercomputer/m2-bert-80M-32k-retrieval"
)
TOGETHER_LLM_MODEL = os.getenv("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct")


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using Together if configured, else OpenAI."""
    client = get_embed_client()
    response = client.embeddings.create(model=get_embed_model(), input=texts)
    return [item.embedding for item in response.data]


async def generate_category_name(texts: List[str]) -> str:
    """Generate a category name using Together only."""
    prompt = (
        "Create a specific and concise category name (1-3 words) for the following texts. "
        "The category should be descriptive and focused on the main theme.\n\n"
        "Texts:\n" + "\n".join(texts)
    )

    client = get_llm_client()
    response = client.chat.completions.create(
        model=get_llm_model(),
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
