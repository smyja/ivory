import os
import json
from typing import List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import asyncio
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants
MAX_QUESTIONS_FOR_TITLE = 5
MAX_RETRIES = 3
MAX_TOKENS_TITLE = 50
MAX_TOKENS_CATEGORY = 50


class LLMClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY environment variable is not set."
                )
            cls._instance = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return cls._instance


class TitleOutput(BaseModel):
    title: str = Field(
        description="A specific and concise title (3-5 words) for the group of questions"
    )
    reasoning: str = Field(
        description="A brief explanation of why this title was chosen"
    )


class CategoryOutput(BaseModel):
    category: str = Field(
        description="A specific, overarching category 1 or 2 wordsfor the group of titles"
    )
    reasoning: str = Field(
        description="A brief explanation of why this category was chosen"
    )


@retry(
    stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60)
)
async def generate_semantic_title(texts: List[str]) -> str:
    """
    Generate a more accurate and descriptive title for a subcluster of texts.
    Uses a more detailed prompt to ensure better titling.
    """
    try:
        # Prepare the prompt with more context and examples
        prompt = f"""Given the following texts, generate a concise but descriptive title that accurately represents their specific theme or subject matter. The title should be 3-6 words maximum.

Texts:
{chr(10).join(f"- {text}" for text in texts[:10])}

Consider:
1. The specific topic or concept
2. The type of information (e.g., Facts, Events, Definitions)
3. Any distinguishing characteristics

Examples of good titles:
- Fundamental Physical Constants
- Major Historical Period Beginnings
- Southeast Asian Political Systems
- Basic Mathematical Theorems
- African Population Statistics

Generate only the title, nothing else."""

        # Get title from LLM
        title = await get_llm_response(prompt)

        # Clean and validate the response
        title = title.strip().strip('"').strip("'")
        if not title or len(title.split()) > 6:
            # Fallback to a generic title if the response is invalid
            return "General Information"

        return title
    except Exception as e:
        logger.error(f"Error generating semantic title: {str(e)}")
        return "General Information"


@retry(
    stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60)
)
async def derive_overarching_category(texts: List[str]) -> str:
    """
    Generate a more accurate and descriptive category name for a group of texts.
    Uses a more detailed prompt to ensure better categorization.
    """
    try:
        # Prepare the prompt with more context and examples
        prompt = f"""Analyze the following texts and identify the single most prominent, common theme or subject matter shared by the *majority* of the items. Generate a concise category name (2-4 words max) that accurately represents this primary theme. 

Texts:
{chr(10).join(f"- {text}" for text in texts[:15])}

Consider:
1. The main subject (e.g., Physics, History, Politics, Economics, Biology, Math)
2. The specific type of information (e.g., Facts, Events, Concepts, Systems, Figures)
3. The *most consistent* characteristic across the texts.

Examples of good category names:
- Physical Constants
- Historical Events
- Economic Concepts
- Mathematical Theorems
- Biological Facts
- Political Systems
- Country Demographics

Example of what NOT to do: If texts include political systems from Singapore, Thailand, Egypt, Kenya, and Morocco, do NOT name the category 'Asian Political Systems'; a better name would be 'Political Systems'.

Generate only the category name, nothing else."""

        # Get category name from LLM
        category_name = await get_llm_response(prompt)

        # Clean and validate the response
        category_name = category_name.strip().strip('"').strip("'")
        # Allow slightly more variation, up to 5 words, but check for emptiness
        if not category_name or len(category_name.split()) > 5:
            logger.warning(
                f"Generated category name '{category_name}' is invalid (empty or too long), falling back."
            )
            return "General Information"

        return category_name
    except Exception as e:
        logger.error(f"Error generating category name: {str(e)}")
        return "General Information"


@retry(
    stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60)
)
async def get_llm_response(prompt: str) -> str:
    """
    Get a response from the LLM using the OpenAI API.

    Args:
        prompt: The prompt to send to the LLM

    Returns:
        The LLM's response as a string
    """
    try:
        client = LLMClient.get_instance()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="anthropic/claude-3.7-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in creating concise, specific titles and categories for groups of related texts. Provide only the requested output, nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS_TITLE,
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": "https://github.com/ivory",
                "X-Title": "Ivory",
            },
        )
        # Add robust checks for the response structure
        if (
            response
            and response.choices
            and len(response.choices) > 0
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            result = response.choices[0].message.content.strip()
            logger.info(f"Generated LLM response: {result}")
            return result
        else:
            logger.error(
                f"Received invalid or empty response structure from LLM: {response}"
            )
            return None  # Or raise a specific error

    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        raise
