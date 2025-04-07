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
async def generate_semantic_title(question_subset: List[str]) -> str:
    prompt = (
        "Create a specific and concise title (3-5 words) for the following group of questions. "
        "The title should be clear, descriptive, and focused on the main theme.\n\n"
        "Questions:\n" + "\n".join(question_subset[:MAX_QUESTIONS_FOR_TITLE])
    )

    try:
        client = LLMClient.get_instance()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="anthropic/claude-3.7-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in creating concise, specific titles for groups of related questions. Provide only the title, nothing else.",
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
        title = response.choices[0].message.content.strip()
        logger.info(f"Generated title: {title}")
        return title
    except Exception as e:
        logger.error(f"Error in semantic title generation: {str(e)}")
        return "Miscellaneous Questions"


@retry(
    stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60)
)
async def derive_overarching_category(cluster_titles: List[str]) -> str:
    if len(cluster_titles) == 1:
        return cluster_titles[0]

    prompt = (
        "Create a specific, overarching category (1-3 words) for the following group of titles. "
        "The category should be descriptive, focused, and accurately reflect the subject matter. "
        "Titles:\n" + "\n".join(cluster_titles)
    )

    try:
        client = LLMClient.get_instance()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="anthropic/claude-3-sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in creating specific, focused categories for groups of related titles. Your categories should accurately reflect the subject matter and be more specific than broad fields like 'Science' or 'General Knowledge'. Provide only the category name, nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS_CATEGORY,
            temperature=0.3,
            extra_headers={
                "HTTP-Referer": "https://github.com/ivory",
                "X-Title": "Ivory",
            },
        )
        category = response.choices[0].message.content.strip()
        logger.info(f"Generated category: {category}")
        return category
    except Exception as e:
        logger.error(f"Error in overarching category derivation: {str(e)}")
        return "Miscellaneous Topics"
