import os
import json
from typing import List
from together import Together
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
            api_key = os.environ.get('TOGETHER_API_KEY')
            if not api_key:
                raise EnvironmentError("TOGETHER_API_KEY environment variable is not set.")
            cls._instance = Together(api_key=api_key)
        return cls._instance

class TitleOutput(BaseModel):
    title: str = Field(description="A specific and concise title (3-5 words) for the group of questions")
    reasoning: str = Field(description="A brief explanation of why this title was chosen")

class CategoryOutput(BaseModel):
    category: str = Field(description="A specific, overarching category (1-3 words) for the group of titles")
    reasoning: str = Field(description="A brief explanation of why this category was chosen")

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60))
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
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are an expert in creating concise, specific titles for groups of related questions. Respond only in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS_TITLE,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            response_format={
                "type": "json_object",
                "schema": TitleOutput.model_json_schema(),
            },
        )
        output = json.loads(response.choices[0].message.content)
        title_output = TitleOutput(**output)
        logger.info(f"Generated title: {title_output.title}. Reasoning: {title_output.reasoning}")
        return title_output.title
    except Exception as e:
        logger.error(f"Error in semantic title generation: {str(e)}")
        return "Miscellaneous Questions"

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, max=60))
async def derive_overarching_category(cluster_titles: List[str]) -> str:
    if len(cluster_titles) == 1:
        return cluster_titles[0]
    
    prompt = (
        "Create a specific, overarching category (1-3 words) for the following group of titles. "
        "The category should be descriptive, focused, and accurately reflect the subject matter. "
        "Avoid overly broad categories like 'General Knowledge' or 'Introductory Sciences'.\n\n"
        "Titles:\n" + "\n".join(cluster_titles)
    )
    
    try:
        client = LLMClient.get_instance()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages = [
                {"role": "system", "content": "You are an expert in creating specific, focused categories for groups of related titles. Your categories should accurately reflect the subject matter and be more specific than broad fields like 'Science' or 'General Knowledge'. Respond only in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS_CATEGORY,
            temperature=0.3,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.2,
            response_format={
                "type": "json_object",
                "schema": CategoryOutput.model_json_schema(),
            },
        )
        output = json.loads(response.choices[0].message.content)
        category_output = CategoryOutput(**output)
        logger.info(f"Generated category: {category_output.category}. Reasoning: {category_output.reasoning}")
        return category_output.category
    except Exception as e:
        logger.error(f"Error in overarching category derivation: {str(e)}")
        return "Miscellaneous Topics"