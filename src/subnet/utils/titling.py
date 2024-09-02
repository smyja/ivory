import os
from typing import List
from together import Together
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging

logger = logging.getLogger(__name__)

class TogetherClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            api_key = os.environ.get('TOGETHER_API_KEY')
            if not api_key:
                raise EnvironmentError("TOGETHER_API_KEY is not set. Please set this environment variable before running the application.")
            cls._instance = Together(api_key=api_key)
        return cls._instance

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
def generate_title_together(cluster_questions: List[str]) -> str:
    prompt = "Generate a short, descriptive title (5 words max) for this group of related questions:\n\n"
    prompt += "\n".join(cluster_questions[:5])  # Use up to 5 questions to generate the title
    
    try:
        client = TogetherClientSingleton.get_instance()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates short, descriptive titles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.5,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating title with Together API: {str(e)}")
        return "Untitled Cluster"