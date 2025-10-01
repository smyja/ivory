import os
import logging
from together import Together

logger = logging.getLogger(__name__)

_llm_client = None
_embed_client = None


def _require_together_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise EnvironmentError("TOGETHER_API_KEY is not set.")
    return key


def get_llm_model() -> str:
    return os.environ.get("TOGETHER_LLM_MODEL", "moonshotai/Kimi-K2-Instruct")


def get_embed_model() -> str:
    return os.environ.get(
        "TOGETHER_EMBED_MODEL", "togethercomputer/m2-bert-80M-32k-retrieval"
    )


def get_llm_client() -> Together:
    global _llm_client
    if _llm_client is None:
        key = _require_together_key()
        _llm_client = Together(api_key=key)
        logger.info(f"Initialized Together LLM client (model={get_llm_model()})")
    return _llm_client


def get_embed_client() -> Together:
    global _embed_client
    if _embed_client is None:
        key = _require_together_key()
        _embed_client = Together(api_key=key)
        logger.info(
            f"Initialized Together Embedding client (model={get_embed_model()})"
        )
    return _embed_client

