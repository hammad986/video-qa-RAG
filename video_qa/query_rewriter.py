import logging
import time
from typing import Optional
import google.generativeai as genai
from .config import config
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
logger = logging.getLogger(__name__)

# Constants
MAX_WORD_COUNT = 8
REWRITE_PROMPT_TEMPLATE = """Rewrite the following user question into clear grammatical English.

Rules:

• Keep the original meaning
• If the question is incomplete, expand it into a full meaningful question
• Do not add new information
• Keep the question under 20 words
• Output ONLY the rewritten question

User Question:
{question}

Rewritten Question:"""

def rewrite_query(question: str, force: bool = False) -> str:
    """
    Rewrites a short user question using Gemini 1.5 Flash to improve grammar
    and expand keywords into full questions for better retrieval.

    Only rewrites if the question has fewer than MAX_WORD_COUNT words, unless `force=True`.
    Falls back to the original question if the API fails, times out, or blocks.

    Args:
        question: The original user question.
        force: If True, bypasses the word count limit (used by the UI button).

    Returns:
        The rewritten question, or the original question.
    """
    if not isinstance(question, str) or not question.strip():
        return question

    original_query = question.strip()
    words = original_query.split()

    if not force and len(words) >= MAX_WORD_COUNT:
        logger.info(f"[QUERY REWRITE] Skipped: length >= {MAX_WORD_COUNT} words.")
        return original_query

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("[QUERY REWRITE] Skipped: GEMINI_API_KEY not configured.")
        return original_query

    try:
        genai.configure(api_key=api_key)
        # Using the recommended fast model for minimal latency
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=importlib_get_generation_config()
        )
        
        prompt = REWRITE_PROMPT_TEMPLATE.format(question=original_query)
        logger.info(f"[QUERY REWRITE] Sending query to Gemini 2.5 Flash: {original_query!r}")
        
        start_time = time.time()
        response = model.generate_content(prompt)
        elapsed = time.time() - start_time
        
        if response.prompt_feedback and getattr(response.prompt_feedback, 'block_reason', None):
            logger.warning(f"[QUERY REWRITE] Blocked by safety filter. Using original. Reason: {response.prompt_feedback.block_reason}")
            return original_query
            
        rewritten_query = response.text.strip() if response.text else ""

        # Robust extraction: Gemini sometimes returns markdown, newlines, or prefixes
        import re
        
        # 1. Remove bold formatting
        rewritten_query = rewritten_query.replace("**", "")
        # 2. Extract the last non-empty line (ignores chatty prefixes on early lines)
        lines = [line.strip() for line in rewritten_query.split("\n") if line.strip()]
        if not lines:
            logger.warning("[QUERY REWRITE] API returned empty lines. Using original.")
            return original_query
            
        rewritten_query = lines[-1]
        
        # 3. Strip common prefixes like "Rewritten Question:" or "Rewrite:"
        # Use a non-capturing group and ensure we only match if there is a colon, or if the string is EXACTLY "Rewritten Question "
        # We use a strict match for the colon `:` to prevent dropping valid first words.
        rewritten_query = re.sub(r'(?i)^(?:rewritten question|question|rewrite|rewritten)\s*:\s*', '', rewritten_query).strip()
        
        # 4. Strip surrounding quotes (Gemini 2.5 Flash likes to output literal quotes around the text)
        rewritten_query = rewritten_query.strip('"\'')

        if not rewritten_query:
            logger.warning("[QUERY REWRITE] Extraction resulted in empty string. Using original.")
            return original_query
            
        # 5. Check true equality (ignore case and punctuation) to detect if model returned the exact same thing
        import string
        def normalize(s):
            return s.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")
            
        if normalize(rewritten_query) == normalize(original_query):
            logger.info("[QUERY REWRITE] Rewritten text is logically identical to original. Returning original.")
            return original_query

        logger.info(f"[QUERY REWRITE] Success ({elapsed:.2f}s). Original: {original_query!r} -> Rewritten: {rewritten_query!r}")
        return rewritten_query

    except Exception as e:
        logger.error(f"[QUERY REWRITE] Gemini API failed: {e}. Falling back to original query.")
        return original_query

def importlib_get_generation_config():
    """Helper to get properly constructed GenerationConfig."""
    import google.generativeai.types as types
    return types.GenerationConfig(
        temperature=0.2,
        # max_output_tokens increased because Gemini 2.5 tokenizes spaces/punctuation heavily
        max_output_tokens=200,
    )
