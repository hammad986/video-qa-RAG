import logging
import time
from typing import Optional
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

    # ── Priority 1: AgentRouter (deepseek-v4-flash) ──────────────────
    agentrouter_key = os.getenv("AGENTROUTER_API_KEY") or config.get("answer.agentrouter_api_key", "sk-opqFzZdgHqOkLnwar29MJhPaUy9cIVKbT8dv3f36tidBMwxP")
    agentrouter_url = os.getenv("AGENTROUTER_BASE_URL") or config.get("answer.agentrouter_url", "https://agentrouter.org/v1")
    agentrouter_model = os.getenv("AGENTROUTER_MODEL") or config.get("answer.agentrouter_model", "deepseek-v4-flash")

    if agentrouter_key:
        try:
            import requests
            url = f"{agentrouter_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {agentrouter_key}",
                "Content-Type": "application/json",
                "User-Agent": "claude-cli/1.0.0 (external, cli)",
            }
            prompt = REWRITE_PROMPT_TEMPLATE.format(question=original_query)
            payload = {
                "model": agentrouter_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 150,
            }
            logger.info(f"[QUERY REWRITE] Sending query to AgentRouter ({agentrouter_model}): {original_query!r}")
            start_time = time.time()
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            elapsed = time.time() - start_time

            if resp.status_code == 200:
                raw_rewritten = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if raw_rewritten:
                    clean_rewritten = _clean_rewritten_query(raw_rewritten, original_query)
                    if clean_rewritten:
                        logger.info(f"[QUERY REWRITE] AgentRouter Success ({elapsed:.2f}s). Original: {original_query!r} -> Rewritten: {clean_rewritten!r}")
                        return clean_rewritten
            else:
                logger.warning(f"[QUERY REWRITE] AgentRouter returned {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            logger.warning(f"[QUERY REWRITE] AgentRouter failed: {e}. Falling back...")

    # ── Priority 2: Gemini API Fallback ──────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY", "") or config.get("answer.gemini_api_key", "")
    if not api_key:
        logger.warning("[QUERY REWRITE] Skipped: Neither AgentRouter nor GEMINI_API_KEY succeeded.")
        return original_query

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
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
        clean_rewritten = _clean_rewritten_query(rewritten_query, original_query)
        if clean_rewritten:
            logger.info(f"[QUERY REWRITE] Gemini Success ({elapsed:.2f}s). Original: {original_query!r} -> Rewritten: {clean_rewritten!r}")
            return clean_rewritten
        return original_query

    except Exception as e:
        logger.error(f"[QUERY REWRITE] Gemini API failed: {e}. Falling back to original query.")
        return original_query

def _clean_rewritten_query(rewritten_query: str, original_query: str) -> Optional[str]:
    """Helper to sanitize and normalize model rewritten query text."""
    import re
    import string

    # 1. Remove bold formatting
    rewritten_query = rewritten_query.replace("**", "")
    # 2. Extract last non-empty line
    lines = [line.strip() for line in rewritten_query.split("\n") if line.strip()]
    if not lines:
        return None
    rewritten_query = lines[-1]
    # 3. Strip common prefixes
    rewritten_query = re.sub(r'(?i)^(?:rewritten question|question|rewrite|rewritten)\s*:\s*', '', rewritten_query).strip()
    # 4. Strip quotes
    rewritten_query = rewritten_query.strip('"\'')
    if not rewritten_query:
        return None

    def normalize(s):
        return s.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")

    if normalize(rewritten_query) == normalize(original_query):
        return original_query

    return rewritten_query

def importlib_get_generation_config():
    """Helper to get properly constructed GenerationConfig."""
    try:
        import google.generativeai.types as types
        return types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=200,
        )
    except Exception:
        return None

