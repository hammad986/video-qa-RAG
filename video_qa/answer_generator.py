"""
Answer Generation Module for Video-QA System
Generates answers using a cascading LLM pipeline with graceful fallback.

LLM Priority Order:
  1. Local Ollama (fastest, fully local)
  2. Grok API (xAI)
  3. Gemini API
  4. OpenAI API
  5. HuggingFace Serverless
  6. Extractive Fallback (when all LLMs unavailable but context exists)
"""


import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

try:
    from .query_rewriter import rewrite_query
except ImportError:
    def rewrite_query(query: str) -> str:
        return query

# ──────────────────────────────────────────────────────────────
# EXTRACTIVE QA (deepset/roberta-base-squad2) — primary HuggingFace path
# Finds exact answer SPANS in the transcript — fully grounded, no hallucination
# ──────────────────────────────────────────────────────────────
_qa_pipeline = None

def _load_extractive_qa():
    """Load distilbert-base-cased-distilled-squad extractive QA pipeline (cached)."""
    global _qa_pipeline
    if _qa_pipeline is not None:
        return True
    model_name = config.get("answer.hf_model", "distilbert-base-cased-distilled-squad")
    try:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading extractive QA model: {model_name}")
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model=model_name,
            device=-1,   # CPU
        )
        logger.info("Extractive QA model loaded.")
        return True
    except ImportError:
        logger.warning("transformers not installed – extractive QA unavailable.")
        return False
    except Exception as exc:
        logger.warning(f"Failed to load extractive QA model {model_name}: {exc}")
        return False


# ──────────────────────────────────────────────────────────────
# HUGGING FACE SERVERLESS API (InferenceClient)
# Uses the v2 serverless chat API for fast, zero-local-memory QA
# ──────────────────────────────────────────────────────────────
_hf_client = None
_hf_token_used: Optional[str] = None   # tracks which token built the client


def _get_hf_token() -> Optional[str]:
    """
    Read the HF token — config.yaml first, then HF_TOKEN env var.
    Returns None (not '') when no valid token is present.

    Config.get() automatically reloads config.yaml when the file changes on
    disk, so editing hf_token while the app is running is reflected here.
    """
    import os
    _cfg_token = (config.get("answer.hf_token") or "").strip()
    _env_token = (os.environ.get("HF_TOKEN", "") or "").strip()
    token      = _cfg_token or _env_token or None

    # ── MANDATORY DEBUG LOGS (visible in video_qa.log) ─────────────────────
    logger.info(f"HF token loaded from config: {bool(_cfg_token)}")
    logger.info(f"HF token loaded from env:    {bool(_env_token)}")
    logger.info(f"HF token available:          {bool(token)}")
    return token


def _load_hf_client() -> bool:
    """
    Initialize (or re-initialize) the HuggingFace InferenceClient.

    ── FIX ────────────────────────────────────────────────────────────────────
    Previously the client was created once and cached forever — if it was
    built with an empty token (config had hf_token: "") it never recovered.
    Now: always re-read the token; recreate the client only when the token has
    changed (or on first call).  Adding the token to config.yaml and clicking
    "Get Answer" again will immediately use the authenticated client.
    """
    global _hf_client, _hf_token_used

    try:
        from huggingface_hub import InferenceClient

        hf_token = _get_hf_token()

        # Reuse the existing client only when the token is unchanged
        if _hf_client is not None and _hf_token_used == hf_token:
            return True

        if not hf_token:
            logger.warning(
                "No HF_TOKEN found — add your token to config.yaml under "
                "'answer.hf_token' or set the HF_TOKEN env var. "
                "Serverless API requires a token to use gated models."
            )

        _hf_client     = InferenceClient(token=hf_token)
        _hf_token_used = hf_token
        logger.info("HuggingFace InferenceClient (re)initialized (token present: %s).", bool(hf_token))
        return True

    except ImportError:
        logger.warning("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as exc:
        logger.warning(f"Failed to init HF InferenceClient: {exc}")
        return False


def format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss format."""
    if seconds is None:
        return "00:00"
    
    total_seconds = int(seconds)
    mins = total_seconds // 60
    secs = total_seconds % 60
    
    return f"{mins:02d}:{secs:02d}"


def format_time_range(start: float, end: float) -> str:
    """Format timestamp range."""
    return f"[{format_timestamp(start)} - {format_timestamp(end)}]"


def build_answer_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build answer prompt that produces grounded answers.

    The LLM should extract or derive the best possible answer from evidence.
    Only return NOT FOUND IN VIDEO if there is absolutely no relevant information.
    This is used as the PRIMARY pass.
    """
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        start_time = ctx.get("start", 0)
        end_time   = ctx.get("end", 0)
        text       = ctx.get("text", "")
        context_text += f"""
Evidence {i}
Timestamp: {format_time_range(start_time, end_time)}
Content: {text}
---------------------------------
"""

    return f"""You are a lecture assistant that answers questions based on transcript excerpts.

Your task:
* Use ONLY the provided transcript evidence to answer
* Provide the best possible answer based on the context
* Keep answers concise (1-2 sentences)
* Include the timestamp of the most relevant segment
* Only respond "NOT FOUND IN VIDEO" if there is absolutely NO relevant information

Output format MUST be exactly:

Answer: <text>
Timestamp: [mm:ss - mm:ss]

Transcript Segments:
{context_text}

Question:
{query}

Answer:"""


def build_semantic_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build STRICT + SEMANTIC CONTEXT prompt (second-pass / fallback).

    The LLM may derive a short, grounded answer from the evidence — it does
    NOT need the exact phrase to appear verbatim.  It must still stay 100%%
    within the provided transcript context and must not hallucinate.

    This prompt is used ONLY when the strict extractive pass returns
    "NOT FOUND IN VIDEO" but relevant evidence chunks exist.
    """
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        start_time = ctx.get("start", 0)
        end_time   = ctx.get("end", 0)
        text       = ctx.get("text", "")
        context_text += f"""
Evidence {i}
Timestamp: {format_time_range(start_time, end_time)}
Content: {text}
---------------------------------
"""

    return f"""You are a VIDEO QUESTION ANSWERING SYSTEM using STRICT SEMANTIC CONTEXT mode.

IMPORTANT RULES — read carefully:
1. You MUST answer ONLY using information present in the transcript segments below.
2. You may DERIVE a concise answer from the context — exact wording is not required.
3. Do NOT introduce any fact, name, or concept that does not appear in the transcript.
4. Do NOT use outside knowledge or make assumptions beyond what is stated.
5. Keep the answer to ONE or TWO sentences maximum.
6. If the transcript segments contain NO relevant information at all, respond exactly with:
   Answer: NOT FOUND IN VIDEO

Output format MUST be exactly:

Answer: <text>
Timestamp: [mm:ss - mm:ss]   (use the timestamp of the most relevant segment)

Transcript Segments:
{context_text}

Question:
{query}
"""


def ask_local_llm(prompt: str, timeout: Optional[int] = None) -> Optional[str]:
    """Ask local Ollama LLM. Returns None if Ollama is not running."""
    try:
        response = requests.post(
            config.get("answer.ollama_url", "http://localhost:11434/api/generate"),
            json={
                "model": config.get("answer.local_model", "phi3:mini"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 512,
                    "temperature": 0,
                }
            },
            timeout=timeout or config.get("answer.timeout", 180)
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama – it may not be running.")
        return None
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return None


def ask_huggingface_llm(prompt: str, question: str = "", contexts: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Use Hugging Face Serverless Inference API (Chat Completion v2) to answer.
    """
    if not _load_hf_client():
        return None

    try:
        if not contexts:
            return None

        # ── FIX: Curated fallback list — only confirmed chat-compatible models
        # mistralai/Mistral-7B-Instruct-v0.3 removed (flagged as "not a chat model")
        # meta-llama/Llama-3.2-3B-Instruct removed (requires specific provider plan)
        models = [
            "Qwen/Qwen2.5-7B-Instruct",         # confirmed working in logs
            "Qwen/Qwen2.5-72B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "HuggingFaceH4/zephyr-7b-beta",
        ]

        # Configured model or first fallback
        primary_model = config.get("answer.hf_model", models[0])
        if primary_model not in models:
            models.insert(0, primary_model)

        # Build precise QA prompt
        q = question or prompt

        # Build a cleaner context text rather than using the Ollama-specific prompt string
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            text  = ctx.get("text", "")
            start = ctx.get("start", 0)
            end   = ctx.get("end",   0)
            context_text += f"[{start:.1f}s - {end:.1f}s] {text}\n"

        sys_prompt = (
            "You are a helpful video lecture assistant. "
            "Answer the user's question based ONLY on the provided transcript excerpt. "
            "Keep the answer concise (1–2 sentences). "
            "If the information is not in the transcript at all, reply exactly: NOT FOUND IN VIDEO."
        )
        user_msg = f"Transcript Excerpt:\n{context_text}\n\nQuestion: {q}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ]

        for model in models:
            logger.info(f"Trying HF Serverless API with model: {model}")
            try:
                response = _hf_client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1,
                )

                answer = response.choices[0].message.content.strip()
                logger.info(f"HF API Answer ({model}): {answer!r}")

                if answer and "not found in video" not in answer.lower():
                    # Format to match what `parse_answer` expects
                    return f"Answer: {answer}\nTimestamp: None"
                elif "not found in video" in answer.lower():
                    logger.info(f"Model {model}: answer not in context.")
                    return "Answer: Question cannot be answered from this video.\nTimestamp: None"

            except ValueError as ve:
                err = str(ve).lower()
                # ── FIX: 401 / auth error affects ALL models — break immediately
                if any(k in err for k in ("api_key", "hf auth login", "401", "unauthorized", "invalid username")):
                    logger.warning(
                        "HF Serverless API authentication failed (401 Unauthorized). "
                        "Your HF_TOKEN may be expired or invalid. "
                        "Get a new token at https://huggingface.co/settings/tokens "
                        "and update .env or config.yaml."
                    )
                    break   # no point trying remaining models with same broken token
                logger.warning(f"HF API Model {model} failed with ValueError: {ve}")
                continue
            except Exception as e:
                err = str(e).lower()
                # ── FIX: also break on HTTP 401 from generic exceptions
                if "401" in err or "unauthorized" in err or "invalid username" in err:
                    logger.warning(f"HF Serverless: 401 auth failure — skipping remaining models.")
                    break
                logger.warning(f"HF API Model {model} failed: {e}")
                continue

                
        logger.warning("All HF Serverless models failed or refused to answer.")
        return None
        
    except Exception as exc:
        logger.error(f"HF Serverless execution failed: {exc}")
        return None


def ask_gemini_llm(prompt: str, question: str = "", contexts: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Use Google Gemini API as secondary LLM fallback.
    Requires 'gemini_api_key' in config under 'answer.gemini_api_key'.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        return None

    api_key = config.get("answer.gemini_api_key", "")
    if not api_key:
        logger.debug("No Gemini API key configured — skipping Gemini.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            config.get("answer.gemini_model", "gemini-1.5-flash")
        )

        q = question or prompt
        context_text = ""
        if contexts:
            for i, ctx in enumerate(contexts, 1):
                text  = ctx.get("text", "")
                start = ctx.get("start", 0)
                end   = ctx.get("end",   0)
                context_text += f"[{format_timestamp(start)} - {format_timestamp(end)}] {text}\n"

        full_prompt = (
            "You are a VIDEO QUESTION ANSWERING SYSTEM.\n"
            "Rules:\n"
            "* Answer ONLY from the provided transcript excerpt.\n"
            "* Do NOT use outside knowledge.\n"
            "* Keep the answer to one sentence.\n"
            "* If the answer is not in the transcript, reply: Question cannot be answered from this video.\n\n"
            f"Transcript Excerpt:\n{context_text}\n"
            f"Question: {q}\n"
            "Answer:"
        )

        response = model.generate_content(full_prompt)
        answer   = response.text.strip() if response.text else None

        if answer:
            logger.info(f"Gemini answer: {answer!r}")
            return f"Answer: {answer}\nTimestamp: None"

    except Exception as exc:
        logger.warning(f"Gemini API failed: {exc}")

    return None


def ask_openai_llm(prompt: str, timeout: Optional[int] = None) -> Optional[str]:
    """Ask OpenAI API (openai>=1.0.0 compatible)."""
    try:
        import openai

        api_key = config.get("answer.openai_api_key", "")
        if not api_key:
            logger.debug("OpenAI API key not configured — skipping.")
            return None

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=config.get("answer.openai_model", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a video question answering system."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            timeout=timeout or config.get("answer.timeout", 180),
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        logger.debug("openai package not installed — skipping OpenAI provider.")
        return None
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return None


def ask_grok_llm(
    prompt: str,
    question: str = "",
    contexts: Optional[List[Dict]] = None,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Ask xAI Grok API (OpenAI-compatible endpoint).
    Requires 'grok_api_key' in config under 'answer.grok_api_key'.
    Priority: 2nd after Ollama, before Gemini.
    """
    api_key = config.get("answer.grok_api_key", "")
    if not api_key:
        logger.debug("No Grok API key configured — skipping Grok provider.")
        return None

    try:
        import openai  # Grok uses the OpenAI-compatible SDK
    except ImportError:
        logger.debug("openai package not installed — cannot use Grok API.")
        return None

    try:
        grok_url = config.get("answer.grok_url", "https://api.x.ai/v1")
        model    = config.get("answer.grok_model", "grok-beta")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=grok_url,
        )

        q = question or prompt
        context_text = ""
        if contexts:
            for ctx in contexts:
                text  = ctx.get("text",  "")
                start = ctx.get("start", 0)
                end   = ctx.get("end",   0)
                context_text += f"[{format_timestamp(start)} – {format_timestamp(end)}] {text}\n"

        full_prompt = (
            "You are a VIDEO QUESTION ANSWERING SYSTEM.\n"
            "Rules:\n"
            "* Answer ONLY from the provided transcript excerpt.\n"
            "* Do NOT use outside knowledge.\n"
            "* Keep the answer to one or two sentences.\n"
            "* If the answer is not in the transcript, reply: Question cannot be answered from this video.\n\n"
            f"Transcript Excerpt:\n{context_text}\n"
            f"Question: {q}\n"
            "Answer:"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            temperature=0,
            max_tokens=256,
            timeout=timeout or config.get("answer.timeout", 60),
        )

        answer = response.choices[0].message.content.strip()
        if answer:
            logger.info(f"Grok answer: {answer!r}")
            return f"Answer: {answer}\nTimestamp: None"

    except Exception as exc:
        logger.warning(f"Grok API failed: {exc}")

def ask_agentrouter_llm(
    prompt: str,
    question: str = "",
    contexts: Optional[List[Dict]] = None,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Ask AgentRouter API (https://agentrouter.org/v1) using deepseek-v4-flash.
    Priority 1 for answer generation.
    """
    import os
    api_key = os.environ.get("AGENTROUTER_API_KEY") or config.get("answer.agentrouter_api_key", "sk-opqFzZdgHqOkLnwar29MJhPaUy9cIVKbT8dv3f36tidBMwxP")
    if not api_key:
        logger.debug("No AgentRouter API key configured — skipping.")
        return None

    base_url = os.environ.get("AGENTROUTER_BASE_URL") or config.get("answer.agentrouter_url", "https://agentrouter.org/v1")
    model = os.environ.get("AGENTROUTER_MODEL") or config.get("answer.agentrouter_model", "deepseek-v4-flash")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "claude-cli/1.0.0 (external, cli)",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout or config.get("answer.timeout", 45))
        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            logger.info(f"AgentRouter answer ({model}): {answer[:120]}...")
            if "answer:" not in answer.lower():
                answer = f"Answer: {answer}\nTimestamp: None"
            return answer
        else:
            logger.warning(f"AgentRouter API returned status {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as exc:
        logger.warning(f"AgentRouter API failed: {exc}")
        return None


def parse_answer(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse LLM response to extract answer and timestamp.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Tuple of (answer, timestamp)
    """
    answer = None
    timestamp = None
    
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        
        if line.lower().startswith("timestamp:"):
            timestamp = line.split(":", 1)[1].strip()
    
    return answer, timestamp


def _is_not_found(answer: Optional[str]) -> bool:
    """Return True if the answer string EXPLICITLY signals 'not found'."""
    if not answer:
        return True
    low = answer.lower().strip()
    # Only match EXACT phrases, not substrings
    # This prevents rejection of valid answers that happen to contain "not" in other contexts
    return (
        low == "not found in video"
        or low == "question cannot be answered from this video"
        or low.startswith("not found in video.")
    )


def _extractive_fallback(
    query: str,
    contexts: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Last-resort grounded answer when ALL LLMs are unavailable.

    Extracts the first 2 sentences from the top-ranked context chunk and
    returns them as a grounded answer with the chunk's timestamp.
    This ensures the user ALWAYS gets a response when evidence exists,
    even when no API key / local model is configured.

    Only called from _run_llm_cascade() after all providers have failed.
    """
    if not contexts:
        return None

    import re

    top_ctx = contexts[0]
    text    = (top_ctx.get("text") or "").strip()
    start   = top_ctx.get("start", 0)
    end     = top_ctx.get("end",   0)

    if not text:
        return None

    # Grab up to 2 sentences as the extracted answer
    sentences = re.split(r'(?<=[.!?])\s+', text)
    excerpt   = " ".join(sentences[:2]).strip()

    if not excerpt:
        return None

    ts = format_time_range(start, end)
    logger.warning(
        "[EXTRACTIVE FALLBACK] All LLM providers failed — "
        "returning extractive answer from top context chunk. "
        "Configure a valid HF_TOKEN or local Ollama model for LLM-based answers."
    )
    logger.info(f"[EXTRACTIVE FALLBACK] Excerpt: {excerpt[:100]}...")
    return f"Answer: {excerpt}\nTimestamp: {ts}"


def _run_llm_cascade(
    prompt: str,
    query: str,
    contexts: List[Dict[str, Any]],
    answer_config: Dict[str, Any],
    use_extractive_fallback: bool = True,
) -> Optional[str]:
    """
    Run the LLM priority cascade:
      1. Ollama (local)
      2. Grok API
      3. Gemini API
      4. OpenAI API
      5. HuggingFace Serverless
      6. Extractive fallback (if contexts available and use_extractive_fallback=True)

    Returns the raw response string, or None if all fail and no fallback.
    """
    response = None

    # 1. AgentRouter (Primary: deepseek-v4-flash via agentrouter.org)
    logger.info("[LLM] Trying AgentRouter (deepseek-v4-flash)…")
    response = ask_agentrouter_llm(prompt, question=query, contexts=contexts)
    if response:
        logger.info("[LLM] AgentRouter answered.")

    # 2. Local Ollama
    if response is None and answer_config.get("use_local", True):
        logger.info("[LLM] Local Ollama check…")
        response = ask_local_llm(prompt)
        if response:
            logger.info("[LLM] Local LLM answered.")

    # 2. Grok API
    if response is None:
        logger.info("[LLM] Local unavailable — trying Grok…")
        response = ask_grok_llm(prompt, question=query, contexts=contexts)
        if response:
            logger.info("[LLM] Grok answered.")

    # 3. Gemini API
    if response is None:
        logger.info("[LLM] Grok unavailable — trying Gemini…")
        response = ask_gemini_llm(prompt, question=query, contexts=contexts)
        if response:
            logger.info("[LLM] Gemini answered.")

    # 4. OpenAI API
    if response is None and answer_config.get("use_openai", False):
        logger.info("[LLM] Gemini unavailable — trying OpenAI…")
        response = ask_openai_llm(prompt)
        if response:
            logger.info("[LLM] OpenAI answered.")

    # 5. HuggingFace Serverless
    if response is None and answer_config.get("use_huggingface", True):
        logger.info("[LLM] All primary LLMs unavailable — using HuggingFace Serverless…")
        response = ask_huggingface_llm(prompt, question=query, contexts=contexts)
        if response:
            logger.info("[LLM] HuggingFace Serverless answered.")

    # 6. Extractive fallback — CRITICAL FIX
    # When ALL LLMs fail but valid context chunks exist, synthesise a grounded
    # answer from the top chunk instead of returning "NOT FOUND IN VIDEO".
    if response is None and use_extractive_fallback and contexts:
        logger.info("[LLM] All providers failed — attempting extractive fallback…")
        response = _extractive_fallback(query, contexts)
        if response:
            logger.info("[LLM] Extractive fallback produced an answer.")

    if response is None:
        logger.warning("[LLM] All providers AND extractive fallback failed — no answer possible.")

    return response


def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    use_verification: bool = False
) -> Dict[str, Any]:
    """
    Generate answer from query and retrieved contexts.

    Two-pass strategy
    ─────────────────
    PASS 1 — STRICT EXTRACTIVE:
      Ask the LLM to extract an exact phrase.  If it finds one, done.

    PASS 2 — STRICT + SEMANTIC CONTEXT (fallback):
      Only runs if Pass 1 returned NOT FOUND but evidence chunks exist.
      The LLM may *derive* a grounded answer from the context — it still
      cannot introduce outside knowledge.  This eliminates false NOT FOUND
      rejections caused by paraphrase distance between question and chunk.

    Strict mode is NOT disabled.  Both passes enforce evidence-only rules.
    """
    logger.info("="*50)
    logger.info("ANSWER GENERATION")
    logger.info("="*50)

    if not contexts:
        return {
            "answer":    "Question cannot be answered from this video.",
            "timestamp": None,
            "contexts":  [],
            "verified":  False
        }

    answer_config = config.get_section("answer")

    # ── FORENSIC DEBUG: Stage 2 - Context Passing ─────────────────────────────
    combined_context_length = sum(len(ctx.get('text', '')) for ctx in contexts)
    logger.info(f"\n[CONTEXT DEBUG]")
    logger.info(f"  contexts_passed: {len(contexts)}")
    logger.info(f"  combined_context_length: {combined_context_length} characters")
    for i, ctx in enumerate(contexts, 1):
        logger.info(
            f"    Context[{i}]: {ctx.get('start', 0):.1f}s-{ctx.get('end', 0):.1f}s | "
            f"Length: {len(ctx.get('text', ''))} chars | "
            f"Score: {ctx.get('score', 0):.4f}"
        )
    # ──────────────────────────────────────────────────────────────────────────

    # FIX: Query rewriting integration (expand for advanced use)
    rewritten_query = rewrite_query(query)

    # ── PASS 1: Strict extractive ────────────────────────────────────────────
    logger.info("[PASS 1] Strict extractive mode…")
    strict_prompt = build_answer_prompt(rewritten_query, contexts)
    
    # ── FORENSIC DEBUG: Stage 2B - Prompt Being Sent ─────────────────────────
    logger.info(f"\n[PROMPT DEBUG]")
    logger.info(f"  prompt_length: {len(strict_prompt)} characters")
    logger.info(f"  prompt_preview: {strict_prompt[:300]}...")
    # ──────────────────────────────────────────────────────────────────────────
    
    response = _run_llm_cascade(strict_prompt, rewritten_query, contexts, answer_config)


    answer, timestamp = parse_answer(response) if response else (None, None)
    
    # ── FORENSIC DEBUG: Stage 3 - LLM Response ────────────────────────────
    logger.info(f"\n[LLM RESPONSE DEBUG - PASS 1]")
    logger.info(f"  raw_response_length: {len(response) if response else 0} characters")
    if response:
        logger.info(f"  raw_response (first 300 chars): {response[:300]}...")
    else:
        logger.info(f"  raw_response: None (LLM failed or timeout)")
    logger.info(f"  parsed_answer: {answer[:150] if answer else 'None'}...")
    logger.info(f"  parsed_timestamp: {timestamp}")
    logger.info(f"  _is_not_found(answer): {_is_not_found(answer)}")
    # ──────────────────────────────────────────────────────────────────────────

    # Treat empty, None, or 'None' (string) as missing timestamp
    def _is_missing_timestamp(ts):
        return not ts or str(ts).strip().lower() == "none"

    # ── PASS 2: Semantic context fallback ────────────────────────────────────────────
    # Trigger if: strict returned NOT FOUND (or no response at all) AND
    # we have real evidence chunks to ground the answer in.
    # FIX: Pass use_extractive_fallback=False to Pass 2 — the extractive fallback
    # is only triggered ONCE at the end of Pass 2 (not redundantly in both passes).
    if _is_not_found(answer) and contexts:
        logger.info(
            "[PASS 2] Strict pass returned NOT FOUND — trying SEMANTIC context mode."
        )

        # ── FORENSIC DEBUG: Stage 4 - Pass 2 Trigger ───────────────────────────────
        logger.info(f"\n[PASS DEBUG]")
        logger.info(f"  pass_1_result: NOT_FOUND")
        logger.info(f"  pass_1_answer: {answer}")
        logger.info(f"  pass_2_triggered: True")
        logger.info(f"  reason_for_pass_2: strict returned NOT_FOUND AND {len(contexts)} contexts exist")
        # ──────────────────────────────────────────────────────────────────────────
        semantic_prompt = build_semantic_prompt(query, contexts)
        # Do NOT use extractive fallback here — reserve it for the final check below
        sem_response    = _run_llm_cascade(
            semantic_prompt, query, contexts, answer_config,
            use_extractive_fallback=False,
        )
        sem_answer, sem_ts = parse_answer(sem_response) if sem_response else (None, None)

        # ── FORENSIC DEBUG: Pass 2 Response ──────────────────────────────
        logger.info(f"\n[LLM RESPONSE DEBUG - PASS 2]")
        logger.info(f"  raw_response_length: {len(sem_response) if sem_response else 0} characters")
        if sem_response:
            logger.info(f"  raw_response (first 300 chars): {sem_response[:300]}...")
        else:
            logger.info(f"  raw_response: None")
        logger.info(f"  parsed_answer: {sem_answer[:150] if sem_answer else 'None'}...")
        logger.info(f"  _is_not_found(sem_answer): {_is_not_found(sem_answer)}")
        # ──────────────────────────────────────────────────────────────────────────

        if sem_answer and not _is_not_found(sem_answer):
            logger.info("[PASS 2] Semantic context answer ACCEPTED.")
            logger.info(f"  using semantic response instead of strict pass rejection")
            answer    = sem_answer
            timestamp = sem_ts
            response  = sem_response
        else:
            logger.info("[PASS 2] Semantic pass also returned NOT FOUND.")
            # ── EXTRACTIVE FALLBACK — final safety net ────────────────────────
            # Both LLM passes failed. If contexts exist, use extractive fallback
            # so the user always gets a grounded response.
            if contexts:
                fb_response = _extractive_fallback(query, contexts)
                if fb_response:
                    fb_answer, fb_ts = parse_answer(fb_response)
                    if fb_answer and not _is_not_found(fb_answer):
                        logger.info("[EXTRACTIVE FALLBACK] Using extracted answer from top context.")
                        answer    = fb_answer
                        timestamp = fb_ts
                        response  = fb_response

    # ── Final answer normalisation ───────────────────────────────────────────
    if _is_not_found(answer):
        answer    = "NOT FOUND IN VIDEO"
        timestamp = None

    # Always set timestamp: use LLM-provided if available, else fallback to top evidence chunk
    if _is_missing_timestamp(timestamp) and contexts:
        # Use the first (top) context's start/end as the timestamp
        top_ctx = contexts[0]
        start = top_ctx.get("start", 0)
        end = top_ctx.get("end", 0)
        timestamp = format_time_range(start, end)

    result = {
        "answer":       answer,
        "timestamp":    timestamp,
        "contexts":     contexts,
        "raw_response": response,
        "verified":     False,
    }
    
    # ── FORENSIC DEBUG: Stage 6 - Final Output (before verification) ──────────────────────────
    logger.info(f"\n[FINAL OUTPUT DEBUG - BEFORE VERIFICATION]")  
    logger.info(f"  final_answer: {result['answer'][:150]}...")
    logger.info(f"  final_timestamp: {result['timestamp']}")
    logger.info(f"  num_contexts_returned: {len(result['contexts'])}")
    logger.info(f"  verified: {result['verified']}")
    # ──────────────────────────────────────────────────────────────────────────

    # ── Evidence Verification ────────────────────────────────────────────────
    if use_verification and config.get("verification.enabled", True):
        from .evidence_verifier import verify_answer as _verify
        
        # ── FORENSIC DEBUG: Stage 5 - Before Verification ────────────────────────
        logger.info(f"\n[VERIFICATION DEBUG - BEFORE]")
        logger.info(f"  answer_before_verification: {answer[:150]}...")
        logger.info(f"  timestamp_before_verification: {timestamp}")
        logger.info(f"  contexts_for_verification: {len(contexts)}")
        logger.info("  running verification...")
        # ──────────────────────────────────────────────────────────────────────────
        
        logger.info("Running evidence verification…")
        verification_result = _verify(answer, contexts, query)
        is_valid            = verification_result.get("is_valid", False)
        status              = verification_result.get("status", "UNKNOWN")
        trust_score         = verification_result.get("trust_score", 0)
        confidence          = verification_result.get("confidence", 0.0)
        
        # ── FORENSIC DEBUG: Stage 5 - Verification Result ────────────────────────────
        logger.info(f"\n[VERIFICATION DEBUG - AFTER]")
        logger.info(f"  status: {status}")
        logger.info(f"  is_valid: {is_valid}")
        logger.info(f"  trust_score: {trust_score}")
        logger.info(f"  confidence: {confidence:.2%}")
        # ──────────────────────────────────────────────────────────────────────────
        
        result["verification"] = verification_result
        result["trust_score"]  = trust_score

        if not is_valid:
            logger.warning(f"[VERIFICATION] Answer REJECTED by verification (status={status})")
            logger.warning(f"  original_answer: {answer[:100]}...")
            if verification_result.get("status") == "HALLUCINATION":
                logger.warning("  reason: HALLUCINATION - answer contradicts evidence")
                result["answer"]    = "Question cannot be answered from this video."
                result["timestamp"] = None
            else:
                logger.warning(f"  reason: is_valid=False (status={status})")
            result["verified"] = False
        else:
            logger.info(f"[VERIFICATION] Answer ACCEPTED by verification (status={status})")
            result["verified"] = True
        
        # ── FORENSIC DEBUG: Stage 5 - After Modification ───────────────────────────────
        logger.info(f"\n[ANSWER MODIFICATION DEBUG]")
        logger.info(f"  answer_after_verification: {result['answer'][:150]}...")
        logger.info(f"  timestamp_after_verification: {result['timestamp']}")
        logger.info(f"  verified_flag: {result['verified']}")
        # ──────────────────────────────────────────────────────────────────────────
    
    # ── FORENSIC DEBUG: Stage 6 - FINAL RETURN ────────────────────────────────────────────────────────
    logger.info(f"\n[FINAL OUTPUT - RETURNING TO UI]")
    logger.info(f"  FINAL_ANSWER: {result['answer'][:200]}")
    logger.info(f"  FINAL_TIMESTAMP: {result['timestamp']}")
    logger.info(f"  VERIFIED: {result['verified']}")
    logger.info(f"  TRUST_SCORE: {result.get('trust_score', 'N/A')}")
    logger.info(f"  NUM_CONTEXTS: {len(result['contexts'])}")
    verification_status = result.get('verification', {}).get('status', 'N/A')
    logger.info(f"  VERIFICATION_STATUS: {verification_status}")
    logger.info(f"\n" + "="*80)
    # ───────────────────────────────────────────────────────────────────────────────
    
    return result


class AnswerGenerator:
    """Answer generation wrapper."""
    
    def __init__(self):
        """Initialize answer generator."""
        self.use_verification = config.get("verification.enabled", True)
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer for query using contexts."""
        return generate_answer(query, contexts, self.use_verification)


if __name__ == "__main__":
    # Test answer generation
    test_contexts = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "start": 0,
            "end": 10
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to learn representations of data.",
            "start": 10,
            "end": 20
        }
    ]
    
    query = "What is machine learning?"
    
    result = generate_answer(query, test_contexts)
    
    print("\nAnswer Result:")
    print(f"Answer: {result['answer']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Verified: {result['verified']}")
