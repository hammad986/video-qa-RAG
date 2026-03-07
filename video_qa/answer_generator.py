"""
Answer Generation Module for Video-QA System
Generates answers using local LLM (Ollama), HuggingFace flan-t5-base, or OpenAI.
Uses strict extractive QA prompt to prevent hallucination.
"""

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from .config import config
from .logger import get_logger


logger = get_logger(__name__)

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
    Build strict extractive QA prompt.

    The LLM must extract an exact phrase from the evidence.
    If the answer is literally absent, it must say NOT FOUND IN VIDEO.
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

    return f"""You are a VIDEO QUESTION ANSWERING SYSTEM.

Rules:
* Only use the provided evidence passages
* Do not use outside knowledge
* Do not summarize
* Extract the answer directly
* One sentence only
* Must include timestamp
* If answer absent output: NOT FOUND IN VIDEO

Output format MUST be exactly:

Answer: <text>
Timestamp: [mm:ss - mm:ss]

Transcript Segments:
{context_text}

Question:
{query}
"""


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

        # Fallback list of fast, free serverless chat models
        # Provided by huggingface_hub serverless API
        models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-7B-Instruct"
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
            text = ctx.get("text", "")
            start = ctx.get("start", 0)
            end = ctx.get("end", 0)
            context_text += f"[{start:.1f}s - {end:.1f}s] {text}\n"

        sys_prompt = "You are a helpful assistant. Answer the user's question based ONLY on the provided transcript excerpt. Keep the answer concise. If the exact answer is not in the transcript, reply strictly with: NOT FOUND IN VIDEO."
        user_msg = f"Transcript Excerpt:\n{context_text}\n\nQuestion: {q}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg}
        ]

        for model in models:
            logger.info(f"Trying HF Serverless API with model: {model}")
            try:
                response = _hf_client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"HF API Answer ({model}): {answer!r}")
                
                if answer and "not found in video" not in answer.lower():
                    # Format to match what `parse_answer` expects
                    return f"Answer: {answer}\nTimestamp: None"
                elif "not found in video" in answer.lower():
                    logger.info(f"Model {model} correctly identified answer not in context.")
                    return "Answer: Question cannot be answered from this video.\nTimestamp: None"
                
            except ValueError as ve:
                err = str(ve).lower()
                if "api_key" in err or "hf auth login" in err:
                    # ── TOKEN ERROR FIX ──────────────────────────────────────
                    # Previously this returned the error message as a valid
                    # answer string — it got stored as a summary bullet point.
                    # Now: log the problem clearly and return None so the
                    # caller falls through to the next LLM in priority order.
                    logger.warning(
                        "HF Serverless API requires a token. "
                        "Add 'hf_token' under the 'answer' section in config.yaml "
                        "or set the HF_TOKEN environment variable."
                    )
                    break   # no point trying remaining models with same missing token
                logger.warning(f"HF API Model {model} failed with ValueError: {ve}")
                continue
            except Exception as e:
                logger.warning(f"HF API Model {model} failed: {e}")
                continue  # fall back to next model

                
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
            logger.error("OpenAI API key not configured")
            return None

        # ── OPENAI v1.x FIX ──────────────────────────────────────────────────
        # openai.ChatCompletion.create was removed in openai>=1.0.0.
        # Use the new client-based API instead.
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=config.get("answer.openai_model", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a video question answering system."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            timeout=timeout or config.get("answer.timeout", 180),
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        logger.error("openai package not installed — run: pip install openai")
        return None
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
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
    """Return True if the answer string signals 'not found'."""
    if not answer:
        return True
    low = answer.lower()
    return (
        "not found" in low
        or "cannot be answered" in low
        or "not in video" in low
    )


def _run_llm_cascade(
    prompt: str,
    query: str,
    contexts: List[Dict[str, Any]],
    answer_config: Dict[str, Any],
) -> Optional[str]:
    """
    Run the standard LLM priority cascade:
      Ollama → Gemini → OpenAI → HuggingFace

    Returns the raw response string, or None if all fail.
    """
    response = None

    if answer_config.get("use_local", True):
        logger.info("[LLM] Trying LOCAL (Ollama)…")
        response = ask_local_llm(prompt)
        if response:
            logger.info("[LLM] Local LLM answered.")

    if response is None:
        logger.info("[LLM] Local unavailable — trying Gemini…")
        response = ask_gemini_llm(prompt, question=query, contexts=contexts)
        if response:
            logger.info("[LLM] Gemini answered.")

    if response is None and answer_config.get("use_openai", False):
        logger.info("[LLM] Gemini unavailable — trying OpenAI…")
        response = ask_openai_llm(prompt)
        if response:
            logger.info("[LLM] OpenAI answered.")

    if response is None and answer_config.get("use_huggingface", True):
        logger.info("[LLM] All primary LLMs unavailable — using HuggingFace Serverless…")
        response = ask_huggingface_llm(prompt, question=query, contexts=contexts)
        if response:
            logger.info("[LLM] HuggingFace Serverless answered.")

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

    # ── PASS 1: Strict extractive ────────────────────────────────────────────
    logger.info("[PASS 1] Strict extractive mode…")
    strict_prompt = build_answer_prompt(query, contexts)
    response = _run_llm_cascade(strict_prompt, query, contexts, answer_config)


    answer, timestamp = parse_answer(response) if response else (None, None)

    # Treat empty, None, or 'None' (string) as missing timestamp
    def _is_missing_timestamp(ts):
        return not ts or str(ts).strip().lower() == "none"

    # ── PASS 2: Semantic context fallback ────────────────────────────────────
    # Trigger if: strict returned NOT FOUND (or no response at all) AND
    # we have real evidence chunks to ground the answer in.
    if _is_not_found(answer) and contexts:
        logger.info(
            "[PASS 2] Strict pass returned NOT FOUND — trying SEMANTIC context mode."
        )
        semantic_prompt  = build_semantic_prompt(query, contexts)
        sem_response     = _run_llm_cascade(semantic_prompt, query, contexts, answer_config)
        sem_answer, sem_ts = parse_answer(sem_response) if sem_response else (None, None)

        if sem_answer and not _is_not_found(sem_answer):
            logger.info("[PASS 2] Semantic context answer accepted.")
            answer    = sem_answer
            timestamp = sem_ts
            response  = sem_response
        else:
            logger.info("[PASS 2] Semantic pass also returned NOT FOUND — evidence insufficient.")

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

    # ── Evidence Verification ────────────────────────────────────────────────
    if use_verification and config.get("verification.enabled", True):
        from .evidence_verifier import verify_answer as _verify
        logger.info("Running evidence verification…")

        verification_result = _verify(answer, contexts, query)
        is_valid            = verification_result.get("is_valid", False)

        result["verification"] = verification_result
        result["trust_score"]  = verification_result.get("trust_score", 0)

        if not is_valid:
            if verification_result.get("status") == "HALLUCINATION":
                result["answer"]    = "Question cannot be answered from this video."
                result["timestamp"] = None
            result["verified"] = False
        else:
            result["verified"] = True

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
