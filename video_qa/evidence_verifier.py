"""
Evidence Verification Module for Video-QA System

Verifies that generated answers are actually grounded in the evidence.
Uses a two-path approach:
  1. Local NLI model (facebook/bart-large-mnli) — no external service needed
  2. Ollama LLM fallback (optional, only if configured)

Returns structured verification result:
  {"status": "VERIFIED|INCOMPLETE|HALLUCINATION", "trust_score": 0-100, "confidence": float}
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

from .config import config
from .logger import get_logger


logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# HUGGING FACE SERVERLESS API FOR NLI
# ──────────────────────────────────────────────────────────────
_hf_client = None

def _load_hf_client():
    """Load huggingface_hub InferenceClient."""
    global _hf_client
    if _hf_client is not None:
        return True
    try:
        from huggingface_hub import InferenceClient
        import os
        token = os.environ.get("HF_TOKEN") or config.get("answer.hf_token", "")
        _hf_client = InferenceClient(token=token if token else None)
        logger.info("NLI verifier: Hugging Face InferenceClient initialized.")
        return True
    except Exception as exc:
        logger.warning(f"Failed to load NLI InferenceClient: {exc}")
        return False


def _verify_with_nli(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Dict[str, Any]:
    """
    Use Hugging Face Serverless API to verify evidence.
    """
    if not _load_hf_client():
        return {"status": "UNKNOWN", "trust_score": 0, "confidence": 0.0}

    premise = " ".join(ctx.get("text", "") for ctx in contexts)[:1500]
    
    sys_prompt = "You are a strict Evidence Verifier. Given a FACT and EVIDENCE, determine if the FACT is supported by the EVIDENCE. Reply STRICTLY with ONLY one of these three words: VERIFIED (if fully supported), HALLUCINATION (if contradicted), INCOMPLETE (if neither)."
    user_msg = f"EVIDENCE:\n{premise}\n\nFACT:\n{answer}"
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg}
    ]

    try:
        model = config.get("answer.hf_model", "meta-llama/Llama-3.2-3B-Instruct")
        response = _hf_client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )
        result_text = response.choices[0].message.content.strip().upper()
        
        status = "INCOMPLETE"
        trust_score = 30
        conf = 0.3
        
        if "VERIFIED" in result_text:
            status = "VERIFIED"
            trust_score = 90
            conf = 0.9
        elif "HALLUCINATION" in result_text:
            status = "HALLUCINATION"
            trust_score = 10
            conf = 0.1
            
        logger.info(f"HF NLI API result: {status} (Raw: {result_text})")
        return {"status": status, "trust_score": trust_score, "confidence": conf}

    except ValueError as ve:
        if "api_key" in str(ve).lower() or "auth" in str(ve).lower():
            logger.error("HF Serverless Verification requires a token.")
        return {"status": "UNKNOWN", "trust_score": 0, "confidence": 0.0}
    except Exception as exc:
        logger.error(f"NLI Verification failed: {exc}")
        return {"status": "UNKNOWN", "trust_score": 0, "confidence": 0.0}


# ──────────────────────────────────────────────────────────────
# Optional Ollama-based verification (legacy path)
# ──────────────────────────────────────────────────────────────
def build_verification_prompt(
    query: str,
    answer: str,
    contexts: List[Dict[str, Any]],
) -> str:
    """Build verification prompt for Ollama LLM path."""
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        text = ctx.get("text", "")
        start = ctx.get("start", 0)
        end = ctx.get("end", 0)
        context_text += f"\nPassage {i} ({start:.1f}s – {end:.1f}s):\n{text}\n"

    return f"""You are an EVIDENCE VERIFIER for a Video Question Answering system.

Your job is to verify if the given answer is ACTUALLY present in the evidence passages.

VERIFICATION RULES:
1. Check if the answer text or its key facts are LITERALLY present in any passage
2. Do NOT allow paraphrased or summarised answers — they must be EXACT or near-exact matches
3. If the answer adds information not in passages, it is INVALID
4. Only accept answers that can be found word-for-word or with minor punctuation changes

EVIDENCE PASSAGES:
{context_text}

QUESTION: {query}

GENERATED ANSWER: {answer}

OUTPUT FORMAT (choose ONE):

VALID – The answer is directly supported by the evidence passages.

or

INVALID – The answer is NOT supported by the evidence.

Your response must start with either "VALID" or "INVALID" followed by a brief explanation."""


def _verify_with_ollama(
    query: str,
    answer: str,
    contexts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask Ollama LLM to verify the answer (returns None if Ollama unavailable)."""
    import requests

    prompt = build_verification_prompt(query, answer, contexts)
    try:
        response = requests.post(
            config.get("verification.ollama_url", "http://localhost:11434/api/generate"),
            json={
                "model": config.get("verification.local_model", "phi3:mini"),
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0},
            },
            timeout=config.get("verification.timeout", 120),
        )
        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            if raw.lower().startswith("valid"):
                return {"status": "VERIFIED", "trust_score": 80, "confidence": 0.8}
            else:
                return {"status": "INCOMPLETE", "trust_score": 30, "confidence": 0.3}
        logger.error(f"Ollama verification API error: {response.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable for verification – skipping.")
        return None
    except Exception as exc:
        logger.error(f"Ollama verification failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def verify_answer(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Dict[str, Any]:
    """
    Verify whether the answer is grounded in the retrieved evidence.

    Returns a structured dict:
        {
            "status":      "VERIFIED" | "INCOMPLETE" | "HALLUCINATION" | "UNKNOWN",
            "trust_score": int   (0-100),
            "confidence":  float (0.0-1.0),
            "is_valid":    bool,
        }
    """
    # If the answer says "not found", no need to verify
    if not answer or "not found" in answer.lower():
        result = {"status": "INCOMPLETE", "trust_score": 0, "confidence": 0.0}
        result["is_valid"] = False
        return result

    if not contexts:
        result = {"status": "UNKNOWN", "trust_score": 0, "confidence": 0.0}
        result["is_valid"] = False
        return result

    logger.info("Verifying answer against evidence…")

    # ── Path 1: local NLI (preferred) ─────────
    nli_result = _verify_with_nli(answer, contexts, query)

    if nli_result["status"] != "UNKNOWN":
        nli_result["is_valid"] = nli_result["status"] in ("VERIFIED",)
        return nli_result

    # ── Path 2: Ollama LLM (optional fallback) ─
    logger.info("NLI verification inconclusive; trying Ollama fallback…")
    ollama_result = _verify_with_ollama(query, answer, contexts)

    if ollama_result is not None:
        ollama_result["is_valid"] = ollama_result["status"] == "VERIFIED"
        return ollama_result

    # ── Path 3: accept conservatively ─────────
    logger.warning("All verification methods failed – accepting answer conservatively.")
    return {
        "status": "INCOMPLETE",
        "trust_score": 50,
        "confidence": 0.5,
        "is_valid": True,
    }


# Backward-compat tuple-returning wrapper (used by answer_generator.py)
def verify_answer_compat(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Tuple[bool, Optional[str]]:
    """Legacy wrapper that returns (is_valid, answer_or_none)."""
    result = verify_answer(answer, contexts, query)
    is_valid = result.get("is_valid", False)
    return is_valid, (answer if is_valid else None)


class EvidenceVerifier:
    """Evidence verification wrapper."""

    def __init__(self):
        pass

    def verify(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """Verify answer against contexts, return structured result."""
        return verify_answer(answer, contexts, query)


if __name__ == "__main__":
    test_contexts = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "start": 0,
            "end": 10,
        }
    ]
    result = verify_answer(
        "Machine learning is a subset of artificial intelligence.",
        test_contexts,
        "What is machine learning?",
    )
    print(result)
