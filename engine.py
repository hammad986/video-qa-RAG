"""
Inference Engine - Persistent Model Loading
Handles FLAN-T5, embeddings, and BART-MNLI verification
"""

import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Persistent model inference engine."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize models (loaded once)
        self._load_models()

    def _load_models(self):
        """Load all heavy models once."""
        logger.info("Loading FLAN-T5 model...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(self.device)

        logger.info("Loading BART-MNLI verifier...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(self.device)

        logger.info("Loading embedding model...")
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        logger.info("All models loaded successfully!")

    def get_status(self) -> Dict[str, bool]:
        """Get model loading status."""
        return {
            "FLAN-T5": hasattr(self, 't5_model'),
            "BART-MNLI": hasattr(self, 'nli_model'),
            "Embeddings": hasattr(self, 'embed_model')
        }

    def generate_answer(self, question: str, evidence: List[Dict]) -> Dict[str, Any]:
        """Generate answer using FLAN-T5 and provided evidence."""

        if not evidence:
            return {"answer": "No evidence available to answer this question."}

        # Format evidence for the model
        context = "\n".join([doc.page_content for doc in evidence[:3]])  # Use top 3

        prompt = f"""
Answer the question based only on the provided context. If the answer is not in the context, say "I cannot answer this question based on the available information."

Context:
{context}

Question: {question}

Answer:"""

        try:
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7
                )

            answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            return {"answer": answer}

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {"answer": "Error generating answer."}

    def verify_answer(self, question: str, answer: str, evidence: List[Dict]) -> Dict[str, Any]:
        """Verify answer using BART-MNLI."""

        if not evidence:
            return {"status": "UNKNOWN", "trust_score": 0, "confidence": 0.0}

        try:
            # Combine all evidence
            context = " ".join([doc.page_content for doc in evidence])

            # Check if answer is entailed by the context
            premise = context[:1024]  # Limit context length
            hypothesis = f"The answer to '{question}' is '{answer}'"

            inputs = self.nli_tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

            # Get probabilities for entailment vs contradiction
            entailment_prob = probs[0][2].item()  # Index 2 is entailment
            contradiction_prob = probs[0][0].item()  # Index 0 is contradiction

            # Calculate trust score
            trust_score = int(entailment_prob * 100)

            # Determine status
            if entailment_prob > 0.7:
                status = "VERIFIED"
            elif contradiction_prob > 0.6:
                status = "HALLUCINATION"
            else:
                status = "INCOMPLETE"

            return {
                "status": status,
                "trust_score": trust_score,
                "confidence": entailment_prob
            }

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"status": "UNKNOWN", "trust_score": 0, "confidence": 0.0}

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        try:
            embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.array([])
