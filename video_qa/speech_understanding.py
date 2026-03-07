"""
Speech Understanding Module for Video-QA System
Handles Whisper transcription, WhisperX forced alignment,
punctuation restoration, and LLM transcript correction.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from .config import config
from .logger import get_logger


logger = get_logger(__name__)


# Global model cache
_whisper_model = None
_align_model = None


def load_whisper_model():
    """Load Whisper model (cached)."""
    global _whisper_model
    
    if _whisper_model is None:
        speech_config = config.get_section("speech")
        model_size = speech_config.get("model_size", "small")
        device = speech_config.get("device", "cpu")
        compute_type = speech_config.get("compute_type", "int8")
        
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model: {model_size}")
            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.warning("faster-whisper not installed, falling back to whisper")
            import whisper
            _whisper_model = whisper.load_model(model_size, device=device)
    
    return _whisper_model


def load_alignment_model():
    """Load WhisperX alignment model."""
    global _align_model
    
    if _align_model is None:
        try:
            import whisperx
            logger.info("Loading WhisperX alignment model...")
            _align_model = whisperx.load_align_model(language_code="en")
            logger.info("Alignment model loaded")
        except ImportError:
            logger.warning("whisperx not available, skipping alignment")
    
    return _align_model


def transcribe_with_whisper(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with transcript and segments
    """
    model = load_whisper_model()
    speech_config = config.get_section("speech")
    
    logger.info(f"Transcribing: {audio_path}")
    
    try:
        # Try faster-whisper API
        segments, info = model.transcribe(
            audio_path,
            language="en",
            beam_size=speech_config.get("beam_size", 1),
            best_of=speech_config.get("best_of", 1),
            temperature=speech_config.get("temperature", 0),
            vad_filter=speech_config.get("vad_filter", True),
            vad_parameters=dict(
                min_silence_duration_ms=speech_config.get("vad_min_silence_ms", 700)
            )
        )
        
        # Convert to list
        segment_list = []
        full_text = ""
        
        for seg in segments:
            text = seg.text.strip()
            full_text += text + " "
            segment_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text
            })
        
        result = {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "full_text": full_text.strip(),
            "segments": segment_list
        }
        
        logger.info(f"Transcription complete: {len(segment_list)} segments")
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


def align_transcript(audio_path: str, transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply WhisperX forced alignment for word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        transcript: Whisper transcript result
        
    Returns:
        Transcript with aligned word timestamps
    """
    align_model = load_alignment_model()
    
    if align_model is None:
        logger.info("Skipping alignment (whisperx not available)")
        return transcript
    
    try:
        import whisperx
        
        # Get aligned segments
        aligned = whisperx.align(
            transcript["segments"],
            align_model,
            "en",
            audio_path
        )
        
        if aligned and "segments" in aligned:
            # Update segments with word-level alignment
            transcript["segments"] = aligned["segments"]
            logger.info("Forced alignment complete")
        
    except Exception as e:
        logger.warning(f"Alignment failed: {e}")
    
    return transcript


def restore_punctuation(text: str) -> str:
    """
    Restore punctuation using a simple rule-based approach.
    
    For production, consider using a trained punctuation model.
    
    Args:
        text: Text to punctuate
        
    Returns:
        Text with restored punctuation
    """
    # Simple heuristics
    # Add period at end if missing
    if text and text[-1] not in '.!?':
        text = text + '.'
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text


def correct_transcript_with_llm(transcript: str) -> str:
    """
    Use LLM to correct transcript (grammar, technical terms).
    
    Args:
        transcript: Raw transcript text
        
    Returns:
        Corrected transcript
    """
    try:
        import requests
        
        correction_prompt = f"""You are a transcript corrector. Correct any grammar errors, 
fix misheard technical terms, and improve readability while keeping the original meaning.
Only output the corrected text, nothing else.

Original transcript:
{transcript}

Corrected transcript:"""

        response = requests.post(
            config.get("answer.ollama_url", "http://localhost:11434/api/generate"),
            json={
                "model": config.get("answer.local_model", "phi3:mini"),
                "prompt": correction_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            corrected = response.json().get("response", "").strip()
            logger.info("LLM correction applied")
            return corrected
        
    except Exception as e:
        logger.warning(f"LLM correction failed: {e}")
    
    return transcript


def transcribe_audio(audio_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Full speech understanding pipeline:
    Whisper → Alignment → Punctuation → LLM Correction
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory for transcript
        
    Returns:
        Path to transcript JSON file
    """
    logger.info("="*50)
    logger.info("SPEECH UNDERSTANDING PIPELINE")
    logger.info("="*50)
    
    # Setup paths
    audio_path_obj = Path(audio_path)
    output_dir = output_dir or "data/transcripts"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    transcript_file = output_path / f"{audio_path_obj.stem}.json"
    
    # Check if already transcribed
    if transcript_file.exists():
        logger.info(f"Transcript already exists: {transcript_file.name}")
        return str(transcript_file)
    
    # Step 1: Whisper transcription
    logger.info("Step 1: Whisper transcription...")
    transcript = transcribe_with_whisper(audio_path)
    
    if not transcript:
        logger.error("Transcription failed")
        return None
    
    # Step 2: WhisperX forced alignment
    if config.get("speech.alignment", True):
        logger.info("Step 2: WhisperX forced alignment...")
        transcript = align_transcript(audio_path, transcript)
    
    # Step 3: Punctuation restoration (simple)
    if config.get("speech.restore_punctuation", True):
        logger.info("Step 3: Restoring punctuation...")
        # Apply to each segment
        for seg in transcript["segments"]:
            seg["text"] = restore_punctuation(seg["text"])
        transcript["full_text"] = restore_punctuation(transcript["full_text"])
    
    # Step 4: LLM correction pass
    if config.get("speech.llm_correction", False):
        logger.info("Step 4: LLM correction pass...")
        corrected_text = correct_transcript_with_llm(transcript["full_text"])
        transcript["full_text"] = corrected_text
        # Note: LLM correction doesn't preserve segment boundaries
    
    # Calculate metadata
    total_duration = 0
    word_count = 0
    if transcript["segments"]:
        total_duration = transcript["segments"][-1]["end"]
        word_count = len(transcript["full_text"].split())
    
    # Save transcript
    output_data = {
        "video_name": audio_path_obj.stem,
        "language": transcript.get("language", "en"),
        "language_confidence": transcript.get("language_probability", 0),
        "total_segments": len(transcript["segments"]),
        "total_duration": round(total_duration, 2),
        "word_count": word_count,
        "full_transcript": transcript["full_text"],
        "segments": transcript["segments"]
    }
    
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Transcript saved: {transcript_file.name}")
    logger.info(f"Duration: {total_duration:.1f}s, Words: {word_count}")
    
    return str(transcript_file)


if __name__ == "__main__":
    # Test transcription
    import sys
    
    if len(sys.argv) > 1:
        audio = sys.argv[1]
    else:
        audio = input("Enter audio path: ")
    
    result = transcribe_audio(audio)
    print(f"\nOutput: {result}")
