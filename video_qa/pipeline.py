"""
Pipeline Module for Video-QA System
Integrates all components: video processing, speech understanding,
knowledge structuring, embeddings, retrieval, re-ranking, and answer generation.

Fixes applied:
- Force fresh audio extraction on every new video (force=True)
- Delete stale transcript/chunk JSON files before re-processing
- Build FAISS index from scratch (reset=True) to prevent cross-video contamination
- Lecture summary generated after transcription (saved to data/summaries/)
- Query router: summary/overview questions return stored summary instantly,
  factual questions continue through the standard RAG pipeline
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import config
from .logger import get_logger, setup_logger
from .query_rewriter import rewrite_query

from .video_processor import (
    take_video_input,
    convert_video_to_audio,
    check_ffmpeg
)

from .speech_understanding import transcribe_audio
from .knowledge_structuring import process_transcript_to_chunks, process_all_transcripts
from .embeddings import build_vector_index, VectorStore
from .retrieval import RetrievalSystem
from .reranker import ReRanker
from .answer_generator import AnswerGenerator
from .query_router import generate_lecture_summary, route_summary_query


logger = get_logger(__name__)


class VideoQAPipeline:
    """
    Complete Video-QA Pipeline.
    
    Integrates all 9 stages:
    1. Video Processing
    2. Speech Recognition (WhisperX)
    3. Knowledge Structuring (Semantic Segmentation)
    4. Indexing (BGE Embeddings)
    5. Retrieval (FAISS)
    6. Re-Ranking (Cross-Encoder)
    7. Answer Generation (LLM)
    8. Evidence Verification
    9. Output
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.retriever = None
        self.reranker = None
        self.answer_generator = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize retrieval components."""
        # Initialize retrieval system
        self.retriever = RetrievalSystem()
        
        # Initialize re-ranker
        self.reranker = ReRanker()
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator()
        
        logger.info("Pipeline components initialized")
    
    def process_video(self, video_path: str) -> Optional[str]:
        """
        Process a video through the complete pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            video_id string if successful, else None
        """
        logger.info("="*60)
        logger.info("VIDEO-QA PIPELINE - FULL PROCESSING")
        logger.info("="*60)
        
        # Stage 1: Video Processing
        logger.info("\n[STAGE 1/5] Video Processing...")

        if not check_ffmpeg():
            logger.error("FFmpeg not available")
            return None

        # Always force re-extract audio so a re-uploaded video is never skipped.
        audio_path = convert_video_to_audio(video_path, force=True)
        if audio_path is None:
            logger.error("Audio extraction failed")
            return None

        # ── STALE-CACHE FIX: remove old transcript + chunk files for this stem
        # so that Whisper always runs and chunking always produces fresh output.
        stem = Path(audio_path).stem
        for _stale in [
            Path("data/transcripts") / f"{stem}.json",
            Path("data/chunks")      / f"{stem}.json",
        ]:
            if _stale.exists():
                _stale.unlink()
                logger.info(f"Removed stale file: {_stale.name}")

        # Stage 2: Speech Recognition
        logger.info("\n[STAGE 2/5] Speech Recognition...")

        transcript_path = transcribe_audio(audio_path)
        if transcript_path is None:
            logger.error("Transcription failed")
            return None

        # Stage 3: Knowledge Structuring
        logger.info("\n[STAGE 3/5] Knowledge Structuring (Semantic Segmentation)...")

        chunks_path = process_transcript_to_chunks(transcript_path)
        if chunks_path is None:
            logger.error("Chunking failed")
            return None

        # Stage 4: Indexing
        # reset=True clears the existing FAISS index so chunks from a *previous*
        # video are not mixed with chunks from this new video.
        logger.info("\n[STAGE 4/5] Building Vector Index (BGE Embeddings) — fresh index...")

        success = build_vector_index(reset=True)
        if not success:
            logger.error("Index building failed")
            return None

        # ── HYBRID RAG: Generate lecture summary ─────────────────────────────
        # Load the raw transcript text and send it to the LLM to produce a
        # 5-bullet overview.  Stored at data/summaries/<video_id>.json.
        # Non-blocking: a failure to generate the summary is only a warning.
        video_id = Path(transcript_path).stem
        logger.info("\n[STAGE 5/5] Generating lecture summary…")
        try:
            import json as _json
            with open(transcript_path, "r", encoding="utf-8") as _f:
                _tdata = _json.load(_f)
            _full_text = _tdata.get("full_transcript", "")
            if _full_text:
                _summary = generate_lecture_summary(_full_text, video_id)
                if _summary:
                    logger.info(f"Lecture summary generated: {len(_summary.get('bullets', []))} bullets")
                else:
                    logger.warning("Lecture summary generation skipped (LLM unavailable).")
            else:
                logger.warning("No full_transcript field in transcript JSON — summary skipped.")
        except Exception as _exc:
            logger.warning(f"Summary generation failed (non-fatal): {_exc}")

        logger.info("\n[COMPLETE] Pipeline finished!")
        logger.info("="*60)

        # Re-initialize components with new index
        self._initialize_components()

        return video_id
    
    def process_directory(self, video_dir: str = "data/videos") -> int:
        """
        Process all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            
        Returns:
            Number of successfully processed videos
        """
        video_path = Path(video_dir)
        
        if not video_path.exists():
            logger.error(f"Video directory not found: {video_dir}")
            return 0
        
        # Get all video files
        supported = config.get("video.supported_formats", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
        video_files = [f for f in video_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in supported]
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return 0
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        success_count = 0
        for video_file in video_files:
            logger.info(f"\nProcessing: {video_file.name}")
            
            if self.process_video(str(video_file)):
                success_count += 1
        
        logger.info(f"\nProcessed {success_count}/{len(video_files)} videos successfully")
        
        return success_count
    
    def ask(self, query: str, active_video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question about processed videos.

        HYBRID ROUTING:
          • Summary / overview questions  →  returns stored lecture summary instantly
          • Factual / evidence questions  →  standard RAG pipeline

        Args:
            query:           User question
            active_video_id: Scopes retrieval to a specific video

        Returns:
            Dictionary with answer, timestamp, contexts, and metadata
        """
        logger.info("="*60)
        logger.info(f"QUERY: {query}")
        logger.info("="*60)

        # ── PATH A: Summary / overview questions ─────────────────────────────
        summary_result = route_summary_query(query, active_video_id)
        if summary_result is not None:
            logger.info("[Pipeline] Returning pre-computed lecture summary.")
            return summary_result

        # ── PATH B: Factual questions — standard RAG ──────────────────────────
        # Check if retrieval is available
        if not self.retriever.is_available():
            logger.error("Vector index not available. Please process videos first.")
            return {
                "answer": "Question cannot be answered from this video.",
                "timestamp": None,
                "error": "No videos processed"
            }

        # Stage 5A: Query Rewrite
        logger.info("\n[QUERY REWRITE] Optimizing search query...")
        search_query = rewrite_query(query)
        if search_query != query:
            logger.info(f"Using search query: '{search_query}'")
        else:
            logger.info("Using original query for search.")

        # Stage 5B: Retrieval (top-5)
        # Use search_query for retrieving relevant chunks
        logger.info("\n[RETRIEVAL] Finding relevant passages...")
        retrieved = self.retriever.retrieve(search_query, top_k=5, video_id=active_video_id)

        if not retrieved:
            logger.warning("No relevant passages found")
            return {
                "answer": "Question cannot be answered from this video.",
                "timestamp": None,
                "contexts": []
            }

        logger.info(f"Retrieved {len(retrieved)} passages")

        # Stage 6: Re-Ranking (top-3)
        logger.info("\n[RE-RANKING] Re-ranking passages...")
        reranked = self.reranker.rerank(query, retrieved)

        logger.info(f"Re-ranked to top {len(reranked)} passages")

        # Stage 7: Answer Generation
        logger.info("\n[ANSWER GENERATION] Generating answer...")
        result = self.answer_generator.generate(query, reranked)

        # Stage 8 & 9: Verification and Output
        logger.info("\n[OUTPUT]")
        logger.info(f"Answer: {result.get('answer')}")
        logger.info(f"Timestamp: {result.get('timestamp')}")
        logger.info(f"Verified: {result.get('verified', False)}")

        return result
    
    def interactive_mode(self):
        """Run interactive question-answering mode."""
        print("\n" + "="*60)
        print("VIDEO-QA INTERACTIVE MODE")
        print("="*60)
        print("\nAsk questions about the video(s).")
        print("Type 'quit' or 'exit' to end session.")
        print("Type 'menu' to see options.\n")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'menu':
                    print("""
Options:
- Ask any question about the video content
- Type 'quit' to exit
                    """)
                    continue
                
                result = self.ask(query)
                
                print("\n" + "-"*40)
                print("ANSWER:")
                print("-"*40)
                print(result.get('answer', 'NOT FOUND IN VIDEO'))
                
                if result.get('timestamp'):
                    print(f"\nTimestamp: {result['timestamp']}")
                
                if result.get('verified'):
                    print("\n[✓] Answer verified against evidence")
                
                print("-"*40)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def run_full_pipeline(video_path: Optional[str] = None) -> bool:
    """
    Run full pipeline: process video and answer questions.
    
    Args:
        video_path: Optional path to video file
        
    Returns:
        True if successful
    """
    # Initialize pipeline
    pipeline = VideoQAPipeline()
    
    # Get video path
    if video_path is None:
        video_path = take_video_input()
    
    if video_path is None:
        print("No video selected")
        return False
    
    # Process video
    success = pipeline.process_video(video_path)
    
    if not success:
        print("Video processing failed")
        return False
    
    # Run interactive mode
    pipeline.interactive_mode()
    
    return True


def run_query_only() -> Dict[str, Any]:
    """
    Run query-only mode (assumes videos already processed).
    
    Returns:
        Query result dictionary
    """
    pipeline = VideoQAPipeline()
    
    if not pipeline.retriever.is_available():
        print("No vector index found. Please process videos first.")
        return {"answer": "NOT FOUND IN VIDEO", "error": "No index"}
    
    query = input("\nAsk a question: ").strip()
    
    if not query:
        return {"answer": "NOT FOUND IN VIDEO", "error": "Empty query"}
    
    return pipeline.ask(query)


if __name__ == "__main__":
    # Setup logging
    setup_logger()
    
    # Run pipeline
    print("""
=================================================
   VIDEO KNOWLEDGE RETRIEVAL AI SYSTEM
   (Research-Level RAG Pipeline)
=================================================

1 → Process New Video
2 → Process All Videos in Directory
3 → Ask Question (Interactive Mode)
4 → Exit
    """)
    
    choice = input("Enter choice: ").strip()
    
    pipeline = VideoQAPipeline()
    
    if choice == "1":
        video_path = take_video_input()
        if video_path:
            pipeline.process_video(video_path)
    
    elif choice == "2":
        pipeline.process_directory()
    
    elif choice == "3":
        if pipeline.retriever.is_available():
            pipeline.interactive_mode()
        else:
            print("No videos processed yet. Please process videos first.")
    
    else:
        print("Exiting.")
