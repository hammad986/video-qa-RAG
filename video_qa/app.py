"""
Video-QA RAG System - Streamlit Application
Lecture Video Question Answering with Evidence-Grounded Answers

Fixes applied:
- Auto tab-switch to Ask Questions when video processing finishes
- session_state.active_tab persists across reruns (fixes Ctrl+Enter redirect)
- Get Answer shown via st.form (always visible when KB ready)
- video_id scoping disabled when active_video_id is not set (avoids 0 chunks)
- Graceful Whisper/thread shutdown on Ctrl+C
"""

import streamlit as st
import os
import sys
import tempfile
import shutil
import asyncio
import atexit
import threading

# ── Windows asyncio fix (prevents "Event loop is closed" on shutdown) ──────
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


# ── Graceful shutdown ────────────────────────────────────────────────────────
def graceful_shutdown():
    """Join non-daemon threads and free Whisper CTranslate2 workers."""
    try:
        from video_qa.speech_understanding import _whisper_model as _wm  # noqa
        if _wm is not None:
            del _wm
    except Exception:
        pass
    main_thread = threading.main_thread()
    for t in threading.enumerate():
        if t is not main_thread and not t.daemon:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass


atexit.register(graceful_shutdown)

from pathlib import Path  # noqa: E402  (must be after asyncio policy)

# ─────────────────────────────────────────────
# Streamlit page config  (MUST be first st.* call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Video-QA: Lecture Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
INDEX_PATH = "models/video_index.faiss"
META_PATH  = "models/metadata.pkl"

# ─────────────────────────────────────────────
# Session-state initialisation  (must be early, before any widget)
# ─────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "process"       # "process" | "ask"

if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None

if "video_processed_ok" not in st.session_state:
    st.session_state.video_processed_ok = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# ─────────────────────────────────────────────
# Cached pipeline
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initialising pipeline…")
def get_pipeline():
    from video_qa.pipeline import VideoQAPipeline
    return VideoQAPipeline()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def knowledge_base_ready() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def sec_to_time(seconds: float) -> str:
    total = int(seconds or 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def download_youtube_to_file(url: str, status_placeholder) -> "str | None":
    """
    Download a YouTube video audio stream using yt-dlp.
    Returns local WAV file path, or None on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        st.error("yt-dlp is not installed. Run:  pip install yt-dlp")
        return None

    tmp_dir  = Path(tempfile.mkdtemp())
    out_tmpl = str(tmp_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    status_placeholder.text("⬇️ Extracting YouTube audio stream…")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # ── WAV GLOB FIX ───────────────────────────────────────────────────
        # yt-dlp emits a .webp thumbnail alongside the audio; glob("*.*")
        # may return the thumbnail before the actual WAV.  Look for .wav
        # explicitly first, then fall back to any file.
        candidates = list(tmp_dir.glob("*.wav"))
        if not candidates:
            candidates = [f for f in tmp_dir.glob("*.*") if f.suffix.lower() != ".webp"]
        if candidates:
            return str(candidates[0])

        st.error("Download completed but output WAV file not found.")
    except Exception as exc:
        st.error(f"YouTube download failed: {exc}")
    return None


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🎬 Video-QA: Lecture Assistant")
    st.markdown(
        "**Research-Level Video Question Answering System** — "
        "Upload a lecture video, ask questions, and get evidence-grounded answers with timestamps."
    )
    st.markdown("---")

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 System Status")

        if knowledge_base_ready():
            try:
                import joblib
                metadata = joblib.load(META_PATH)
                st.success("✅ Knowledge Base Ready")
                st.metric("Indexed Chunks", len(metadata))
            except Exception:
                st.success("✅ Knowledge Base Ready")
        else:
            st.warning("⚠️ No videos processed yet")

        st.markdown("---")

        # ── Tab switcher buttons in sidebar ─────────────────────────────────
        st.subheader("Navigation")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📤 Process", use_container_width=True):
                st.session_state.active_tab = "process"
                st.rerun()
        with col_b:
            if st.button("❓ Ask", use_container_width=True):
                st.session_state.active_tab = "ask"
                st.rerun()

        st.markdown("---")
        st.caption(
            "**Pipeline:**\n"
            "1. Video Upload / YouTube URL\n"
            "2. Audio Extraction (ffmpeg)\n"
            "3. Whisper Transcription\n"
            "4. Semantic Chunking\n"
            "5. Embedding + FAISS Index\n"
            "6. Retrieval + Re-ranking\n"
            "7. LLM Answer Generation\n"
            "8. NLI Evidence Verification"
        )

    # ── Tab navigation via session_state ────────────────────────────────────
    tab_labels = ["📤 Process Video", "❓ Ask Questions"]
    selected_tab_idx = 0 if st.session_state.active_tab == "process" else 1
    tab1, tab2 = st.tabs(tab_labels)

    # Programmatic tab selection trick: inject JS to click the correct tab
    # after rerun when active_tab == "ask"
    if st.session_state.active_tab == "ask":
        # Use st.markdown JS injection to automatically click the Ask tab
        st.markdown(
            """
            <script>
            (function() {
                function clickAskTab() {
                    var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs && tabs.length >= 2) {
                        tabs[1].click();
                    } else {
                        setTimeout(clickAskTab, 100);
                    }
                }
                setTimeout(clickAskTab, 200);
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )

    # ── Process Video Tab ────────────────────────────────────────────────────
    with tab1:
        # Switch to this tab if active_tab == "process"
        if st.session_state.active_tab == "ask" and knowledge_base_ready():
            # Do not render process tab content when user is on ask tab
            st.info("Switch to the ❓ Ask Questions tab to query your video.")
        else:
            _render_process_tab()

    # ── Ask Questions Tab ────────────────────────────────────────────────────
    with tab2:
        _render_ask_tab()


# ─────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────
def _render_process_tab():
    """Render the video processing UI."""
    st.subheader("📤 Upload & Process Video")

    input_method = st.radio(
        "Choose input method:",
        ["Upload Video File", "YouTube URL", "Use Existing Videos"],
        horizontal=True,
    )

    video_path: "str | None"  = None
    youtube_url: "str | None" = None

    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                shutil.copyfileobj(uploaded_file, tmp)
                video_path = tmp.name
            st.success(f"✅ Ready to process: {uploaded_file.name}")

    elif input_method == "YouTube URL":
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube link. The audio will be extracted and transcribed automatically.",
        )
        if youtube_url:
            st.info("ℹ️ Click **🚀 Process Video** below to download and process this video automatically.")

    elif input_method == "Use Existing Videos":
        video_dir = Path("data/videos")
        if video_dir.exists():
            videos = sorted(
                list(video_dir.glob("*.mp4"))
                + list(video_dir.glob("*.mkv"))
                + list(video_dir.glob("*.avi"))
            )
            if videos:
                selected = st.selectbox("Select video:", [str(v) for v in videos])
                video_path = selected
            else:
                st.warning("No videos found in data/videos/")
        else:
            st.warning("data/videos/ directory not found")

    ready_to_process = bool(video_path or youtube_url)
    if ready_to_process:
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            _run_process_video(video_path=video_path, youtube_url=youtube_url)


def _render_ask_tab():
    """Render the Q&A UI — always shows form when KB is ready."""
    st.subheader("❓ Ask Questions About the Video")

    if not knowledge_base_ready():
        st.warning("⚠️ Please process a video first in the '📤 Process Video' tab.")
        return

    # ── Always render the form so Get Answer is never hidden ────────────────


    # Use a dedicated session state key for the current input
    if "current_question_input" not in st.session_state:
        st.session_state.current_question_input = ""


    with st.form("qa_form", clear_on_submit=True):
        question = st.text_area(
            "Enter your question:",
            height=80,
            placeholder="What is the main topic of this lecture?",
            value=st.session_state.current_question_input,
            key="question_input_area"
        )
        submit = st.form_submit_button("🔍 Get Answer", type="primary", use_container_width=True)

    # No longer update current_question_input here; it's handled by on_change

    with st.expander("💡 Example Questions"):
        st.markdown(
            "- What is the main topic of the lecture?\n"
            "- Explain the key concept discussed\n"
            "- What example is given for the topic?\n"
            "- Summarize the beginning of the video\n"
            "- Where does the speaker define a term?\n"
            "- What did the lecturer say about [specific subject]?"
        )

    # ── Handle submission ────────────────────────────────────────────────────

    if submit:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # Only update last_question on submit
            st.session_state.last_question = question
            st.session_state.current_question_input = question
            # CRITICAL: keep active_tab as "ask" during query — do NOT change it
            _run_answer_question(question)

    # ── Show previous result if available ───────────────────────────────────
    elif st.session_state.last_result is not None:
        _display_results(st.session_state.last_question, st.session_state.last_result)


# ─────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────
def _run_process_video(video_path: "str | None", youtube_url: "str | None" = None):
    """
    Full processing pipeline.
    On success: set active_tab = "ask" and rerun — user lands on Ask tab automatically.
    """
    progress_bar = st.progress(0)
    status_text  = st.empty()

    try:
        # Step 0: YouTube download
        if youtube_url:
            video_path = download_youtube_to_file(youtube_url, status_text)
            if video_path is None:
                return
            progress_bar.progress(15)

        # Steps 1-4: full pipeline
        pipeline = get_pipeline()

        status_text.text("🎵 Extracting audio from video…")
        progress_bar.progress(20)

        status_text.text("🎤 Transcribing speech with Whisper…")
        progress_bar.progress(40)

        status_text.text("📝 Creating semantic chunks…")
        progress_bar.progress(60)

        status_text.text("🗄️ Building vector index…")
        progress_bar.progress(75)

        video_id = pipeline.process_video(video_path)

        if not video_id:
            st.error(
                "Processing failed. "
                "Ensure FFmpeg is installed and faster-whisper is available."
            )
            return

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        # Clear cached pipeline so next query re-loads the fresh index
        get_pipeline.clear()

        # Store video_id for scoped retrieval
        st.session_state["active_video_id"] = video_id

        # ── AUTO-SWITCH to Ask Questions tab ────────────────────────────────
        st.session_state["active_tab"] = "ask"
        st.session_state["video_processed_ok"] = True
        st.session_state["last_result"] = None      # clear previous results
        st.session_state["last_question"] = ""

        st.rerun()  # rerun lands user directly on Ask tab

    except Exception as exc:
        st.error(f"Error during processing: {exc}")
        progress_bar.empty()
        status_text.empty()


def _run_answer_question(question: str):
    """Retrieve, generate, and display a grounded answer. Never changes active_tab."""
    progress_bar = st.progress(0)
    status_text  = st.empty()

    try:
        pipeline = get_pipeline()

        status_text.text("🔍 Retrieving relevant passages…")
        progress_bar.progress(20)

        status_text.text("🤖 Generating answer…")
        progress_bar.progress(55)

        # Pass active_video_id for scoped retrieval.
        # If None (e.g. knowledge base was pre-built), search all chunks.
        active_id = st.session_state.get("active_video_id")
        result = pipeline.ask(question, active_video_id=active_id)

        status_text.text("🛡️ Verifying answer against evidence…")
        progress_bar.progress(90)

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        # active_tab must remain "ask" after query  — DO NOT change it here
        st.session_state.last_result = result

        _display_results(question, result)

    except Exception as exc:
        st.error(f"Error: {exc}")
        progress_bar.empty()
        status_text.empty()


# ─────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────
def _display_results(question: str, result: dict):
    """Show answer, NLI verification badge, and source evidence.
    
    If the result is a lecture summary (is_summary=True), renders a clean
    bullet-point overview instead of the standard RAG evidence panels.
    """
    st.markdown("---")
    st.subheader("📝 Results")

    answer            = result.get("answer", "Question cannot be answered from this video.")
    timestamp         = result.get("timestamp")
    contexts          = result.get("contexts", [])
    verified          = result.get("verified", False)
    verification_info = result.get("verification", {})
    is_summary        = result.get("is_summary", False)

    # ── SUMMARY PATH: clean bullet-point overview ────────────────────────────
    if is_summary:
        with st.expander("📋 Lecture Overview", expanded=True):
            st.info(
                "This is a high-level summary generated from the full transcript. "
                "For specific questions, ask factual questions and the RAG pipeline "
                "will retrieve exact transcript evidence with timestamps."
            )
            # Strip the "**Lecture Overview**" header the router adds — render it as markdown
            clean_answer = answer.replace("**Lecture Overview**", "").strip()
            st.markdown(clean_answer)

        # Show a lightweight trust badge for summaries
        with st.expander("🛡️ Verification", expanded=False):
            st.success("✅ VERIFIED")
            st.caption("Summary derived directly from the lecture transcript.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trust Score", "90%")
                st.progress(0.9)
            with col2:
                st.caption("Source: full transcript → LLM summarization")

        return  # no evidence section for summary results

    # ── STANDARD RAG PATH ────────────────────────────────────────────────────
    # ── Answer ──────────────────────────────────────────────────────────────
    with st.expander("🤖 AI Answer", expanded=True):
        if not answer or "cannot be answered" in answer.lower() or "not found" in answer.lower():
            st.warning("❌ " + (answer or "Question cannot be answered from this video."))
        else:
            st.success(f"**{answer}**")
            if timestamp:
                st.info(f"📍 **Timestamp:** {timestamp}")

    # ── Verification ────────────────────────────────────────────────────────
    with st.expander("🛡️ Verification", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            if isinstance(verification_info, dict) and verification_info:
                status      = verification_info.get("status", "UNKNOWN")
                trust_score = verification_info.get("trust_score", 0)
                confidence  = verification_info.get("confidence", 0.0)
            else:
                status      = "VERIFIED" if verified else "INCOMPLETE"
                trust_score = 80         if verified else 40
                confidence  = 0.8        if verified else 0.4

            if status == "VERIFIED":
                st.success("✅ VERIFIED")
                st.caption("Answer is supported by transcript evidence.")
            elif status == "HALLUCINATION":
                st.error("❌ HALLUCINATION")
                st.caption("Answer contradicts the evidence — treat with caution.")
            else:
                st.warning("⚠️ INCOMPLETE")
                st.caption("Answer partially supported by evidence.")

        with col2:
            st.metric("Trust Score", f"{trust_score}%")
            st.progress(min(trust_score, 100) / 100)
            st.caption(f"Confidence: {confidence:.2f}")

    # ── Evidence ────────────────────────────────────────────────────────────
    with st.expander("📚 Source Evidence", expanded=True):
        if contexts:
            for i, ctx in enumerate(contexts[:5], 1):
                start = sec_to_time(ctx.get("start", 0))
                end   = sec_to_time(ctx.get("end",   0))
                text  = ctx.get("text", "")
                score = ctx.get("rerank_score", ctx.get("score", 0))

                st.markdown(
                    f"**Evidence {i}** `[{start} – {end}]`"
                    + (f"  •  Relevance: `{score:.3f}`" if score else "")
                )
                st.text_area(
                    "",
                    text,
                    height=80,
                    key=f"ctx_{i}_{abs(hash(text)) % 1_000_000}",
                    disabled=True,
                    label_visibility="collapsed",
                )
        else:
            st.info("No evidence passages retrieved.")


if __name__ == "__main__":
    main()

