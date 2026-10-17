"""
Video-QA RAG: Evidence-Grounded Video Question Answering
Powered by DeepSeek V4 Flash (agentrouter.org), Faster-Whisper, BGE/E5 Embeddings & FAISS
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
load_dotenv()

# Compatibility patch for huggingface_hub >= 1.0 with older Gradio oauth
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                import os
                return os.environ.get("HF_TOKEN")
            @staticmethod
            def save_token(token):
                pass
            @staticmethod
            def delete_token():
                pass
        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

# ZeroGPU integration if running on Hugging Face Spaces with ZeroGPU
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

import gradio as gr
from video_qa.pipeline import VideoQAPipeline
from video_qa.query_rewriter import rewrite_query

# Singleton pipeline instance
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = VideoQAPipeline()
    return _pipeline

def sec_to_time(seconds: float) -> str:
    total = int(seconds or 0)
    return f"{total // 60:02d}:{total % 60:02d}"

def prepare_cookies(cookie_file=None, cookie_text: str = "") -> Optional[str]:
    """Resolve cookie file from user upload, paste, local file, or env var."""
    if cookie_file and os.path.exists(str(cookie_file)):
        return str(cookie_file)
    if cookie_text and cookie_text.strip():
        tmp = Path(tempfile.mkdtemp()) / "cookies.txt"
        tmp.write_text(cookie_text.strip(), encoding="utf-8")
        return str(tmp)
    local_cookie = Path("cookies.txt")
    if local_cookie.exists():
        return str(local_cookie)
    env_cookie = os.getenv("YOUTUBE_COOKIES") or os.getenv("COOKIES_TXT")
    if env_cookie and env_cookie.strip():
        tmp = Path(tempfile.mkdtemp()) / "cookies.txt"
        tmp.write_text(env_cookie.strip(), encoding="utf-8")
        return str(tmp)
    return None

def download_youtube_audio(url: str, cookie_file=None, cookie_text: str = "") -> Optional[str]:
    """Download audio stream from YouTube URL using yt-dlp with cookie support."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed.")

    tmp_dir = Path(tempfile.mkdtemp())
    out_tmpl = str(tmp_dir / "%(id)s.%(ext)s")

    cookie_path = prepare_cookies(cookie_file, cookie_text)

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "ba/b",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web"]
            }
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    if cookie_path and os.path.exists(cookie_path):
        ydl_opts["cookiefile"] = cookie_path
        print(f"Loaded YouTube cookies from: {cookie_path}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as primary_err:
        err_str = str(primary_err)
        print(f"Primary YouTube extraction failed: {err_str}")
        if "Sign in to confirm you're not a bot" in err_str or "cookies" in err_str.lower():
            if not cookie_path:
                raise RuntimeError(
                    "YouTube Bot Protection on Cloud Servers: YouTube requires cookies on cloud datacenter IPs. "
                    "Please either:\n"
                    "1. Upload the video/audio file (.mp4, .wav, .mp3) directly using 'Or Upload Video / Audio File', or\n"
                    "2. Export your YouTube cookies using 'Get cookies.txt LOCALLY' extension and upload/paste them in the '🍪 YouTube Cookies' section below."
                )
        # Try fallback options without player_client restrictions
        try:
            fallback_opts = {
                "outtmpl": out_tmpl,
                "format": "ba/b",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }],
                "quiet": True,
            }
            if cookie_path and os.path.exists(cookie_path):
                fallback_opts["cookiefile"] = cookie_path
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                ydl.download([url])
        except Exception as fallback_err:
            raise RuntimeError(f"YouTube extraction failed: {fallback_err}")

    candidates = list(tmp_dir.glob("*.wav"))
    if not candidates:
        candidates = [f for f in tmp_dir.glob("*.*") if f.suffix.lower() not in [".webp", ".jpg", ".png", ".json"]]
    return str(candidates[0]) if candidates else None

def _run_processing_cpu(file_to_process: str):
    pipeline = get_pipeline()
    video_id = pipeline.process_video(file_to_process)
    return video_id

if HAS_ZEROGPU:
    @spaces.GPU(duration=30)
    def _run_processing_gpu(file_to_process: str):
        return _run_processing_cpu(file_to_process)

    def _run_processing(file_to_process: str):
        # Always run directly on CPU:
        # All models (Whisper int8, MiniLM-L6, FAISS) are tuned for CPU and complete
        # in ~2 seconds. Running directly on CPU avoids ZeroGPU worker errors,
        # visitor quota limits, and CUDA init emulation collisions.
        return _run_processing_cpu(file_to_process)
else:
    def _run_processing(file_to_process: str):
        return _run_processing_cpu(file_to_process)

def process_video_fn(video_file, youtube_url: str, cookie_file=None, cookie_text: str = ""):
    """Process video/audio file or YouTube link into FAISS vector index."""
    if not video_file and not (youtube_url and youtube_url.strip()):
        return "⚠️ Please provide either a video/audio file or a YouTube URL.", "", ""

    file_to_process = None

    try:
        if youtube_url and youtube_url.strip():
            target_url = youtube_url.strip()
            file_to_process = download_youtube_audio(target_url, cookie_file, cookie_text)
            if not file_to_process:
                return "❌ Failed to extract audio stream from YouTube URL.", "", ""
        elif video_file:
            file_to_process = video_file

        # Process through RAG ingestion pipeline
        video_id = _run_processing(file_to_process)
        if not video_id:
            return "❌ Processing failed during transcription or chunking.", "", ""

        # Retrieve transcript
        transcript_file = Path("data/transcripts") / f"{video_id}.json"
        transcript_text = ""
        if transcript_file.exists():
            import json
            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    segments = tdata.get("segments", [])
                    transcript_text = "\n".join(
                        f"[{sec_to_time(s.get('start', 0))} - {sec_to_time(s.get('end', 0))}] {s.get('text', '')}"
                        for s in segments
                    )
            except Exception:
                pass

        summary_file = Path("data/summaries") / f"{video_id}_summary.txt"
        summary_text = ""
        if summary_file.exists():
            try:
                summary_text = summary_file.read_text(encoding="utf-8")
            except Exception:
                pass

        success_msg = f"✅ **Video successfully processed and indexed!**\n\n- **Video ID**: `{video_id}`\n- **Vector Index**: Ready for questions in the 'Ask Questions' tab."
        return success_msg, transcript_text, summary_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error processing video: {str(e) or repr(e)}", "", ""

def handle_query_rewrite(question: str):
    """Rewrite question using AgentRouter deepseek-v4-flash."""
    if not question or not question.strip():
        return ""
    rewritten = rewrite_query(question.strip(), force=True)
    return rewritten

def answer_user_question(question: str, use_rewrite: bool):
    """Answer question using AgentRouter deepseek-v4-flash with retrieved evidence."""
    if not question or not question.strip():
        return "⚠️ Please enter a question.", "N/A", "No evidence retrieved."

    pipeline = get_pipeline()
    if not pipeline.retriever.is_available():
        return "⚠️ Knowledge base is empty. Please process a video in Tab 1 first.", "N/A", "No evidence retrieved."

    q = question.strip()
    if use_rewrite:
        rewritten = rewrite_query(q, force=False)
        if rewritten != q:
            q = rewritten

    result = pipeline.ask(q)

    ans = result.get("answer", "No answer generated.")
    ts = result.get("timestamp") or "N/A"
    contexts = result.get("contexts", [])

    evidence_text = ""
    for i, c in enumerate(contexts, 1):
        c_text = c.get("text", "")
        c_start = c.get("start", 0)
        c_end = c.get("end", 0)
        c_score = c.get("score", 0.0)
        evidence_text += f"**Segment {i}** `[{sec_to_time(c_start)} - {sec_to_time(c_end)}]` *(Score: {c_score:.3f})*\n>{c_text}\n\n"

    if not evidence_text:
        evidence_text = "No relevant context segments found."

    return ans, ts, evidence_text


# ──────────────────────────────────────────────────────────────
# Gradio UI Design
# ──────────────────────────────────────────────────────────────
with gr.Blocks(title="Video-QA RAG Assistant") as demo:
    with gr.Column(elem_id="app-container"):
        gr.Markdown(
            """
            # 🎬 Video-QA RAG: Evidence-Grounded Video Question Answering
            ### Powered by **DeepSeek V4 Flash** (`agentrouter.org`), **Faster-Whisper**, **BGE/E5 Embeddings** & **FAISS**
            *Zero hallucination: answers are strictly derived from lecture transcript evidence with precise timestamps.*
            """
        )

        with gr.Tabs():
            # ── TAB 1: Ingestion ──────────────────────────────────────
            with gr.TabItem("📥 1. Ingest Video / Audio"):
                gr.Markdown("Provide a video or audio file to transcribe, segment, and index into the FAISS vector database.")
                with gr.Row():
                    yt_input = gr.Textbox(
                        label="YouTube Video URL",
                        placeholder="e.g. https://youtu.be/UoccmaLwZoI",
                        info="Directly extracts audio from YouTube."
                    )
                    file_input = gr.File(
                        label="Or Upload Video / Audio File",
                        file_types=["video", "audio", ".mp4", ".mkv", ".avi", ".wav", ".mp3"],
                        type="filepath"
                    )

                with gr.Accordion("🍪 YouTube Cookies (Recommended if YouTube blocks direct link on cloud)", open=False):
                    gr.Markdown(
                        """
                        YouTube restricts cloud datacenters (like Hugging Face Spaces) with bot verification.
                        If you encounter bot detection, you can either:
                        1. **Upload your video/audio directly** in the box above (100% reliable, no cookies required), **OR**
                        2. Export your YouTube cookies using the extension [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) and provide them below:
                        """
                    )
                    with gr.Row():
                        cookie_file_input = gr.File(
                            label="Upload cookies.txt",
                            file_types=[".txt"],
                            type="filepath"
                        )
                        cookie_text_input = gr.TextArea(
                            label="Or Paste Netscape Cookie Text",
                            placeholder="# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t...",
                            lines=3
                        )

                process_btn = gr.Button("🚀 Process Video & Ingest Knowledge", variant="primary")
                status_out = gr.Markdown()

                with gr.Accordion("📜 Full Transcript (Timestamps)", open=False):
                    transcript_out = gr.TextArea(label="Transcript", interactive=False, lines=10)

                with gr.Accordion("📝 Pre-computed Lecture Summary", open=False):
                    summary_out = gr.TextArea(label="Summary", interactive=False, lines=6)

                process_btn.click(
                    fn=process_video_fn,
                    inputs=[file_input, yt_input, cookie_file_input, cookie_text_input],
                    outputs=[status_out, transcript_out, summary_out]
                )

            # ── TAB 2: Question Answering ──────────────────────────────
            with gr.TabItem("💬 2. Ask Questions (RAG QA)"):
                gr.Markdown("Ask any factual or concept question about the processed video.")
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What is the main topic discussed? What does the speaker explain about neural networks?",
                    lines=2
                )

                with gr.Row():
                    rewrite_btn = gr.Button("✨ Rewrite Query (DeepSeek V4 Flash)")
                    ask_btn = gr.Button("🔍 Search & Answer (RAG)", variant="primary")

                use_rewrite_chk = gr.Checkbox(label="Automatically rewrite query before search", value=True)
                rewritten_preview = gr.Textbox(label="Rewritten Query Preview", interactive=False, visible=False)

                rewrite_btn.click(
                    fn=lambda q: (handle_query_rewrite(q), gr.update(visible=True)),
                    inputs=[question_input],
                    outputs=[rewritten_preview, rewritten_preview]
                )

                gr.Markdown("### 🎯 Answer & Grounded Evidence")
                answer_output = gr.Textbox(label="Answer", lines=3, interactive=False)
                timestamp_output = gr.Textbox(label="Timestamp Range", interactive=False)

                with gr.Accordion("🔎 Retrieved Supporting Segments", open=True):
                    evidence_output = gr.Markdown()

                ask_btn.click(
                    fn=answer_user_question,
                    inputs=[question_input, use_rewrite_chk],
                    outputs=[answer_output, timestamp_output, evidence_output]
                )

                gr.Examples(
                    examples=[
                        ["what is the main idea of this video?"],
                        ["explain the key concepts discussed"],
                        ["what did the speaker mention in the conclusion?"]
                    ],
                    inputs=[question_input]
                )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
