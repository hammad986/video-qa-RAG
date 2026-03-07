# Video-QA RAG System — Technical Architecture & Workflow

## 1. Project Overview

**Video-QA** is a research-level Video Question Answering assistant that analyzes lecture videos and answers user questions strictly using transcript evidence. The system is designed to:

- Accept video uploads (or YouTube links)
- Transcribe the audio to text
- Structure and index the transcript for efficient retrieval
- Answer questions using only the transcript evidence
- Return **"NOT FOUND IN VIDEO"** if the answer is not present in the transcript
- Display answers with precise timestamps and supporting evidence

All answers are strictly grounded in the video transcript, with robust anti-hallucination logic.

---

## 2. System Architecture

The system is composed of modular components, each responsible for a stage in the pipeline:

1. **Video Processing**: Handles video upload, validation, and audio extraction (ffmpeg)
2. **Speech Recognition**: Transcribes audio to text using WhisperX
3. **Transcript Structuring**: Segments transcript into semantically meaningful chunks
4. **Embedding Generation**: Converts transcript chunks into dense vector embeddings (BGE/E5)
5. **Vector Search (FAISS)**: Indexes embeddings for fast similarity search
6. **Retrieval**: Finds the most relevant transcript chunks for a given question
7. **Re-Ranking**: (Optional) Improves retrieval quality using a cross-encoder
8. **Answer Generation**: Uses an LLM to generate answers strictly from retrieved evidence
9. **Evidence Verification**: NLI model checks that the answer is supported by the evidence
10. **Streamlit UI**: Provides an interactive web interface for users

**Architecture Diagram:**

```mermaid
graph TD
    A[Video Input] --> B[Audio Extraction]
    B --> C[Speech Recognition]
    C --> D[Transcript Segmentation]
    D --> E[Semantic Chunking]
    E --> F[Embedding Generation]
    F --> G[FAISS Vector Index]
    G --> H[Similarity Retrieval]
    H --> I[Context Construction]
    I --> J[Answer Generation (LLM)]
    J --> K[Evidence Verification (NLI)]
    K --> L[UI Response]
```

---

## 3. Pipeline Workflow

**Step-by-step process:**

1. **Video Input**: User uploads a video file or provides a YouTube URL via the Streamlit UI.
2. **Audio Extraction**: The system extracts audio from the video using ffmpeg.
3. **Speech Transcription**: Audio is transcribed to text using WhisperX, producing a timestamped transcript.
4. **Transcript Segmentation**: The transcript is split into logical segments (e.g., by topic or pause).
5. **Semantic Chunking**: Segments are further divided into semantically coherent chunks for retrieval.
6. **Embedding Generation**: Each chunk is embedded into a vector space using a BGE/E5 model.
7. **Vector Index Creation**: All embeddings are indexed using FAISS for efficient similarity search.
8. **Question Embedding**: When a user asks a question, it is embedded using the same model.
9. **Similarity Retrieval**: The system retrieves the top-k most similar transcript chunks from the FAISS index.
10. **Context Construction**: Retrieved chunks are assembled (with timestamps) to form the context for answering.
11. **Answer Generation**: An LLM is prompted with the context and question, instructed to answer only from evidence.
12. **Evidence Verification**: An NLI model checks if the answer is truly supported by the retrieved evidence.
13. **UI Response**: The answer, supporting evidence, and timestamps are displayed in the Streamlit UI.

---

## 4. Module Responsibilities

**video_processor.py**
: Handles video validation, audio extraction, and input management.

**speech_understanding.py**
: Runs WhisperX for speech-to-text transcription, including timestamp alignment and optional correction.

**knowledge_structuring.py**
: Segments transcripts into logical and semantic chunks for downstream retrieval.

**embeddings.py**
: Generates dense vector embeddings for transcript chunks and questions; manages FAISS index creation and storage.

**retrieval.py**
: Implements the FAISS-based retrieval system for finding relevant transcript chunks given a question embedding.

**reranker.py**
: (Optional) Uses a cross-encoder to re-rank retrieved chunks for improved answer quality.

**answer_generator.py**
: Prompts an LLM with the retrieved context and question, enforcing strict evidence-based answering.

**evidence_verifier.py**
: Uses an NLI model to verify that the generated answer is actually supported by the transcript evidence.

**pipeline.py**
: Orchestrates the entire end-to-end workflow, integrating all modules and managing state.

**query_router.py**
: Routes different types of questions (e.g., summary vs. factoid) and manages special query logic.

**app.py**
: Implements the Streamlit web interface, manages user interaction, and connects UI actions to the pipeline.

---

## 5. Retrieval-Augmented Generation Logic

- **Transcript Chunk Embeddings**: Each transcript chunk is embedded using a BGE/E5 model.
- **FAISS Vector Search**: All chunk embeddings are indexed in FAISS for fast similarity search.
- **Top-k Retrieval**: For each question, the top-k most similar chunks are retrieved.
- **Context Assembly**: Retrieved chunks (with timestamps) are concatenated to form the context for the LLM.
- **LLM Prompting**: The LLM is strictly instructed to answer only from the provided context.

---

## 6. Strict Evidence-Based Answering

- **Design Philosophy**: The system must never hallucinate or answer from general knowledge.
- **Prompt Engineering**: LLM prompts explicitly instruct the model to answer only using the transcript evidence.
- **Output Enforcement**: If the answer is not present in the retrieved context, the system outputs:

  **NOT FOUND IN VIDEO**

- **Code Enforcement**: Output is normalized and checked to ensure compliance; any deviation is overridden.

---

## 7. Timestamp Support

- **Preservation**: Timestamps are preserved throughout the pipeline, from transcription to retrieval.
- **Display**: Answers are shown with start and end timestamps, allowing users to locate the relevant video segment.
- **Evidence**: Retrieved transcript chunks are displayed with their corresponding time intervals.

---

## 8. Verification Layer

- **NLI Verification**: After answer generation, an NLI (Natural Language Inference) model checks if the answer is entailed by the retrieved evidence.
- **Anti-Hallucination**: If the NLI model determines the answer is not supported, the system returns "NOT FOUND IN VIDEO".
- **Transparency**: The UI indicates when an answer has been verified against evidence.

---

## 9. Streamlit Interface

- **Video Upload**: Users can upload videos or provide YouTube URLs.
- **Processing**: The UI guides users through video processing and displays progress.
- **Question Input**: Users can ask questions about the processed video.
- **Answer Display**: Answers are shown with timestamps and supporting evidence; source transcript chunks are viewable.
- **State Management**: UI state is robust, with tab switching and reruns handled gracefully.

---

## 10. Configuration System

- **config.yaml**: Central configuration file controlling model selection, device, chunking strategy, retrieval parameters, and more.
- **Customizable**: Developers can adjust model paths, thresholds, and pipeline options without code changes.
- **Usage**: The system loads `config.yaml` at startup and applies settings throughout the pipeline.

---

**This document provides a complete technical overview of the Video-QA RAG system architecture and workflow.**
