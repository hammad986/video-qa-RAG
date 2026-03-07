import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import os

INDEX_PATH = "models/video_index.faiss"
META_PATH = "models/metadata.pkl"

embed_model = None
index = None
metadata = None


def load_resources():
    global embed_model, index, metadata

    if embed_model is None:
        print("Initializing embedding model (first time only)...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    if index is None:
        if not os.path.exists(INDEX_PATH):
            print("No vector index found. Please process a video first.")
            return False
        print("Loading FAISS index...")
        index = faiss.read_index(INDEX_PATH)

    if metadata is None:
        if not os.path.exists(META_PATH):
            print("Metadata missing. Please process a video first.")
            return False
        metadata = joblib.load(META_PATH)

    return True


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def search(query, top_k=5, min_score=0.30):

    # ensure resources exist
    if not load_resources():
        return None

    # create query embedding
    query_vec = embed_model.encode([query])
    query_vec = np.array(query_vec).astype("float32")
    query_vec = normalize(query_vec)

    # FAISS search
    scores, indices = index.search(query_vec, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        # FAISS may return -1
        if idx == -1:
            continue

        # score is cosine similarity already (0 to 1 approx)
        if score < min_score:
            continue

        results.append(metadata[idx])

    if len(results) == 0:
        return None

    return results