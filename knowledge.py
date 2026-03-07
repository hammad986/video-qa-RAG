"""
Knowledge Manager - Wikipedia Retrieval and Vector Database
Automatically learns new topics and stores them offline
"""

import os
import json
import wikipediaapi
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """Manages knowledge base with Wikipedia learning and ChromaDB storage."""

    def __init__(self):
        self.db_path = "db"
        self.collection_name = "knowledge_base"

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path, settings=Settings(anonymized_telemetry=False))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(name=self.collection_name)

        # Initialize embedding model
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='AI-Trustworthiness-System/1.0'
        )

        logger.info("Knowledge Manager initialized")

    def get_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        try:
            count = self.collection.count()
            return {"documents": count, "chunks": count}
        except:
            return {"documents": 0, "chunks": 0}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge base for relevant documents."""

        # Check if we have any documents
        if self.collection.count() == 0:
            # Try to learn about the topic
            self._learn_topic(query)
            if self.collection.count() == 0:
                return []

        # Embed query
        query_embedding = self.embed_model.encode([query])[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas']
        )

        # Format results
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append({
                    'page_content': doc,
                    'metadata': metadata
                })

        return documents

    def _learn_topic(self, topic: str):
        """Learn about a topic from Wikipedia and store in database."""

        try:
            # Search for the topic
            page = self.wiki.page(topic)

            if not page.exists():
                # Try a more general search
                search_results = self.wiki.search(topic, results=1)
                if search_results:
                    page = self.wiki.page(search_results[0])

            if page.exists():
                # Get page content
                content = page.text

                # Clean and chunk the content
                chunks = self._chunk_text(content, chunk_size=500, overlap=50)

                # Embed chunks
                embeddings = self.embed_model.encode(chunks)

                # Store in database
                ids = [f"{topic}_{i}" for i in range(len(chunks))]
                metadatas = [{"topic": topic, "source": "wikipedia", "url": page.fullurl} for _ in chunks]

                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )

                logger.info(f"Learned about topic: {topic} ({len(chunks)} chunks)")

        except Exception as e:
            logger.error(f"Failed to learn about topic {topic}: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(' '.join(chunk))

        return chunks

    def find_related_video(self, query: str) -> Optional[Dict[str, str]]:
        """Find related educational video (placeholder - would integrate with YouTube API)."""

        # This is a placeholder - in a real implementation, you'd search YouTube
        # For now, return a sample video
        return {
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Related Educational Video"
        }

    def get_source_url(self, query: str) -> Optional[str]:
        """Get source URL for a query."""

        results = self.search(query, top_k=1)
        if results:
            return results[0].get('metadata', {}).get('url')

        return None
