"""
Vector database storage using Qdrant.
Qdrant stores embeddings and finds similar ones quickly - perfect for RAG.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from data_loader import getEmbeddingDimension


class QdrantStorage:
    """Wrapper for interacting with Qdrant vector database."""

    def __init__(self, url="http://localhost:6333", embeddingModel="voyageai"):
        """
        Connect to Qdrant and create collection if it doesn't exist.

        Args:
            url: Qdrant server URL
            embeddingModel: Embedding model name (voyageai, openai, or gemini)
                           Determines collection name and vector dimensions
        """
        self.client = QdrantClient(url=url, timeout=30)
        self.embeddingModel = embeddingModel

        # Use model-specific collection to keep different embeddings separate
        self.collection = f"docs_{embeddingModel}"
        dim = getEmbeddingDimension(embeddingModel)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,  # Must match embedding dimension
                    distance=Distance.COSINE  # Cosine similarity for comparing vectors
                ),
            )

    def upsert(self, ids, vectors, payloads):
        """
        Insert or update vectors in the database.
        "Upsert" = insert if new, update if exists.
        """
        # A Point combines id, vector, and metadata (payload)
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, queryVector, topK: int = 5):
        """
        Find the most similar vectors to the query.
        This is how we retrieve relevant chunks for RAG.
        """
        results = self.client.query_points(
            collection_name=self.collection,
            query=queryVector,
            limit=topK,
            with_payload=True
        )

        contexts = []
        sources = set()  # Avoid duplicates

        # results.points contains the matched points
        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}