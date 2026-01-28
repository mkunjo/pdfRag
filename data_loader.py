"""
Handles PDF loading and text embedding.
Supports multiple embedding providers: Voyage AI, OpenAI, and Gemini.
"""

import os
from openai import OpenAI
import voyageai
from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# Embedding model configurations
EMBEDDING_MODELS = {
    "voyageai": {"model": "voyage-3", "dim": 1024},
    "openai": {"model": "text-embedding-3-large", "dim": 3072},
    "gemini": {"model": "gemini-embedding-001", "dim": 3072},
}

# chunkOverlap helps maintain context between chunks
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=250)


def getEmbeddingDimension(embeddingModel: str) -> int:
    """Get the vector dimension for a given embedding model."""
    return EMBEDDING_MODELS[embeddingModel]["dim"]


def loadAndChunkPdf(path: str):
    """
    Load a PDF and split it into chunks.
    Smaller chunks work better for embedding and retrieval.
    """
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


def embedTexts(texts: list[str], embeddingModel: str = "voyageai") -> list[list[float]]:
    """
    Convert text into vector embeddings.

    Embeddings capture semantic meaning - similar texts will have similar vectors.
    This lets us search by meaning instead of just keywords.

    Args:
        texts: List of text strings to embed
        embeddingModel: One of "voyageai", "openai", or "gemini"

    Returns:
        List of embedding vectors
    """
    if embeddingModel == "voyageai":
        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        model = EMBEDDING_MODELS["voyageai"]["model"]
        result = vo.embed(texts, model=model, input_type="document")
        return result.embeddings

    elif embeddingModel == "openai":
        client = OpenAI()
        model = EMBEDDING_MODELS["openai"]["model"]
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    elif embeddingModel == "gemini":
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        model = f"models/{EMBEDDING_MODELS['gemini']['model']}"
        # Gemini embeddings API requires processing each text individually
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=model,
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    else:
        raise ValueError(f"Unknown embedding model: {embeddingModel}")