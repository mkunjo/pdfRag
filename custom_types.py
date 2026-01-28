"""
Data models for the RAG system.
These Pydantic classes provide type safety and validation.
"""

import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    """Text chunks from a PDF with their source identifier."""
    chunks: list[str]
    sourceId: str = None


class RAGUpsertResult(pydantic.BaseModel):
    """Result after adding documents to the vector database."""
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    """Search results from the vector database."""
    contexts: list[str]  # The actual text chunks found
    sources: list[str]   # Where they came from


class RAGQueryResult(pydantic.BaseModel):
    """Final answer with sources and metadata."""
    answer: str
    sources: list[str]
    numContexts: int