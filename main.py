"""
PDF RAG (Retrieval-Augmented Generation) system.

Workflow:
1. Ingest PDFs → chunk → embed → store in Qdrant
2. Query → search for relevant chunks → send to GPT for answer
"""

import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from google import genai
from data_loader import loadAndChunkPdf, embedTexts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

# Inngest handles background jobs with automatic retries and step-based execution
inngestClient = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngestClient.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingestPdf")
)
async def ragIngestPdf(ctx: inngest.Context):
    """
    Ingest a PDF into the RAG system.
    Triggered by sending event: {"name": "rag/ingestPdf", "data": {"pdfPath": "...", "embeddingModel": "..."}}
    """

    def load(ctx: inngest.Context) -> RAGChunkAndSrc:
        """Step 1: Load and chunk the PDF."""
        pdfPath = ctx.event.data["pdfPath"]
        sourceId = ctx.event.data.get("sourceId", pdfPath)
        chunks = loadAndChunkPdf(pdfPath)
        return RAGChunkAndSrc(chunks=chunks, sourceId=sourceId)

    def upsert(chunksAndSrc: RAGChunkAndSrc) -> RAGUpsertResult:
        """Step 2: Embed and store in Qdrant."""
        chunks = chunksAndSrc.chunks
        sourceId = chunksAndSrc.sourceId
        embeddingModel = ctx.event.data.get("embeddingModel", "voyageai")

        vecs = embedTexts(chunks, embeddingModel)

        # UUID5 is deterministic - same chunk always gets same ID
        # Include embedding model in ID to keep different embeddings separate
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sourceId}:{embeddingModel}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": sourceId, "text": chunks[i]} for i in range(len(chunks))]

        QdrantStorage(embeddingModel=embeddingModel).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunksAndSrc = await ctx.step.run("loadAndChunk", lambda: load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embedAndUpsert", lambda: upsert(chunksAndSrc), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngestClient.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/queryPdfGpt")
)
async def ragQueryPdfGpt(ctx: inngest.Context):
    """
    Answer questions using RAG pattern:
    1. Embed question → search for relevant chunks
    2. Send chunks + question to GPT for answer
    """

    def search(question: str, topK: int = 5) -> RAGSearchResult:
        """Step 1: Find relevant chunks from vector DB."""
        embeddingModel = ctx.event.data.get("embeddingModel", "voyageai")
        queryVec = embedTexts([question], embeddingModel)[0]
        store = QdrantStorage(embeddingModel=embeddingModel)
        found = store.search(queryVec, topK)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    topK = int(ctx.event.data.get("topK", 5))

    found = await ctx.step.run("embedAndSearch", lambda: search(question, topK), output_type=RAGSearchResult)

    # Build prompt with retrieved chunks
    contextBlock = "\n\n".join(f"- {c}" for c in found.contexts)
    userContent = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{contextBlock}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # OpenAI Adapter
    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # Step 2: Get answer from GPT
    res = await ctx.step.ai.infer(
        "gptAnswer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,  # Low = more focused/deterministic
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": userContent}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "numContexts": len(found.contexts)}


@inngestClient.create_function(
    fn_id="RAG: Query PDF Gemini",
    trigger=inngest.TriggerEvent(event="rag/queryPdfGemini")
)
async def ragQueryPdfGemini(ctx: inngest.Context):
    """
    Answer questions using RAG pattern with Gemini:
    1. Embed question → search for relevant chunks
    2. Send chunks + question to Gemini for answer
    """

    def search(question: str, topK: int = 5) -> RAGSearchResult:
        """Step 1: Find relevant chunks from vector DB."""
        embeddingModel = ctx.event.data.get("embeddingModel", "voyageai")
        queryVec = embedTexts([question], embeddingModel)[0]
        store = QdrantStorage(embeddingModel=embeddingModel)
        found = store.search(queryVec, topK)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    topK = int(ctx.event.data.get("topK", 5))

    found = await ctx.step.run("embedAndSearch", lambda: search(question, topK), output_type=RAGSearchResult)

    # Build prompt with retrieved chunks
    contextBlock = "\n\n".join(f"- {c}" for c in found.contexts)
    userContent = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{contextBlock}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # Step 2: Get answer from Gemini
    def generate_with_gemini():
        """Call Gemini API to generate answer."""
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        prompt = (
            "You answer questions using only the provided context.\n\n"
            f"{userContent}"
        )

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip()

    answer = await ctx.step.run("geminiAnswer", generate_with_gemini)
    return {"answer": answer, "sources": found.sources, "numContexts": len(found.contexts)}


app = FastAPI()

# Register Inngest functions - creates /api/inngest endpoint
inngest.fast_api.serve(app, inngestClient, [ragIngestPdf, ragQueryPdfGpt, ragQueryPdfGemini])