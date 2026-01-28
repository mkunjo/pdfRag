"""
Streamlit UI for the RAG system.
Two main features:
1. Upload PDFs to ingest into the vector database
2. Ask questions and get AI-generated answers
"""

from pathlib import Path
import time

import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def inngestEventUrl() -> str:
    """Get Inngest dev server event URL."""
    base = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288")
    return f"{base}/e/rag_app"


def saveUploadedPdf(file) -> Path:
    """Save uploaded file to local 'uploads' directory."""
    uploadsDir = Path("uploads")
    uploadsDir.mkdir(parents=True, exist_ok=True)
    filePath = uploadsDir / file.name
    fileBytes = file.getbuffer()
    filePath.write_bytes(fileBytes)
    return filePath


def sendRagIngestEvent(pdfPath: Path, embeddingModel: str) -> str:
    """Trigger the PDF ingestion background job via HTTP."""
    payload = {
        "name": "rag/ingestPdf",
        "data": {
            "pdfPath": str(pdfPath.resolve()),
            "sourceId": pdfPath.name,
            "embeddingModel": embeddingModel,
        }
    }
    resp = requests.post(inngestEventUrl(), json=payload)
    resp.raise_for_status()
    return resp.json()["ids"][0]  # Event ID


# Embedding Model Selection (applies to both ingestion and querying)
st.title("PDF RAG System")

embeddingModelDisplay = st.radio(
    "Embedding Model (used for both upload and search)",
    options=["Voyage AI (Recommended)", "OpenAI", "Gemini"],
    index=0,
    horizontal=True,
    help="This determines which collection your PDFs are stored in and searched from"
)
# Map display name to internal name
embeddingModelMap = {
    "Voyage AI (Recommended)": "voyageai",
    "OpenAI": "openai",
    "Gemini": "gemini"
}
selectedEmbeddingModel = embeddingModelMap[embeddingModelDisplay]

st.divider()

# PDF Upload Section
st.subheader("Upload a PDF to Ingest")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = saveUploadedPdf(uploaded)
        sendRagIngestEvent(path, selectedEmbeddingModel)  # Trigger Inngest function
    st.success(f"Triggered ingestion for: {path.name} using {embeddingModelDisplay}")
    st.caption("You can upload another PDF if you like.")

st.divider()

# Query Section
st.subheader("Ask a question about your PDFs")


def sendRagQueryEvent(question: str, topK: int, embeddingModel: str) -> str:
    """
    Send question to Inngest and return the event ID.
    We'll use the ID to poll for results.
    """
    payload = {
        "name": "rag/queryPdfGpt",
        "data": {
            "question": question,
            "topK": topK,
            "embeddingModel": embeddingModel,
        }
    }
    resp = requests.post(inngestEventUrl(), json=payload)
    resp.raise_for_status()
    return resp.json()["ids"][0]  # Event ID


def sendRagQueryGeminiEvent(question: str, topK: int, embeddingModel: str) -> str:
    """
    Send question to Inngest for Gemini and return the event ID.
    We'll use the ID to poll for results.
    """
    payload = {
        "name": "rag/queryPdfGemini",
        "data": {
            "question": question,
            "topK": topK,
            "embeddingModel": embeddingModel,
        }
    }
    resp = requests.post(inngestEventUrl(), json=payload)
    resp.raise_for_status()
    return resp.json()["ids"][0]  # Event ID


def inngestApiBase() -> str:
    """Get Inngest dev server API URL (local by default)."""
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetchRuns(eventId: str) -> list[dict]:
    """Get all runs associated with an event from Inngest API."""
    url = f"{inngestApiBase()}/events/{eventId}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def waitForRunOutput(eventId: str, timeoutS: float = 120.0, pollIntervalS: float = 0.5) -> dict:
    """
    Poll Inngest until the function run completes and return its output.
    This is how we wait for the answer to be generated.
    """
    start = time.time()
    lastStatus = None
    while True:
        runs = fetchRuns(eventId)
        if runs:
            run = runs[0]
            status = run.get("status")
            lastStatus = status or lastStatus
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeoutS:
            raise TimeoutError(f"Timed out waiting for run output (last status: {lastStatus})")
        time.sleep(pollIntervalS)


with st.form("ragQueryForm"):
    question = st.text_input("Your question")

    answerModel = st.radio(
        "Answer model",
        options=["GPT", "Gemini"],
        index=0,
        horizontal=True,
        help="Which LLM should generate the answer (independent from embedding model)"
    )

    topK = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Trigger the appropriate query function based on model selection
            if answerModel == "GPT":
                eventId = sendRagQueryEvent(question.strip(), int(topK), selectedEmbeddingModel)
            else:  # Gemini
                eventId = sendRagQueryGeminiEvent(question.strip(), int(topK), selectedEmbeddingModel)

            # Wait for Inngest to complete the run and return output
            output = waitForRunOutput(eventId)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")