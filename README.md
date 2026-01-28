# PDF RAG System

A Retrieval-Augmented Generation (RAG) system that allows you to upload PDFs, ingest them into a vector database, and ask questions about their contents. Features flexible model selection for both embeddings and answer generation.

## Features

- **PDF Ingestion**: Upload PDFs through a user-friendly Streamlit interface
- **Smart Chunking**: Automatically splits documents into optimally-sized chunks with overlap for context preservation
- **Multiple Embedding Models**: Choose between Voyage AI, OpenAI, or Gemini embeddings
- **Flexible Answer Models**: Select GPT-4o-mini or Gemini 1.5 Flash for answer generation
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **AI-Powered Answers**: Leverages state-of-the-art LLMs to generate accurate answers using retrieved context
- **Background Processing**: Powered by Inngest for reliable, observable background jobs
- **Vector Storage**: Fast similarity search with Qdrant vector database
- **Model-Specific Collections**: Separate vector collections for each embedding model ensure retrieval accuracy

## Tech Stack

- **Backend**: FastAPI
- **Orchestration**: Inngest (workflow engine with automatic retries)
- **Vector Database**: Qdrant
- **Embeddings**:
  - **Voyage AI** voyage-3 (1024-dim)
  - **OpenAI** text-embedding-3-large (3072-dim)
  - **Gemini** text-embedding-004 (768-dim)
- **LLMs**: OpenAI GPT-4o-mini & Google Gemini 1.5 Flash
- **PDF Processing**: LlamaIndex
- **Frontend**: Streamlit
- **Language**: Python 3.12

## Architecture

```
User uploads PDF → Select embedding model (Voyage AI/OpenAI/Gemini)
                 ↓
         Inngest job triggered
                 ↓
         Load & chunk PDF
                 ↓
      Generate embeddings with selected model
                 ↓
      Store in model-specific Qdrant collection
      (e.g., docs_voyageai, docs_openai, docs_gemini)

User asks question → Select embedding model + answer model
                   ↓
            Embed question with selected embedding model
                   ↓
            Search matching Qdrant collection
                   ↓
         Send chunks + question to selected LLM (GPT/Gemini)
                   ↓
            Return AI-generated answer with sources
```

## Setup

### Prerequisites

- Python 3.12+
- API keys for the models you plan to use:
  - **Voyage AI** - [Get key](https://www.voyageai.com/)
  - **OpenAI** - [Get key](https://platform.openai.com/api-keys)
  - **Google Gemini** - [Get key](https://makersuite.google.com/app/apikey)
- Qdrant running locally or remotely

**Note**: You only need API keys for the models you plan to use.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mkunjo/pdfRag.git
cd pdfRag
```

2. Install dependencies:
```bash
uv sync
```

3. Create `.env` file with your API keys:
```bash
# Add only the keys you need for your chosen models
VOYAGE_API_KEY=your_voyage_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Running the Application

1. **Start the FastAPI server**:
```bash
uv run uvicorn main:app
```

2. **Start Inngest dev server**:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

3. **Start Streamlit UI**:
```bash
streamlit run streamlit_app.py
```

4. Open your browser to `http://localhost:8501`

## Usage

### Uploading a PDF

1. Open the Streamlit interface
2. **Select embedding model**: Choose Voyage AI, OpenAI, or Gemini
   - This determines how your PDF is converted to vectors
3. Click "Browse files" and select a PDF
4. Wait for ingestion to complete (you'll see a success message)

### Asking Questions

1. Type your question in the text input
2. **Select embedding model**: Must match the model used during PDF ingestion
3. **Select answer model**: Choose GPT or Gemini for answer generation
   - Answer model is independent from embedding model
4. (Optional) Adjust the number of chunks to retrieve (default: 5)
5. Click "Ask"
6. View the AI-generated answer and sources

**Important**: The embedding model for queries must match the embedding model used during ingestion.

## Project Structure

```
pdfRag/
├── main.py              # FastAPI app with Inngest functions
├── streamlit_app.py     # Streamlit UI
├── data_loader.py       # PDF loading and embedding utilities
├── vector_db.py         # Qdrant wrapper
├── custom_types.py      # Pydantic models
├── .env                 # Environment variables (not in repo)
└── uploads/             # Uploaded PDFs (created automatically)
```

## How It Works

### PDF Ingestion Flow

1. **Select Embedding Model**: User chooses Voyage AI, OpenAI, or Gemini
2. **Load**: PDF is read and text is extracted using LlamaIndex
3. **Chunk**: Text is split into 1000-character chunks with 250-character overlap
4. **Embed**: Each chunk is converted to a vector using selected embedding model:
   - Voyage AI: 1024-dimensional vector
   - OpenAI: 3072-dimensional vector
   - Gemini: 768-dimensional vector
5. **Store**: Vectors are stored in model-specific Qdrant collection (e.g., `docs_voyageai`) with metadata (source, text)
6. **ID Generation**: Uses UUID5 with embedding model name for deterministic IDs

### Query Flow

1. **Select Models**: User chooses embedding model (must match ingestion) and answer model (GPT or Gemini)
2. **Embed Question**: Convert user's question to a vector using selected embedding model
3. **Search**: Find top K most similar chunks from the matching collection using cosine similarity
4. **Build Context**: Format retrieved chunks as context for the LLM
5. **Generate Answer**: Send context + question to selected answer model (GPT-4o-mini or Gemini 1.5 Flash)
6. **Return**: Display answer with source citations

### Why Separate Collections?

Each embedding model produces vectors in different dimensional spaces with different semantic representations. Mixing them would break similarity search. That's why:
- PDFs embedded with Voyage AI are stored in `docs_voyageai`
- PDFs embedded with OpenAI are stored in `docs_openai`
- PDFs embedded with Gemini are stored in `docs_gemini`

You can have the same PDF in multiple collections if you want to compare embedding models.

## Configuration

### Environment Variables

- `VOYAGE_API_KEY`: Your Voyage AI API key (required for Voyage embeddings)
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI embeddings and/or GPT)
- `GOOGLE_API_KEY`: Your Google Gemini API key (required for Gemini embeddings and/or Gemini model)
- `INNGEST_API_BASE`: Inngest dev server URL (default: `http://127.0.0.1:8288`)

### Embedding Models

Configured in `data_loader.py` - `EMBEDDING_MODELS` dictionary:

| Model | Dimensions | Model Name |
|-------|------------|------------|
| Voyage AI | 1024 | `voyage-3` |
| OpenAI | 3072 | `text-embedding-3-large` |
| Gemini | 768 | `text-embedding-004` |

### Chunking Settings

In `data_loader.py`:
- `chunk_size`: Characters per chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 250)

### Vector Database

In `vector_db.py`:
- `url`: Qdrant server URL (default: `http://localhost:6333`)
- `collection`: Auto-generated based on embedding model:
  - `docs_voyageai` for Voyage AI
  - `docs_openai` for OpenAI
  - `docs_gemini` for Gemini
- `dim`: Auto-configured based on embedding model

## API Endpoints

The FastAPI server exposes:

- `GET /`: Root endpoint
- `POST /api/inngest`: Inngest webhook endpoint for triggering functions

## Embedding Model Comparison

| Feature | Voyage AI | OpenAI | Gemini |
|---------|-----------|--------|--------|
| **MTEB Score** | ~67.7% | ~64.6% | ~62-63% |
| **Dimensions** | 1024 | 3072 | 768 |
| **Cost/1M tokens** | $0.06 | $0.13 | Free tier, then $0.025 |
| **Optimized for** | RAG & retrieval | General purpose | Google ecosystem |

## Inngest Functions

Three background functions handle the workflow:

1. **ragIngestPdf**: Triggered by `rag/ingestPdf` event
   - Loads PDF, chunks text, generates embeddings with selected model
   - Stores in model-specific Qdrant collection

2. **ragQueryPdfGpt**: Triggered by `rag/queryPdfGpt` event
   - Embeds question with selected model, searches matching collection
   - Generates answer with GPT-4o-mini

3. **ragQueryPdfGemini**: Triggered by `rag/queryPdfGemini` event
   - Embeds question with selected model, searches matching collection
   - Generates answer with Gemini 1.5 Flash

## Development


### Adding New Features

1. Define new Pydantic models in `custom_types.py`
2. Add business logic to appropriate modules
3. Create Inngest functions in `main.py` if background processing needed
4. Update Streamlit UI in `streamlit_app.py`

## Frequently Asked Questions

### Which embedding model should I choose?

Consider your priorities:
- **Voyage AI**: Highest MTEB scores, optimized for retrieval tasks
- **OpenAI**: Strong all-around performance, widely adopted
- **Gemini**: Lower cost, good for budget-conscious projects

### Can I use different embedding and answer models?

**Yes!** The architecture separates embeddings (retrieval) from answer generation:
- Example: Voyage AI embeddings + GPT answers
- Example: OpenAI embeddings + Gemini answers
- Example: Gemini embeddings + GPT answers

This gives you flexibility to choose the best model for each task.

### Can I query PDFs with a different embedding model than I used for ingestion?

**No.** The embedding model must match between ingestion and query because:
- Each model creates vectors in different dimensional spaces
- Mixing models breaks semantic similarity search
- Each model has its own Qdrant collection

If you want to compare models, ingest the same PDF with multiple embedding models.

### How do I switch embedding models for existing PDFs?

You'll need to re-ingest your PDFs with the new embedding model. The system maintains separate collections, so you can keep both versions if you want to compare.

## Troubleshooting

### 401 API Key Errors
- Verify the correct API key is set in `.env` for the model you selected
- Voyage AI requires `VOYAGE_API_KEY`
- OpenAI requires `OPENAI_API_KEY`
- Gemini requires `GOOGLE_API_KEY`
- Restart uvicorn after updating `.env`

### "Collection not found" Errors
- Make sure you selected the same embedding model for queries as you used during ingestion
- Check that the PDF was successfully ingested (look for success message in Streamlit)
- Verify Qdrant is running: `docker ps` should show qdrant/qdrant

### Timeout Errors
- Check that Inngest dev server is running
- Verify event names match between Streamlit and main.py
- Check Qdrant is running and accessible

### Empty or Poor Quality Answers
- Try increasing the number of chunks to retrieve (topK)
- Ensure your question is clear and relates to the PDF content
- Verify the PDF was properly ingested (check Inngest dev UI)
- Consider trying a different embedding model for better retrieval




Pull requests are welcome!
