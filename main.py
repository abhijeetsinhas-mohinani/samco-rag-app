"""
RAG FastAPI Application
=======================
Production-ready FastAPI app wrapping the RAG pipeline.

Features:
  - Auto-ingest from Azure Blob on startup
  - Upload documents via API (auto-ingest)
  - Query endpoint with full RAG pipeline
  - Health check & status endpoints
  - Reset endpoint for fresh start
  - CORS enabled for web frontend integration
"""

import os
import shutil
import logging
from typing import List, Optional

import dotenv

# Load .env BEFORE importing rag_pipeline so its module-level
# os.getenv() calls pick up the configured values.
dotenv.load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGSystem, DocumentProcessor

logger = logging.getLogger("rag_fastapi")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Pipeline API",
    description="Production-ready RAG API with automatic ingestion, "
                "hybrid search, document filtering, and Ollama LLM generation.",
    version="2.0.0",
)
# CORS
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://synergy.mohinani.com",
]
# CORS — allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global RAG system instance
# ---------------------------------------------------------------------------
rag: Optional[RAGSystem] = None

# Directory for uploaded files
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/home/z/my-project/upload")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    source_type: str = "all"

class QueryResponse(BaseModel):
    answer: str
    doc_filter_active: bool
    matched_docs: List[str]
    chunks_sent: int
    query: str

class IngestResponse(BaseModel):
    status: str
    total_docs: int
    total_chunks: int
    message: str

class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    message: str

class StatusResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    faiss_vectors: int
    faiss_index_type: str
    manifest_entries: int
    embedding_model: str
    ollama_url: str
    ollama_model: str

class ResetResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Startup — Initialize RAG system and auto-ingest
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global rag
    logger.info("=" * 60)
    logger.info("  RAG FastAPI — Starting up")
    logger.info("=" * 60)

    rag = RAGSystem()

    # Try to load existing index first
    loaded = rag.load_index()
    if loaded:
        logger.info(
            f"  Loaded existing index: {rag.vector_store.ntotal} vectors, "
            f"{len(rag.documents)} chunks"
        )
    else:
        # Auto-ingest from Azure Blob if configured
        azure_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if azure_conn:
            logger.info("  Azure Blob configured — auto-ingesting...")
            try:
                rag.ingest_from_blob()
            except Exception as e:
                logger.error(f"  Auto-ingest from blob failed: {e}")
        else:
            logger.info("  No Azure Blob configured. Use /ingest-blob or /upload to add documents.")

    logger.info("=" * 60)
    logger.info("  RAG FastAPI — Ready")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Root health check."""
    return {"status": "ok", "service": "RAG Pipeline API", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    if rag is None:
        return {"status": "initializing", "message": "RAG system not ready yet"}
    return {
        "status": "ok",
        "documents": len(rag.documents),
        "vectors": rag.vector_store.ntotal if rag.vector_store else 0,
    }


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get detailed system status."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    return StatusResponse(
        status="ok",
        total_documents=rag._ingestion_stats.get("total_docs", 0),
        total_chunks=len(rag.documents),
        faiss_vectors=rag.vector_store.ntotal if rag.vector_store else 0,
        faiss_index_type=rag._index_type,
        manifest_entries=rag.manifest.count,
        embedding_model=rag.embedding_model_name,
        ollama_url=rag.ollama_base_url,
        ollama_model=rag.model_name,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """Ask a question and get an answer from the RAG pipeline."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    if not rag.documents:
        raise HTTPException(status_code=400, detail="No documents ingested. Upload or ingest first.")

    try:
        answer = rag.chat(request.question, source_type=request.source_type)

        # Get debug info from last retrieval
        debug = getattr(rag, '_last_retrieval_debug', {})

        return QueryResponse(
            answer=answer,
            doc_filter_active=debug.get("doc_filter_active", False),
            matched_docs=debug.get("matched_docs", []),
            chunks_sent=debug.get("filtered_count", 0),
            query=request.question,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/simple", tags=["Query"])
async def query_simple(
    question: str = QueryParam(..., description="Your question"),
    source_type: str = QueryParam("all", description="Filter: 'all' or 'excel'"),
):
    """Simple query endpoint — just pass the question as a query parameter."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    if not rag.documents:
        raise HTTPException(status_code=400, detail="No documents ingested.")

    try:
        answer = rag.chat(question, source_type=source_type)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and auto-ingest it into the RAG pipeline."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    # Validate file type
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    supported_exts = set(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())
    if ext not in supported_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {sorted(supported_exts)}"
        )

    # Save uploaded file
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Uploaded file saved: {upload_path} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Ingest the uploaded file
    try:
        file_chunks = rag._process_single_file(upload_path)
        if not file_chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from the file.")

        # Add chunks to the RAG system
        for chunk in file_chunks:
            chunk["chunk_id"] = len(rag.documents) + len(file_chunks)
        rag._add_chunks(file_chunks)
        rag.save_index()
        rag.manifest.mark_ingested(
            upload_path,
            len(file_chunks),
            sum(len(c.get("text", "")) for c in file_chunks)
        )
        rag.manifest.save()
        rag.chunk_writer.write_chunks(file_chunks)
        rag.chunk_writer.save_to_file()

        logger.info(f"Ingested {file.filename}: {len(file_chunks)} chunks")

        return UploadResponse(
            status="ok",
            filename=file.filename,
            chunks=len(file_chunks),
            message=f"Successfully ingested {file.filename} ({len(file_chunks)} chunks)"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/upload-batch", tags=["Ingestion"])
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple documents and auto-ingest them."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    results = []
    supported_exts = set(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())

    for file in files:
        _, ext = os.path.splitext(file.filename)
        ext = ext.lower()
        if ext not in supported_exts:
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": f"Unsupported file type: {ext}"
            })
            continue

        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        try:
            with open(upload_path, "wb") as f:
                content = await file.read()
                f.write(content)

            file_chunks = rag._process_single_file(upload_path)
            if file_chunks:
                for chunk in file_chunks:
                    chunk["chunk_id"] = len(rag.documents) + len(file_chunks)
                rag._add_chunks(file_chunks)
                rag.manifest.mark_ingested(
                    upload_path,
                    len(file_chunks),
                    sum(len(c.get("text", "")) for c in file_chunks)
                )
                results.append({
                    "filename": file.filename,
                    "status": "ok",
                    "chunks": len(file_chunks)
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "reason": "No content extracted"
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "reason": str(e)
            })

    # Save after all files
    rag.save_index()
    rag.manifest.save()
    rag.chunk_writer.save_to_file()

    return {"status": "ok", "results": results}


@app.post("/ingest-blob", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_from_blob(prefix: Optional[str] = None):
    """Ingest documents from Azure Blob Storage."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        raise HTTPException(status_code=400, detail="Azure Blob not configured. Set AZURE_STORAGE_CONNECTION_STRING.")

    try:
        rag.ingest_from_blob(prefix=prefix)

        return IngestResponse(
            status="ok",
            total_docs=rag._ingestion_stats.get("total_docs", 0),
            total_chunks=len(rag.documents),
            message=f"Ingestion complete from Azure Blob (prefix: {prefix or 'all'})"
        )
    except Exception as e:
        logger.error(f"Blob ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest-local", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_local(path: str = QueryParam(..., description="Local file or directory path")):
    """Ingest documents from a local file or directory path."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")

    try:
        rag.ingest_documents([path])

        return IngestResponse(
            status="ok",
            total_docs=rag._ingestion_stats.get("total_docs", 0),
            total_chunks=len(rag.documents),
            message=f"Ingestion complete from local path: {path}"
        )
    except Exception as e:
        logger.error(f"Local ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/reset", response_model=ResetResponse, tags=["System"])
async def reset_system():
    """Reset the RAG system — clear all indexes and start fresh."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    try:
        rag.reset()
        return ResetResponse(
            status="ok",
            message="System reset complete. All indexes, manifest, and FAISS files cleared."
        )
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/documents", tags=["System"])
async def list_documents():
    """List all ingested document source files."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    source_files = list(set(d.get("source_file", "") for d in rag.documents if d.get("source_file")))
    return {"total_sources": len(source_files), "documents": sorted(source_files)}


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
