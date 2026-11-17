import sys
import os

import logging
from error_logging import configure_error_logging

configure_error_logging()
import secrets
import hashlib
import tempfile
import threading
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Security, Depends, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from pythonjsonlogger import jsonlogger

logger = logging.getLogger("backend")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from config import resolve_ingest_path, settings
from agents.multi_agent_rag import MultiAgentRAGOrchestrator, RetrievalAgent, SynthesisAgent, GeminiAnswerProvider

app = FastAPI(title="Google RAG API", description="Production RAG application for multimodal search")
# Initialize rate limiter (10 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=settings.extracted_images_dir), name="images")

# CORS setup - Hardened for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Add common security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Initialize heavy dependencies lazily to speed up startup
_embedder = None
_vector_db = None
_gemini = None
_rag_orchestrator = None
_rag_initialization_lock = threading.Lock()

def get_embedder():
    global _embedder
    if _embedder is None:
        from embedder import MultimodalEmbedder

        _embedder = MultimodalEmbedder()
    return _embedder

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        from pinecone_client import VectorDBClient

        _vector_db = VectorDBClient()
    return _vector_db

def get_gemini():
    global _gemini
    if _gemini is None:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required to generate answers.")
        from google import genai

        class GeminiGenerator:
            def __init__(self, api_key: str, model_name: str):
                self.client = genai.Client(api_key=api_key)
                self.model_name = model_name

            def generate_content(self, prompt: str):
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

        _gemini = GeminiGenerator(settings.google_api_key, settings.gemini_model)
    return _gemini

def get_answer_provider():
    settings.validate_answer_provider()
    if settings.answer_provider == "vertex_conversation":
        from conversation_client import ConversationAnswerClient

        return ConversationAnswerClient(settings)
    return GeminiAnswerProvider(get_gemini())


def get_rag_orchestrator():
    global _rag_orchestrator
    with _rag_initialization_lock:
        if _rag_orchestrator is None:
            provider = get_answer_provider()
            _rag_orchestrator = MultiAgentRAGOrchestrator(
                retrieval_agent=RetrievalAgent(
                    embedder=get_embedder(),
                    vector_db=get_vector_db(),
                ),
                synthesis_agent=SynthesisAgent(answer_provider=provider),
            )
    return _rag_orchestrator

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def is_local_request(request: Request) -> bool:
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    return bool(request.client and request.client.host in loopback_hosts
                and request.url.hostname in loopback_hosts)


def get_api_key(request: Request, provided_api_key: str = Security(api_key_header)):
    if not settings.api_secret_key:
        raise HTTPException(status_code=500, detail="API_SECRET_KEY is not configured")
    if provided_api_key and secrets.compare_digest(provided_api_key, settings.api_secret_key):
        return provided_api_key
    if is_local_request(request):
        return provided_api_key or "local-request"
    raise HTTPException(status_code=403, detail="Could not validate API KEY")

class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None

class IngestRequest(BaseModel):
    filepath: str | None = None


class ResetRequest(BaseModel):
    confirm: bool = False


class DatasetBusyError(RuntimeError):
    """Another writer owns this process's active dataset."""


_dataset_mutation_lock = threading.Lock()


@contextmanager
def _dataset_mutation():
    # The desktop launcher uses one backend process. Other workers or external
    # indexing scripts require shared coordination, not a separate local lock.
    if not _dataset_mutation_lock.acquire(blocking=False):
        raise DatasetBusyError(
            "An ingestion or reset is already running. Wait for it to finish, then retry."
        )
    try:
        yield
    finally:
        _dataset_mutation_lock.release()


def _vector_id(source_filename: str, chunk_index: int, content_type: str) -> str:
    key = f"{source_filename}::{content_type}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def build_vector(item: dict, embedder: Any, doc_metadata: dict) -> dict:
    headings = item.get("headings", {})
    metadata = {
        "type": item["type"],
        "page": item["page"],
        "content": item["content"],
        **{key: str(doc_metadata.get(key) or "") for key in (
            "source_filename", "title", "author", "keywords", "subject", "creator", "creationDate", "modDate"
        )},
        "section_path": item.get("section_path") or "",
        "h1": headings.get("h1") or "",
        "h2": headings.get("h2") or "",
        "h3": headings.get("h3") or "",
        "chunk_index": item["chunk_index"],
    }
    if item["type"] == "text":
        embedding = embedder.embed_text(item["content"])
    elif item["type"] == "image":
        embedding = embedder.embed_image(item["content"])
    else:
        raise ValueError(f"Unsupported chunk type: {item['type']}")

    if not embedding:
        raise ValueError("Embedding model returned an empty vector.")
    return {
        "id": _vector_id(metadata["source_filename"], item["chunk_index"], item["type"]),
        "values": embedding,
        "metadata": metadata,
    }


def ingest_pdf(pdf_path: Path) -> dict:
    with _dataset_mutation():
        return _ingest_pdf(pdf_path)


def _ingest_pdf(pdf_path: Path) -> dict:
    from document_processor import DocumentProcessor

    # Keep old image references valid during preparation and after failures.
    # Retain generations: readers and partial upserts may still reference them.
    settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
    image_directory = Path(tempfile.mkdtemp(prefix="ingest-", dir=settings.extracted_images_dir))
    processed = DocumentProcessor(output_image_dir=image_directory).process(pdf_path)
    extracted_data = processed.chunks
    if not extracted_data:
        raise ValueError("Document produced no chunks; existing vectors were preserved.")
    embedder = get_embedder()
    vector_db = get_vector_db()
    batch_size = settings.ingest_batch_size
    total_upserted = 0

    logger.info("ingest_started", extra={"item_count": len(extracted_data)})

    # Prepare the replacement completely before clearing the active dataset.
    # Extraction or model errors must never erase working retrieval data.
    vectors = [build_vector(item, embedder, processed.metadata) for item in extracted_data]
    vector_db.delete_all()
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        vector_db.upsert(batch)
        total_upserted += len(batch)
        logger.info(
            "batch_upserted",
            extra={"batch": i // batch_size + 1, "total_upserted": total_upserted},
        )

    return {
        "message": f"Successfully ingested {total_upserted} chunks.",
        "upserted": total_upserted,
        "failed": 0,
        "metadata": processed.metadata,
    }


def reset_dataset() -> dict:
    with _dataset_mutation():
        get_vector_db().delete_all()
        return {"message": "Index cleared successfully."}


@limiter.limit("10/minute")
@app.post("/api/ingest")
async def ingest_document(request: Request, req: IngestRequest, api_key: str = Depends(get_api_key)):
    """Parses a local PDF, embeds it, and upserts to Pinecone."""
    try:
        pdf_path = resolve_ingest_path(req.filepath)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        return await run_in_threadpool(ingest_pdf, pdf_path)
    except DatasetBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ingest_failed", extra={"path": str(pdf_path)})
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc

@limiter.limit("10/minute")
@app.post("/api/ask")
async def ask_question(request: Request, query_req: QueryRequest, api_key: str = Depends(get_api_key)):
    """Answers a question through the multi-agent RAG orchestration pipeline."""
    try:
        orchestrator = await run_in_threadpool(get_rag_orchestrator)
        return await run_in_threadpool(orchestrator.ask, query_req.query, metadata_filter=query_req.metadata_filter)
    except Exception as exc:
        correlation_id = str(uuid.uuid4())
        status_code = 502
        public_message = "Could not generate an answer. Please retry or contact the administrator."
        # Import only the lightweight error type when handling errors. Never
        # return arbitrary upstream exception text, which can contain evidence.
        from conversation_client import ConversationError
        if isinstance(exc, ConversationError):
            status_code = exc.status_code
            public_message = exc.public_message
        # Frame locations preserve useful tracebacks without source lines, local
        # variables, exception text, or chained exceptions that may contain data.
        frames = "\n".join(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
            for frame in traceback.extract_tb(exc.__traceback__)
        )
        logger.error(
            "ask_failed provider=%s correlation_id=%s error_type=%s status_code=%s safe_message=%s\nTraceback (sanitized):\n%s",
            settings.answer_provider, correlation_id, type(exc).__name__, status_code, public_message, frames,
        )
        raise HTTPException(
            status_code=status_code,
            detail=f"{public_message} Reference: {correlation_id}",
            headers={"X-Correlation-ID": correlation_id},
        ) from None

@limiter.limit("2/minute")
@app.post("/api/reset")
async def reset_index(request: Request, req: ResetRequest, api_key: str = Depends(get_api_key)):
    """Deletes all existing vectors in Pinecone."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear the index.")
    try:
        return await run_in_threadpool(reset_dataset)
    except DatasetBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("reset_failed")
        raise HTTPException(status_code=502, detail=f"Reset failed: {exc}") from exc

# --- Production Static Rendering ---
# This serves the React frontend (dist) at the root URL
# Note: This is placed AFTER all other API routes to ensure they are handled first
# We check for both ../frontend/dist (source repo) and ./dist (packaged)
current_dir = Path(__file__).resolve().parent

if settings.frontend_dist_dir:
    dist_path = Path(settings.frontend_dist_dir)
elif getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    dist_path = Path(sys._MEIPASS) / "dist"
else:
    dist_path = (current_dir / "../../frontend/dist").resolve()
    if not dist_path.exists():
        dist_path = (current_dir / "../frontend/dist").resolve()
    
if dist_path.exists():
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="frontend")
    logger.info("frontend_mounted", extra={"path": str(dist_path)})
else:
    logger.warning("frontend_dist_missing", extra={"path": str(dist_path)})

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    
    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")
        
    # Open the browser shortly after the server starts
    if os.environ.get("COFFEE_MAESTRO_OPEN_BROWSER", "1") != "0":
        threading.Timer(1.5, open_browser).start()
    
    # Use 127.0.0.1 to avoid firewall prompts for local-only app
    uvicorn.run(app, host="127.0.0.1", port=8000)
