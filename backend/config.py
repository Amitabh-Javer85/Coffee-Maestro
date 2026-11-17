import os
import math
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")


def _resolve_relative_env_path(value: str | None) -> str | None:
    if not value:
        return None

    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)

    resolved = (BACKEND_DIR / candidate).resolve()
    return str(resolved)


class Settings:
    api_secret_key: str | None = os.getenv("API_SECRET_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gcp_project_id: str | None = os.getenv("GCP_PROJECT_ID")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    answer_provider: str = os.getenv("ANSWER_PROVIDER", "gemini")
    frontend_dist_dir: str | None = _resolve_relative_env_path(os.getenv("FRONTEND_DIST_DIR"))
    conversation_project_id: str | None = os.getenv("CONVERSATION_PROJECT_ID")
    conversation_location: str | None = os.getenv("CONVERSATION_LOCATION")
    conversation_agent_id: str | None = os.getenv("CONVERSATION_AGENT_ID")
    conversation_environment_id: str | None = os.getenv("CONVERSATION_ENVIRONMENT_ID")
    conversation_language_code: str = os.getenv("CONVERSATION_LANGUAGE_CODE", "en")
    conversation_event_name: str = os.getenv("CONVERSATION_EVENT_NAME", "RAG_ANSWER")
    # Parse these only for the selected provider, so rollback ignores unused values.
    conversation_timeout_seconds = os.getenv("CONVERSATION_TIMEOUT_SECONDS", "30")
    # Conservative application bound, not a claimed Dialogflow service limit.
    conversation_max_payload_bytes = os.getenv("CONVERSATION_MAX_PAYLOAD_BYTES", "16384")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_location: str = os.getenv("EMBEDDING_LOCATION", "global")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "50"))
    context_snippet_max_length: int = int(os.getenv("CONTEXT_SNIPPET_MAX_LENGTH", "0"))
    text_top_k: int = int(os.getenv("TEXT_TOP_K", "7"))
    image_top_k: int = int(os.getenv("IMAGE_TOP_K", "3"))
    ingest_batch_size: int = int(os.getenv("INGEST_BATCH_SIZE", "50"))
    min_image_size_bytes: int = int(os.getenv("MIN_IMAGE_SIZE_BYTES", "20000"))
    diagram_capture_pages: int = int(os.getenv("DIAGRAM_CAPTURE_PAGES", "8"))
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "google-rag-index")
    google_application_credentials: str | None = _resolve_relative_env_path(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    extracted_images_dir: Path = Path(_resolve_relative_env_path(os.getenv("EXTRACTED_IMAGES_DIR")) or BACKEND_DIR / "extracted_images")
    default_pdf_path: Path = PROJECT_ROOT / "Saeco GranAroma 6850.pdf"
    cors_origins: tuple[str, ...] = (
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    def __init__(self):
        for name in ("embedding_dimensions", "chunk_size", "min_chunk_size", "ingest_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("context_snippet_max_length", "text_top_k", "image_top_k", "min_image_size_bytes", "diagram_capture_pages"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size.")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size.")

    def validate_answer_provider(self):
        """Validate only the selected transport at lazy answer initialization."""
        if self.answer_provider == "gemini":
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for ANSWER_PROVIDER=gemini.")
            return
        if self.answer_provider != "vertex_conversation":
            raise ValueError("ANSWER_PROVIDER must be gemini or vertex_conversation.")
        self.conversation_project_id = self.conversation_project_id or self.gcp_project_id
        for name in (
            "conversation_project_id", "conversation_location", "conversation_agent_id",
            "conversation_environment_id", "conversation_language_code", "conversation_event_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name.upper()} is required for vertex_conversation.")
            # These values are resource path segments or simple identifiers.
            if value != value.strip() or "/" in value or any(char.isspace() for char in value):
                raise ValueError(f"{name.upper()} must be a single identifier.")
        try:
            timeout = float(self.conversation_timeout_seconds)
            payload = int(self.conversation_max_payload_bytes)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Conversation timeout and payload bound must be positive numbers.") from None
        if not math.isfinite(timeout) or timeout <= 0 or payload <= 0:
            raise ValueError("Conversation timeout and payload bound must be positive finite numbers.")
        self.conversation_timeout_seconds = timeout
        self.conversation_max_payload_bytes = payload


settings = Settings()

if settings.google_application_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials


def resolve_ingest_path(filepath: str | None) -> Path:
    candidate = Path(filepath) if filepath else settings.default_pdf_path
    if not candidate.is_absolute():
        candidate = (BACKEND_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()

    allowed_roots = (PROJECT_ROOT.resolve(), BACKEND_DIR.resolve())
    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise ValueError("Ingestion path must stay inside the project workspace.")
    if candidate.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested.")
    return candidate
