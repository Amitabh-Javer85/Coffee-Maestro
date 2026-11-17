"""Gemini Embedding 2 text/image vectors in one retrieval space."""

import math
import mimetypes
from pathlib import Path
from typing import Any

import pymupdf as fitz
from google import genai
from google.genai import types

from config import settings


class MultimodalEmbedder:
    def __init__(self, client: Any | None = None):
        self.client = client if client is not None else genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.embedding_location,
        )

    def _embed(self, contents: Any) -> list[float]:
        response = self.client.models.embed_content(
            model=settings.embedding_model,
            contents=contents,
            config={"output_dimensionality": settings.embedding_dimensions},
        )
        embeddings = response.embeddings
        if not embeddings or len(embeddings) != 1:
            raise RuntimeError("Expected one embedding from the embedding model.")
        values = embeddings[0].values
        if not values or len(values) != settings.embedding_dimensions:
            raise RuntimeError("The embedding model returned an unexpected embedding dimension.")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("The embedding model returned a nonfinite embedding value.")
        return list(values)

    def embed_text(self, text: str, task: str = "retrieval_document") -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")
        # Gemini Embedding 2 uses instruction text, not the legacy task_type.
        # https://ai.google.dev/gemini-api/docs/embeddings#task-types-with-embeddings-2
        if task == "retrieval_document":
            content = f"title: none | text: {text}"
        elif task == "search_query":
            content = f"task: search result | query: {text}"
        else:
            raise ValueError(f"Unsupported embedding task: {task}")
        return self._embed(content)

    def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        data = path.read_bytes()
        mime_type = mimetypes.guess_type(str(path))[0]
        if mime_type not in {
            "image/png", "image/jpeg", "image/webp", "image/bmp",
            "image/heic", "image/heif", "image/avif",
        }:
            # Normalize uncommon PDF image formats (such as JPEG2000) to PNG;
            # preserve the original bytes for formats supported by Vertex AI.
            pixmap = fitz.Pixmap(data)
            if pixmap.colorspace and pixmap.colorspace.n > 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            data = pixmap.tobytes("png")
            mime_type = "image/png"
        return self._embed(types.Part.from_bytes(data=data, mime_type=mime_type))

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query, task="search_query")
