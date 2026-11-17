"""Heading-aware PDF extraction with source-page provenance."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pymupdf as fitz
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import settings
from document_metadata import extract_document_metadata


@dataclass
class ProcessedDocument:
    chunks: list[dict[str, Any]]
    metadata: dict[str, str]


class DocumentProcessor:
    """Dispatch supported formats to extractors returning one ingest contract."""

    def __init__(self, output_image_dir: str | Path | None = None):
        self.output_image_dir = Path(output_image_dir or settings.extracted_images_dir)

    def process(self, filepath: str | Path) -> ProcessedDocument:
        path = Path(filepath)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported document format: {path.suffix or '(none)'}")
        return self._process_pdf(path)

    def chunk_markdown_pages(self, pages: list[str]) -> list[dict[str, Any]]:
        """Split the complete document; markers carry pages through heading splits.

        Whitespace is the final split boundary, so indivisible words longer than
        chunk_size remain intact. Overlap consists of whole words up to the
        configured overlap size. A heading boundary starts a fresh section.
        """
        if not any(page.strip() for page in pages):
            return []
        # Comments survive the heading splitter but are removed before embedding.
        # They avoid matching repeated text back to the wrong physical page.
        marker = re.compile(r"<!-- rag-source-page:(\d+) -->\s*")
        markdown = "\n\n".join(
            f"<!-- rag-source-page:{number} -->\n\n{text}"
            for number, text in enumerate(pages, start=1)
        )
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            # Preserve original whitespace and never fall back to characters.
            separators=[r"\s+"],
            is_separator_regex=True,
            keep_separator="end",
            add_start_index=True,
        )
        chunks = []
        current_page = 1
        for section in header_splitter.split_text(markdown):
            pieces = []
            spans = []
            cursor = 0
            clean_length = 0
            for match in marker.finditer(section.page_content):
                piece = section.page_content[cursor:match.start()]
                pieces.append(piece)
                if piece:
                    spans.append((clean_length, clean_length + len(piece), current_page))
                    clean_length += len(piece)
                current_page = int(match.group(1))
                cursor = match.end()
            piece = section.page_content[cursor:]
            pieces.append(piece)
            if piece:
                spans.append((clean_length, clean_length + len(piece), current_page))
            content = "".join(pieces)
            headings = {key: section.metadata[key] for key in ("h1", "h2", "h3") if key in section.metadata}
            for split in splitter.create_documents([content]):
                text = split.page_content
                if len(text) < settings.min_chunk_size:
                    continue
                offset = split.metadata["start_index"]
                page = next((page for start, end, page in spans if start <= offset < end), current_page)
                chunks.append({
                    "type": "text",
                    "page": page,
                    "content": text,
                    "headings": dict(headings),
                    "section_path": " > ".join(headings.values()),
                    "chunk_index": len(chunks),
                })
        return chunks

    def _process_pdf(self, pdf_path: Path) -> ProcessedDocument:
        with fitz.open(pdf_path) as document:
            metadata = extract_document_metadata(document, pdf_path)
            # Join authoritative physical pages before heading/size splitting.
            pages = pymupdf4llm.to_markdown(document, page_chunks=True)
            chunks = self.chunk_markdown_pages([page["text"] for page in pages])
            self.output_image_dir.mkdir(parents=True, exist_ok=True)
            for page_number, page in enumerate(document, start=1):
                if page_number <= settings.diagram_capture_pages:
                    image_path = self.output_image_dir / f"page_{page_number}_full_diagram.png"
                    page.get_pixmap(dpi=150).save(image_path)
                    chunks.append({"type": "image", "page": page_number, "content": str(image_path), "id": f"p{page_number}_illustration"})
                for image_index, image in enumerate(page.get_images(full=True)):
                    extracted = document.extract_image(image[0])
                    if len(extracted["image"]) < settings.min_image_size_bytes:
                        continue
                    image_path = self.output_image_dir / f"page_{page_number}_img_{image_index}.{extracted['ext']}"
                    image_path.write_bytes(extracted["image"])
                    chunks.append({"type": "image", "page": page_number, "content": str(image_path), "id": f"p{page_number}_img{image_index}"})
        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
        return ProcessedDocument(chunks=chunks, metadata=metadata)


def extract_text_and_images(pdf_path: str, output_image_dir: str | None = None) -> list[dict[str, Any]]:
    """Compatibility wrapper for local diagnostic scripts."""
    return DocumentProcessor(output_image_dir).process(pdf_path).chunks
