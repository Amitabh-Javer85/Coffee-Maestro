"""Document-level metadata shared by PDF processing and vector ingestion."""

from pathlib import Path


def extract_document_metadata(document, filepath) -> dict[str, str]:
    metadata = document.metadata or {}
    result = {
        field: str(metadata.get(field) or "")
        for field in ("title", "author", "subject", "keywords", "creator", "creationDate", "modDate")
    }
    result["source_filename"] = Path(filepath).name
    return result
