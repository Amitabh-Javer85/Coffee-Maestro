from types import SimpleNamespace

from document_metadata import extract_document_metadata


def test_metadata_normalizes_absent_values_to_empty_strings():
    assert extract_document_metadata(SimpleNamespace(metadata={"title": None, "author": "A"}), "folder/manual.pdf") == {
        "title": "", "author": "A", "subject": "", "keywords": "", "creator": "",
        "creationDate": "", "modDate": "", "source_filename": "manual.pdf",
    }


def test_metadata_handles_missing_pdf_metadata():
    assert all(value == "" for key, value in extract_document_metadata(SimpleNamespace(metadata=None), "x.pdf").items() if key != "source_filename")
