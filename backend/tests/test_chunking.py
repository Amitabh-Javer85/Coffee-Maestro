"""Offline integration coverage for the manual and controlled PDF fixtures."""

from pathlib import Path
import re
import sys

import pymupdf as fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from document_processor import DocumentProcessor


@pytest.fixture(scope="module")
def manual(tmp_path_factory):
    source = Path(__file__).resolve().parents[2] / "Saeco GranAroma 6850.pdf"
    if not source.is_file():
        pytest.skip("A locally supplied manual is required for this test")
    images = tmp_path_factory.mktemp("manual-images")
    return DocumentProcessor(output_image_dir=images).process(source)


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 100)
    monkeypatch.setattr(settings, "chunk_overlap", 30)
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    monkeypatch.setattr(settings, "diagram_capture_pages", 0)


def text_chunks(result):
    return [chunk for chunk in result.chunks if chunk["type"] == "text"]


def test_manual_conversion_retains_markdown_headings(manual):
    assert any(re.search(r"(?m)^#{1,3} ", chunk["content"]) for chunk in text_chunks(manual))


def test_manual_chunks_respect_size_except_indivisible_words(manual):
    assert all(len(chunk["content"]) <= max(settings.chunk_size, max(map(len, chunk["content"].split()))) for chunk in text_chunks(manual))


def test_manual_has_heading_metadata(manual):
    assert any(chunk["headings"] and chunk["section_path"] for chunk in text_chunks(manual))


def test_manual_has_source_filename(manual):
    assert manual.metadata["source_filename"] == "Saeco GranAroma 6850.pdf"


def test_manual_has_reasonable_chunk_count(manual):
    assert 2 < len(text_chunks(manual)) < 10000


def test_every_chunk_has_a_unique_sequential_index(manual):
    assert [chunk["chunk_index"] for chunk in manual.chunks] == list(range(len(manual.chunks)))


def test_manual_chunks_have_valid_page_numbers(manual):
    with fitz.open(Path(__file__).resolve().parents[2] / "Saeco GranAroma 6850.pdf") as document:
        page_count = len(document)
    assert all(1 <= chunk["page"] <= page_count for chunk in manual.chunks)


def test_manual_contains_overlapping_section_chunks(manual):
    chunks = text_chunks(manual)
    assert any(
        left["headings"] == right["headings"]
        and any(left["content"].endswith(right["content"][:length]) for length in range(20, settings.chunk_overlap + 1))
        for left, right in zip(chunks, chunks[1:])
    )


def test_oversized_section_overlaps_on_complete_words(small_chunks):
    words = [f"word{number:03d}" for number in range(90)]
    chunks = DocumentProcessor().chunk_markdown_pages(["# Heading\n\n" + " ".join(words)])
    bodies = [chunk["content"] for chunk in chunks]
    assert all(any(left.endswith(right[:length]) for length in range(1, 31)) for left, right in zip(bodies, bodies[1:]))


def test_splitter_does_not_cut_words(small_chunks):
    words = [f"word{number:03d}" for number in range(90)]
    chunks = DocumentProcessor().chunk_markdown_pages([" ".join(words)])
    assert all(word in words for chunk in chunks for word in chunk["content"].split())


def test_cross_page_section_is_one_chunk_when_it_fits(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    chunks = DocumentProcessor().chunk_markdown_pages(["# Care\n\nFirst page instructions.", "Second page continuation."])
    assert len(chunks) == 1 and "Second page continuation." in chunks[0]["content"]


def test_cross_page_section_uses_starting_page(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    chunks = DocumentProcessor().chunk_markdown_pages(["# Care\n\nFirst page instructions.", "Second page continuation."])
    assert chunks[0]["page"] == 1


def test_later_heading_is_attributed_to_its_actual_page(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    chunks = DocumentProcessor().chunk_markdown_pages(["# First\n\nEarlier content.", "# Second\n\nLater content."])
    assert next(chunk for chunk in chunks if chunk["headings"].get("h1") == "Second")["page"] == 2


def test_repeated_page_text_keeps_page_provenance(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    chunks = DocumentProcessor().chunk_markdown_pages(["# First\n\nSame repeated words.", "# Second\n\nSame repeated words."])
    assert [chunk["page"] for chunk in chunks] == [1, 2]


def test_nested_headings_build_section_path(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    chunks = DocumentProcessor().chunk_markdown_pages(["# Care\n## Cleaning\n### Weekly\n\nClean the brew group."])
    assert chunks[-1]["section_path"] == "Care > Cleaning > Weekly"


def test_short_sections_are_discarded(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 50)
    assert DocumentProcessor().chunk_markdown_pages(["Short."]) == []


def test_empty_pages_produce_no_text_chunks():
    assert DocumentProcessor().chunk_markdown_pages(["", "   "]) == []


def test_content_on_second_page_is_not_attributed_to_empty_first_page(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    assert DocumentProcessor().chunk_markdown_pages(["", "Second page only."])[0]["page"] == 2


def test_table_and_list_markdown_remain_in_the_section(monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    markdown = "# Cleaning\n\n| Item | Frequency |\n| --- | --- |\n| Brew group | Weekly |\n\n- Rinse thoroughly\n- Leave to dry"
    chunks = DocumentProcessor().chunk_markdown_pages([markdown])
    assert all(part in chunks[0]["content"] for part in ["| Brew group | Weekly |", "- Rinse thoroughly", "- Leave to dry"])


def test_page_after_a_cross_page_chunk_uses_its_own_provenance(small_chunks):
    chunks = DocumentProcessor().chunk_markdown_pages([" ".join(f"first{n:03d}" for n in range(35)), " ".join(f"second{n:03d}" for n in range(60))])
    assert all(chunk["page"] == 2 for chunk in chunks if chunk["content"].startswith("second"))


def test_indivisible_long_words_are_retained(small_chunks):
    word = "x" * 150
    assert any(word in chunk["content"] for chunk in DocumentProcessor().chunk_markdown_pages([word]))


def test_processor_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentProcessor(output_image_dir=tmp_path).process(tmp_path / "notes.txt")


def test_constructed_pdf_preserves_page_metadata_and_diagram(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "min_chunk_size", 1)
    monkeypatch.setattr(settings, "diagram_capture_pages", 1)
    filepath = tmp_path / "constructed.pdf"
    with fitz.open() as document:
        document.set_metadata({"title": "Controlled manual", "author": "Test Author"})
        for heading, body in [("FIRST SECTION", "Unique first page text."), ("SECOND SECTION", "Unique second page text.")]:
            page = document.new_page()
            page.insert_text((72, 72), heading, fontsize=22)
            page.insert_text((72, 110), body, fontsize=11)
        document.save(filepath)
    result = DocumentProcessor(output_image_dir=tmp_path / "images").process(filepath)
    assert result.metadata["title"] == "Controlled manual" and any(chunk["type"] == "image" and Path(chunk["content"]).is_file() for chunk in result.chunks)


@pytest.mark.parametrize("minimum_bytes, expected_images", [(1, 1), (1000000000, 0)])
def test_embedded_image_threshold_is_configurable(tmp_path, monkeypatch, minimum_bytes, expected_images):
    monkeypatch.setattr(settings, "diagram_capture_pages", 0)
    monkeypatch.setattr(settings, "min_image_size_bytes", minimum_bytes)
    filepath = tmp_path / "image.pdf"
    with fitz.open() as canvas:
        canvas.new_page(width=20, height=20)
        image_bytes = canvas[0].get_pixmap().tobytes("png")
    with fitz.open() as document:
        page = document.new_page()
        page.insert_image(fitz.Rect(72, 72, 172, 172), stream=image_bytes)
        document.save(filepath)
    result = DocumentProcessor(output_image_dir=tmp_path / "images").process(filepath)
    assert sum(chunk["type"] == "image" for chunk in result.chunks) == expected_images
