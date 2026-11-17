from types import SimpleNamespace
import os
import tempfile
import unittest

from agents.multi_agent_rag import (
    ContextAnalysisAgent,
    EvidenceChunk,
    MultiAgentRAGOrchestrator,
    QualityControlAgent,
    RetrievalAgent,
    SynthesisAgent,
)


class FakeEmbedder:
    def __init__(self):
        self.requests = []

    def embed_query(self, query):
        self.requests.append(query)
        return [0.1, 0.2, 0.3]


class FakeVectorDB:
    def __init__(self, image_path="extracted_images/page_12_brew_group.png"):
        self.searches = []
        self.image_path = image_path

    def search_by_type(self, vector, content_type, top_k, metadata_filter=None):
        self.searches.append(
            {"vector": vector, "content_type": content_type, "top_k": top_k,
             "metadata_filter": metadata_filter}
        )
        if content_type == "text":
            return [
                SimpleNamespace(
                    score=0.91,
                    metadata={
                        "type": "text",
                        "page": 12,
                        "content": "Clean the brew group weekly under lukewarm water.",
                    },
                )
            ]
        return [
            SimpleNamespace(
                score=0.73,
                metadata={
                    "type": "image",
                    "page": 12,
                    "content": self.image_path,
                },
            )
        ]


class FakeGemini:
    def __init__(self):
        self.prompts = []

    def generate_content(self, contents):
        self.prompts.append(contents[0])
        return SimpleNamespace(text="Clean the brew group weekly with lukewarm water.")


class BrokenGemini:
    def generate_content(self, contents):
        raise RuntimeError("missing Gemini credentials")


class MultiAgentRAGTests(unittest.TestCase):
    def test_retrieval_agent_searches_text_and_images_separately(self):
        embedder = FakeEmbedder()
        vector_db = FakeVectorDB()
        agent = RetrievalAgent(embedder=embedder, vector_db=vector_db)

        result = agent.retrieve("How do I clean the brew group?")

        self.assertEqual(
            embedder.requests,
            ["How do I clean the brew group?"],
        )
        self.assertEqual(
            vector_db.searches,
            [
                {"vector": [0.1, 0.2, 0.3], "content_type": "text", "top_k": 7, "metadata_filter": None},
                {"vector": [0.1, 0.2, 0.3], "content_type": "image", "top_k": 3, "metadata_filter": None},
            ],
        )
        self.assertEqual([chunk.content_type for chunk in result], ["text", "image"])

    def test_orchestrator_returns_answer_images_citations_and_agent_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "page_12_brew_group.png")
            with open(image_path, "wb") as image_file:
                image_file.write(b"fake image bytes")

            orchestrator = MultiAgentRAGOrchestrator(
                retrieval_agent=RetrievalAgent(FakeEmbedder(), FakeVectorDB(image_path)),
                gemini=FakeGemini(),
            )

            response = orchestrator.ask("How do I clean the brew group?")

            self.assertEqual(
                response["answer"], "Clean the brew group weekly with lukewarm water."
            )
            self.assertEqual(response["context_images"], ["page_12_brew_group.png"])
            self.assertEqual(
                response["citations"],
                [
                    {"page": 12, "type": "text", "score": 0.91},
                    {"page": 12, "type": "image", "score": 0.73},
                ],
            )
            self.assertEqual(response["quality_warnings"], [])
            self.assertEqual(
                response["agent_trace"],
                ["retrieval", "context_analysis", "synthesis", "quality_control"],
            )

    def test_quality_control_flags_empty_or_ungrounded_answers(self):
        agent = QualityControlAgent()

        warnings = agent.review(
            query="How do I descale?",
            answer="",
            text_context="",
            citations=[],
        )

        self.assertEqual(
            warnings,
            [
                "Answer generation returned no content.",
                "No text evidence was retrieved for grounding.",
                "No citations were available for the answer.",
            ],
        )

    def test_retrieval_agent_accepts_dict_shaped_matches(self):
        class DictVectorDB:
            def search_by_type(self, vector, content_type, top_k, metadata_filter=None):
                return [
                    {
                        "score": 0.82,
                        "metadata": {
                            "type": content_type,
                            "page": 3,
                            "content": f"{content_type} evidence",
                        },
                    }
                ]

        agent = RetrievalAgent(embedder=FakeEmbedder(), vector_db=DictVectorDB())

        result = agent.retrieve("Where is the grinder setting?")

        self.assertEqual(result[0].content_type, "text")
        self.assertEqual(result[0].page, 3)
        self.assertEqual(result[0].content, "text evidence")
        self.assertEqual(result[0].score, 0.82)

    def test_retrieval_agent_dedupes_duplicate_matches_and_sorts_by_score(self):
        class DuplicateVectorDB:
            def search_by_type(self, vector, content_type, top_k, metadata_filter=None):
                if content_type == "text":
                    return [
                        {
                            "score": 0.15,
                            "metadata": {
                                "type": "text",
                                "page": 7,
                                "content": "Repeated text match",
                            },
                        },
                        {
                            "score": 0.15,
                            "metadata": {
                                "type": "text",
                                "page": 7,
                                "content": "Repeated text match",
                            },
                        },
                    ]
                return [
                    {
                        "score": 0.9,
                        "metadata": {
                            "type": "image",
                            "page": 7,
                            "content": "extracted_images/page_7_full_diagram.png",
                        },
                    }
                ]

        agent = RetrievalAgent(embedder=FakeEmbedder(), vector_db=DuplicateVectorDB())

        result = agent.retrieve("How do I clean the milk carafe?")

        self.assertEqual(len(result), 2)
        self.assertEqual([chunk.content_type for chunk in result], ["image", "text"])
        self.assertEqual(result[0].score, 0.9)

    def test_context_analysis_formats_evidence_for_grounding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "page_12_brew_group.png")
            with open(image_path, "wb") as image_file:
                image_file.write(b"fake image bytes")

            context = ContextAnalysisAgent().build_context(
                [
                    EvidenceChunk(
                        content_type="text",
                        page=12,
                        content="Clean the brew group weekly under lukewarm water.",
                        score=0.91,
                    ),
                    EvidenceChunk(
                        content_type="image",
                        page=12,
                        content=image_path,
                        score=0.73,
                    ),
                ]
            )

            self.assertIn("- Page 12 [text, score 0.910]: Clean the brew group weekly under lukewarm water.", context.text_context)
            self.assertIn("- Page 12 [image, score 0.730]: page_12_brew_group.png", context.text_context)
            self.assertEqual(context.context_images, ["page_12_brew_group.png"])
            self.assertEqual(
                context.citations,
                [
                    {"page": 12, "type": "text", "score": 0.91},
                    {"page": 12, "type": "image", "score": 0.73},
                ],
            )

    def test_synthesis_agent_raises_when_generation_fails(self):
        agent = SynthesisAgent(BrokenGemini())

        with self.assertRaises(RuntimeError) as exc:
            agent.synthesize(
                "How do I descale?",
                context=SimpleNamespace(text_context="Descale from page 20."),
            )

        self.assertIn("missing Gemini credentials", str(exc.exception))

    def test_context_keeps_full_text_when_snippet_limit_is_zero(self):
        from unittest.mock import patch

        content = "A" * 500
        with patch("agents.multi_agent_rag.settings.context_snippet_max_length", 0, create=True):
            context = ContextAnalysisAgent().build_context(
                [EvidenceChunk(content_type="text", page=4, content=content)]
            )
        self.assertIn(content, context.text_context)

    def test_context_truncates_text_to_configured_limit(self):
        from unittest.mock import patch

        with patch("agents.multi_agent_rag.settings.context_snippet_max_length", 12, create=True):
            context = ContextAnalysisAgent().build_context(
                [EvidenceChunk(content_type="text", page=4, content="A" * 40)]
            )
        snippet = context.text_context.rsplit(": ", 1)[1]
        self.assertEqual(len(snippet), 12)
        self.assertTrue(snippet.endswith("..."))

    def test_context_handles_tiny_positive_snippet_limits(self):
        from unittest.mock import patch

        for limit in (1, 2, 3):
            with self.subTest(limit=limit), patch(
                "agents.multi_agent_rag.settings.context_snippet_max_length", limit, create=True
            ):
                context = ContextAnalysisAgent().build_context(
                    [EvidenceChunk(content_type="text", page=4, content="A" * 40)]
                )
                self.assertEqual(len(context.text_context.rsplit(": ", 1)[1]), limit)

    def test_section_path_is_included_with_graceful_fallback(self):
        match = {"metadata": {"type": "text", "page": 6, "content": "Text", "section_path": "Care > Cleaning"}}
        chunk = EvidenceChunk.from_match(match)
        context = ContextAnalysisAgent().build_context([chunk])
        self.assertIn("Page 6 | Care > Cleaning [text", context.text_context)

        fallback = ContextAnalysisAgent().build_context(
            [EvidenceChunk(content_type="text", page=6, content="Text")]
        )
        self.assertIn("Page 6 [text", fallback.text_context)

    def test_retrieval_forwards_query_and_metadata_filter(self):
        filter_value = {"author": "Saeco"}
        embedder, vector_db = FakeEmbedder(), FakeVectorDB()
        result = RetrievalAgent(embedder, vector_db).retrieve("clean", metadata_filter=filter_value)
        self.assertEqual(embedder.requests, ["clean"])
        self.assertEqual([search["metadata_filter"] for search in vector_db.searches], [filter_value, filter_value])
        self.assertEqual([search["top_k"] for search in vector_db.searches], [7, 3])

    def test_retrieval_skips_disabled_modalities(self):
        vector_db = FakeVectorDB()
        result = RetrievalAgent(FakeEmbedder(), vector_db).retrieve(
            "clean", text_top_k=0, image_top_k=3
        )
        self.assertEqual([search["content_type"] for search in vector_db.searches], ["image"])
        self.assertEqual([chunk.content_type for chunk in result], ["image"])

    def test_retrieval_can_disable_both_modalities(self):
        vector_db = FakeVectorDB()
        result = RetrievalAgent(FakeEmbedder(), vector_db).retrieve(
            "clean", text_top_k=0, image_top_k=0
        )
        self.assertEqual(vector_db.searches, [])
        self.assertEqual(result, [])

    def test_orchestrator_passes_metadata_filter_to_retrieval(self):
        class RecordingRetrieval:
            def __init__(self):
                self.metadata_filter = None

            def retrieve(self, query, metadata_filter=None):
                self.metadata_filter = metadata_filter
                return []

        retrieval = RecordingRetrieval()
        orchestrator = MultiAgentRAGOrchestrator(retrieval, FakeGemini())
        filter_value = {"title": "Manual"}
        orchestrator.ask("question", metadata_filter=filter_value)
        self.assertEqual(retrieval.metadata_filter, filter_value)


if __name__ == "__main__":
    unittest.main()
