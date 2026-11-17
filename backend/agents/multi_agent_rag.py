import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import settings


@dataclass(frozen=True)
class EvidenceChunk:
    content_type: str
    page: Optional[int]
    content: str
    score: Optional[float] = None
    section_path: Optional[str] = None

    @classmethod
    def from_match(cls, match: Any) -> "EvidenceChunk":
        if isinstance(match, dict):
            metadata = match.get("metadata") or {}
            score = match.get("score")
        else:
            metadata = getattr(match, "metadata", None) or {}
            score = getattr(match, "score", None)

        return cls(
            content_type=metadata.get("type", "unknown"),
            page=metadata.get("page"),
            content=metadata.get("content", ""),
            score=score,
            section_path=metadata.get("section_path"),
        )

    def citation(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "type": self.content_type,
            "score": self.score,
        }


@dataclass(frozen=True)
class ContextPackage:
    text_context: str
    context_images: List[str]
    citations: List[Dict[str, Any]]


class RetrievalAgent:
    def __init__(
        self,
        embedder: Any,
        vector_db: Any,
        text_top_k: Optional[int] = None,
        image_top_k: Optional[int] = None,
    ):
        self.embedder = embedder
        self.vector_db = vector_db
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k

    def retrieve(
        self,
        query: str,
        text_top_k: Optional[int] = None,
        image_top_k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> List[EvidenceChunk]:
        query_embedding = self.embedder.embed_query(query)
        resolved_text_top_k = next(
            (value for value in (text_top_k, self.text_top_k) if value is not None),
            settings.text_top_k,
        )
        resolved_image_top_k = next(
            (value for value in (image_top_k, self.image_top_k) if value is not None),
            settings.image_top_k,
        )
        text_matches = (
            self.vector_db.search_by_type(
                vector=query_embedding,
                content_type="text",
                top_k=resolved_text_top_k,
                metadata_filter=metadata_filter,
            )
            if resolved_text_top_k > 0
            else []
        )
        image_matches = (
            self.vector_db.search_by_type(
                vector=query_embedding,
                content_type="image",
                top_k=resolved_image_top_k,
                metadata_filter=metadata_filter,
            )
            if resolved_image_top_k > 0
            else []
        )
        evidence = []
        seen = set()
        for match in list(text_matches) + list(image_matches):
            chunk = EvidenceChunk.from_match(match)
            dedupe_key = (chunk.content_type, chunk.page, chunk.content)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            evidence.append(chunk)

        return sorted(
            evidence,
            key=lambda chunk: (
                chunk.score is None,
                -(chunk.score or 0.0),
                0 if chunk.content_type == "text" else 1,
                chunk.page or 0,
            ),
        )


class ContextAnalysisAgent:
    def build_context(self, evidence: List[EvidenceChunk]) -> ContextPackage:
        text_blocks = []
        context_images = []
        citations = []

        for chunk in evidence:
            citations.append(chunk.citation())
            if chunk.content_type == "text" and chunk.content:
                snippet = " ".join(chunk.content.split())
                max_length = settings.context_snippet_max_length
                if max_length > 0 and len(snippet) > max_length:
                    if max_length <= 3:
                        snippet = "." * max_length
                    else:
                        snippet = snippet[: max_length - 3].rstrip() + "..."
                score = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
                section = f" | {chunk.section_path}" if chunk.section_path else ""
                text_blocks.append(
                    f"- Page {chunk.page}{section} [text, score {score}]: {snippet}"
                )
            elif chunk.content_type == "image" and chunk.content:
                if os.path.exists(chunk.content):
                    filename = os.path.basename(chunk.content)
                    try:
                        image_reference = Path(chunk.content).resolve().relative_to(
                            settings.extracted_images_dir.resolve()
                        ).as_posix()
                    except ValueError:
                        # Preserve the legacy filename contract for old paths.
                        image_reference = filename
                    context_images.append(image_reference)
                    score = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
                    text_blocks.append(f"- Page {chunk.page} [image, score {score}]: {filename}")

        return ContextPackage(
            text_context="\n".join(text_blocks),
            context_images=context_images,
            citations=citations,
        )


class GeminiAnswerProvider:
    def __init__(self, gemini: Any):
        self.gemini = gemini

    def generate_answer(self, question: str, evidence: str) -> str:
        prompt = f"""
You are a professional Saeco assistant. Use the following PDF manual context to answer the user's question accurately.
- Be professional, helpful, and concise.
- Use Markdown formatting for readability (bold text, bullet points).
- Prefer the highest-score evidence first.
- If the answer is not clearly supported by the manual, say you do not know based strictly on the manual.
- Do not invent steps, parts, warnings, or page references.
- When the manual uses a different term than the user's wording, mention the manual term.
- Cite page numbers inline with each factual claim using parentheses like "(p. 12)".
- Break long answers into distinct sections or steps.

Manual Evidence:
{evidence}

User Question: {question}
"""
        response = self.gemini.generate_content(prompt)
        return response.text or ""


class SynthesisAgent:
    def __init__(self, gemini: Any = None, *, answer_provider: Any = None):
        # Retain positional/keyword Gemini injection used by existing callers.
        self.gemini = gemini
        self.answer_provider = answer_provider if answer_provider is not None else GeminiAnswerProvider(gemini)

    def synthesize(self, query: str, context: ContextPackage) -> str:
        return self.answer_provider.generate_answer(query, context.text_context)


class QualityControlAgent:
    def review(
        self,
        query: str,
        answer: str,
        text_context: str,
        citations: List[Dict[str, Any]],
    ) -> List[str]:
        warnings = []
        if not answer.strip():
            warnings.append("Answer generation returned no content.")
        if not text_context.strip():
            warnings.append("No text evidence was retrieved for grounding.")
        if not citations:
            warnings.append("No citations were available for the answer.")
        return warnings


class MultiAgentRAGOrchestrator:
    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        gemini: Any = None,
        context_agent: Optional[ContextAnalysisAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None,
        quality_agent: Optional[QualityControlAgent] = None,
    ):
        self.retrieval_agent = retrieval_agent
        self.context_agent = context_agent or ContextAnalysisAgent()
        self.synthesis_agent = synthesis_agent or SynthesisAgent(gemini)
        self.quality_agent = quality_agent or QualityControlAgent()

    def ask(self, query: str, metadata_filter: Optional[dict] = None) -> Dict[str, Any]:
        agent_trace = []

        evidence = self.retrieval_agent.retrieve(query, metadata_filter=metadata_filter)
        agent_trace.append("retrieval")

        context = self.context_agent.build_context(evidence)
        agent_trace.append("context_analysis")

        answer = self.synthesis_agent.synthesize(query, context)
        agent_trace.append("synthesis")

        quality_warnings = self.quality_agent.review(
            query=query,
            answer=answer,
            text_context=context.text_context,
            citations=context.citations,
        )
        agent_trace.append("quality_control")

        return {
            "answer": answer,
            "context_images": context.context_images,
            "citations": context.citations,
            "quality_warnings": quality_warnings,
            "agent_trace": agent_trace,
        }
