"""Local RAG laboratory primitives (chunking, fake embeddings and retrieval)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .evaluators import EvaluatorSuite, EvaluationResult
from .providers import LLMProvider, MockProvider


@dataclass
class Document:
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Retrieval:
    chunk: Chunk
    score: float
    rank: int


def chunk_document(document: Document, *, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(document.text), step)):
        end = min(start + chunk_size, len(document.text))
        chunks.append(Chunk(f"{document.document_id}:{index}", document.document_id, document.text[start:end], start, end, dict(document.metadata)))
        if end >= len(document.text):
            break
    return chunks


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore:
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or MockProvider("mock-balanced")
        self._items: List[tuple[Chunk, List[float]]] = []

    def add(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        self._items.extend(zip(chunks, self.provider.embed([chunk.text for chunk in chunks])))

    def clear(self) -> None:
        self._items.clear()

    def search(self, query: str, *, top_k: int = 5) -> List[Retrieval]:
        if top_k <= 0:
            return []
        query_vector = self.provider.embed([query])[0]
        ranked = sorted(((chunk, _cosine(query_vector, vector)) for chunk, vector in self._items), key=lambda item: item[1], reverse=True)
        return [Retrieval(chunk, round(score, 8), i + 1) for i, (chunk, score) in enumerate(ranked[:top_k])]


class Reranker:
    def rerank(self, query: str, results: Iterable[Retrieval], *, top_k: Optional[int] = None) -> List[Retrieval]:
        terms = set(query.casefold().split())
        rescored = []
        for result in results:
            overlap = sum(term in result.chunk.text.casefold() for term in terms)
            rescored.append((result, result.score + overlap * 0.05))
        rescored.sort(key=lambda item: item[1], reverse=True)
        limit = top_k if top_k is not None else len(rescored)
        return [Retrieval(item[0].chunk, round(item[1], 8), i + 1) for i, item in enumerate(rescored[:limit])]


@dataclass
class RAGResult:
    answer: str
    retrieved: List[Retrieval]
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, *, provider: Optional[LLMProvider] = None, store: Optional[VectorStore] = None, reranker: Optional[Reranker] = None):
        self.provider = provider or MockProvider()
        self.store = store or VectorStore(self.provider)
        self.reranker = reranker

    def index(self, documents: Iterable[Document], *, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
        chunks = [chunk for document in documents for chunk in chunk_document(document, chunk_size=chunk_size, overlap=overlap)]
        self.store.add(chunks)
        return chunks

    def query(self, question: str, *, top_k: int = 3, system: str = "Answer using the supplied context.", evaluator: Optional[EvaluatorSuite] = None, expected: Any = None, expected_context: Optional[str] = None) -> RAGResult:
        import time
        started = time.perf_counter()
        retrieved = self.store.search(question, top_k=top_k)
        if self.reranker:
            retrieved = self.reranker.rerank(question, retrieved, top_k=top_k)
        context = "\n\n".join(f"[{item.rank}] {item.chunk.text}" for item in retrieved)
        generated = self.provider.generate(system, f"Context:\n{context}\n\nQuestion: {question}", max_tokens=256)
        metrics: Dict[str, float] = {"latency": generated.latency_ms, "tokens": float(generated.input_tokens + generated.output_tokens), "retrieved": float(len(retrieved))}
        # Recall is optional because a real corpus may not expose labels.  If
        # callers provide the relevant text, report whether retrieval surfaced
        # it; this keeps the metric honest instead of inventing relevance.
        if expected_context is not None:
            needle = str(expected_context).casefold()
            metrics["recall"] = 1.0 if any(needle in item.chunk.text.casefold() for item in retrieved) else 0.0
        if evaluator:
            evaluation = evaluator.evaluate(generated.text, expected, latency_ms=generated.latency_ms, token_count=generated.input_tokens + generated.output_tokens)
            metrics.update(evaluation.metrics)
            quality = [value for key, value in evaluation.metrics.items() if key not in {"latency", "tokens", "token_count", "length", "cost", "cost_estimate"}]
            if quality:
                metrics["answer_quality"] = sum(quality) / len(quality)
        return RAGResult(generated.text, retrieved, round((time.perf_counter() - started) * 1000, 3), metrics)
