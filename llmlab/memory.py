"""Memory strategies for the Memory Lab, all fully local and inspectable."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, Iterable, List, Optional

from .providers import LLMProvider, MockProvider


@dataclass
class MemoryItem:
    memory_id: str
    text: str
    kind: str = "episodic"
    importance: float = 0.5
    recency: float = 1.0
    relevance: float = 0.5
    created_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_score(self) -> float:
        return round(0.45 * self.importance + 0.25 * self.recency + 0.30 * self.relevance, 8)


class MemoryStrategy:
    name = "base"

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        if self.capacity > 0 and len(self.items) > self.capacity:
            self.items.sort(key=lambda value: value.retrieval_score, reverse=True)
            del self.items[self.capacity :]

    def retrieve(self, query: str = "", *, limit: int = 5, step: int = 0) -> List[MemoryItem]:
        terms = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for item in self.items:
            words = set(re.findall(r"\w+", item.text.casefold()))
            item.relevance = len(terms & words) / max(1, len(terms)) if terms else 0.5
            age = max(0, step - item.created_step)
            item.recency = 1.0 / (1.0 + age / 10.0)
            scored.append(item)
        return sorted(scored, key=lambda value: value.retrieval_score, reverse=True)[: max(0, limit)]

    def consolidate(self, step: int) -> None:
        for item in self.items:
            item.recency = 1.0 / (1.0 + max(0, step - item.created_step) / 10.0)


class NoMemory(MemoryStrategy):
    name = "no_memory"

    def __init__(self):
        super().__init__(capacity=0)

    def add(self, item: MemoryItem) -> None:
        return

    def retrieve(self, query: str = "", *, limit: int = 5, step: int = 0) -> List[MemoryItem]:
        return []


class SlidingWindow(MemoryStrategy):
    name = "sliding_window"

    def add(self, item: MemoryItem) -> None:
        """Keep the most recent ``capacity`` events in insertion order.

        The base strategy evicts by retrieval score, which is useful for a
        ranked long-term store but violates the defining semantics of a
        sliding window (and can retain old events when scores tie).
        """
        self.items.append(item)
        if self.capacity > 0 and len(self.items) > self.capacity:
            del self.items[: len(self.items) - self.capacity]

    def retrieve(self, query: str = "", *, limit: int = 5, step: int = 0) -> List[MemoryItem]:
        return list(reversed(self.items[-limit:]))


class SummaryMemory(MemoryStrategy):
    name = "summary_memory"

    def consolidate(self, step: int) -> None:
        super().consolidate(step)
        if len(self.items) > 10:
            important = sorted(self.items, key=lambda item: item.importance, reverse=True)[:5]
            rest = self.items[-4:]
            summary = MemoryItem(f"summary-{step}", "; ".join(item.text for item in sorted(rest, key=lambda x: x.created_step)), "semantic", 0.7, 1.0, 0.5, step, {"source_count": len(rest)})
            self.items = important + [summary]


class VectorMemory(MemoryStrategy):
    name = "vector_memory"

    def __init__(self, capacity: int = 100, provider: Optional[LLMProvider] = None):
        super().__init__(capacity)
        self.provider = provider or MockProvider()
        self._vectors: Dict[str, List[float]] = {}

    def add(self, item: MemoryItem) -> None:
        super().add(item)
        self._vectors[item.memory_id] = self.provider.embed([item.text])[0]

    def retrieve(self, query: str = "", *, limit: int = 5, step: int = 0) -> List[MemoryItem]:
        if not self.items:
            return []
        query_vector = self.provider.embed([query])[0]
        def similarity(item: MemoryItem) -> float:
            vector = self._vectors.get(item.memory_id, [])
            dot = sum(a * b for a, b in zip(query_vector, vector))
            na = math.sqrt(sum(a * a for a in query_vector)) or 1.0
            nb = math.sqrt(sum(a * a for a in vector)) or 1.0
            item.relevance = (dot / (na * nb) + 1.0) / 2.0
            return item.retrieval_score
        for item in self.items:
            age = max(0, step - item.created_step)
            item.recency = 1 / (1 + age / 10)
        return sorted(self.items, key=similarity, reverse=True)[: max(0, limit)]


class EpisodicMemory(MemoryStrategy):
    name = "episodic_memory"


class SemanticMemory(MemoryStrategy):
    name = "semantic_memory"

    def add(self, item: MemoryItem) -> None:
        item.kind = "semantic"
        super().add(item)


def make_memory(name: str, *, capacity: int = 100, provider: Optional[LLMProvider] = None) -> MemoryStrategy:
    table = {"none": NoMemory, "no_memory": NoMemory, "sliding": SlidingWindow, "sliding_window": SlidingWindow, "summary": SummaryMemory, "summary_memory": SummaryMemory, "vector": VectorMemory, "vector_memory": VectorMemory, "episodic": EpisodicMemory, "episodic_memory": EpisodicMemory, "semantic": SemanticMemory, "semantic_memory": SemanticMemory}
    cls = table.get(name.casefold())
    if cls is None:
        raise ValueError(f"Unknown memory strategy: {name}")
    if cls is VectorMemory:
        return cls(capacity, provider)
    if cls is NoMemory:
        return cls()
    return cls(capacity)
