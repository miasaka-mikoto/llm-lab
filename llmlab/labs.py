"""Convenience facades for the specialised LLM Lab workbenches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .agent import Agent, AgentRun
from .evaluators import EvaluatorSuite
from .memory import MemoryItem, MemoryStrategy, make_memory
from .providers import LLMProvider, MockProvider
from .rag import Document, RAGPipeline, RAGResult
from .structured import StructuredDiagnostics, parse_json_output


@dataclass
class PromptPlayground:
    provider: LLMProvider = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.provider = self.provider or MockProvider()

    def run(self, system: str, user: str, **parameters: Any) -> Any:
        return self.provider.generate(system, user, **parameters)

    def compare(self, variants: Iterable[Mapping[str, Any]], *, system: str = "") -> list[Any]:
        return [self.run(system, str(variant.get("prompt", variant.get("user", ""))), **{k: v for k, v in variant.items() if k not in {"prompt", "user"}}) for variant in variants]


class RAGLab:
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.pipeline = RAGPipeline(provider=provider or MockProvider())

    def index(self, documents: Iterable[Document], **kwargs: Any) -> int:
        return len(self.pipeline.index(documents, **kwargs))

    def query(self, question: str, **kwargs: Any) -> RAGResult:
        return self.pipeline.query(question, **kwargs)


class MemoryLab:
    def __init__(self, strategy: str = "sliding_window", *, capacity: int = 100, provider: Optional[LLMProvider] = None):
        self.memory: MemoryStrategy = make_memory(strategy, capacity=capacity, provider=provider)

    def remember(self, text: str, *, memory_id: str, importance: float = 0.5, step: int = 0, kind: str = "episodic", **metadata: Any) -> MemoryItem:
        item = MemoryItem(memory_id, text, kind=kind, importance=importance, created_step=step, metadata=metadata)
        self.memory.add(item)
        return item

    def recall(self, query: str = "", *, limit: int = 5, step: int = 0) -> list[MemoryItem]:
        return self.memory.retrieve(query, limit=limit, step=step)


class AgentLab:
    def __init__(self, agent: Optional[Agent] = None):
        self.agent = agent or Agent()

    def run(self, task: Any) -> AgentRun:
        return self.agent.run(task)


class StructuredOutputLab:
    def inspect(self, output: Any, schema: Optional[Mapping[str, Any]] = None) -> StructuredDiagnostics:
        return parse_json_output(output, schema)

