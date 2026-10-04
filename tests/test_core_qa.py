"""Offline regression tests for the LLM Lab foundation.

These tests intentionally exercise only the deterministic/local path.  They
must never require an API key or make a network request, which makes them safe
to run in CI and on a fresh checkout.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llmlab.datasets import DatasetViewer, from_records, load_dataset, sample_dataset
from llmlab.evaluators import EvaluatorSuite, schema_compliance
from llmlab.prompts import make_prompt
from llmlab.providers import MockProvider, UnavailableProvider, default_registry
from llmlab.rag import Document, RAGPipeline, Reranker, VectorStore, chunk_document
from llmlab.memory import MemoryItem, NoMemory, SummaryMemory, make_memory
from llmlab.statistics import chart_data, confidence_interval, percentile, summarize


class ProviderQATest(unittest.TestCase):
    def test_mock_is_deterministic_and_supports_all_interface_methods(self):
        provider = MockProvider("mock-balanced", latency_ms=0)
        first = provider.generate("system", "classify: this is excellent", seed=7)
        second = provider.generate("system", "classify: this is excellent", seed=7)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.output_tokens, provider.count_tokens(first.text))
        self.assertEqual(list(provider.stream("system", "hello", seed=1)), ["Mock ", "answer: ", "hello"])
        vectors = provider.embed(["a", "b"], dimensions=4)
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 4 for vector in vectors))

    def test_registry_exposes_mock_and_blocks_future_network_adapters(self):
        registry = default_registry()
        self.assertIn("mock", registry.names())
        self.assertIn("ollama", registry.names())
        with self.assertRaises(RuntimeError):
            UnavailableProvider("openai").generate("", "")

    def test_structured_output_respects_schema(self):
        schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
        result = MockProvider(latency_ms=0).generate("", "hello", structured_output=schema, seed=3)
        parsed = json.loads(result.text)
        self.assertIn("label", parsed)
        self.assertEqual(schema_compliance(result.text, schema), 1.0)


class DatasetPromptQATest(unittest.TestCase):
    def test_dataset_formats_and_field_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.json").write_text(json.dumps([{"question": "q", "answer": "a", "fold": "dev"}]), encoding="utf-8")
            dataset = load_dataset(root / "rows.json", field_mapping={"input": "question", "expected": "answer", "split": "fold"})
            self.assertEqual(dataset.size, 1)
            self.assertEqual(dataset.records[0].input, "q")
            self.assertEqual(dataset.records[0].expected, "a")
            self.assertEqual(dataset.records[0].split, "dev")
            self.assertEqual(DatasetViewer(dataset).page(split="dev")[0]["input"], "q")

    def test_sample_dataset_and_prompt_version_rendering(self):
        dataset = sample_dataset(7)
        self.assertEqual(dataset.size, 7)
        prompt = make_prompt("demo", "You are concise", "Answer {topic}", variables={"topic": "LLMs"})
        prompt.add_version("v2", system="You are detailed", user="Explain {topic}", variables={"topic": "RAG"})
        self.assertEqual(prompt.render(version="v2"), ("You are detailed", "Explain RAG"))
        self.assertEqual([item["version"] for item in prompt.compare_versions()], ["v1", "v2"])


class EvaluationStatisticsQATest(unittest.TestCase):
    def test_evaluator_suite_metrics_and_failures_are_data(self):
        suite = EvaluatorSuite(["exact_match", "contains", "json_validity", "schema_compliance"], schema={"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}})
        result = suite.evaluate('{"x": 2}', '{"x": 2}')
        self.assertEqual(result.metrics["json_validity"], 1.0)
        self.assertEqual(result.metrics["schema_compliance"], 1.0)
        self.assertTrue(result.passed)
        unknown = EvaluatorSuite(["not_a_metric"]).evaluate("a", "a")
        self.assertFalse(unknown.passed)
        self.assertTrue(unknown.errors)

    def test_statistics_percentiles_ci_and_chart_data(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        low, high = confidence_interval([1, 2, 3])
        self.assertLess(low, high)
        low90, high90 = confidence_interval([1, 2, 3], 0.90)
        low99, high99 = confidence_interval([1, 2, 3], 0.99)
        self.assertLess(high90 - low90, high99 - low99)
        with self.assertRaises(ValueError):
            confidence_interval([1, 2, 3], 95)
        summary = summarize([1, 2, 3], success=[True, True, False])
        self.assertEqual(summary["count"], 3.0)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3, places=7)
        spec = chart_data([], kind="histogram", x="model", y="score")
        self.assertEqual(spec["kind"], "histogram")
        self.assertEqual(spec["data"], [])


class RagMemoryQATest(unittest.TestCase):
    def test_rag_chunking_retrieval_and_query_are_offline(self):
        document = Document("doc-1", "alpha beta gamma " * 40, {"source": "qa"})
        chunks = chunk_document(document, chunk_size=24, overlap=4)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].document_id, "doc-1")
        store = VectorStore(MockProvider(latency_ms=0))
        store.add(chunks)
        retrieved = Reranker().rerank("alpha", store.search("alpha", top_k=3), top_k=2)
        self.assertLessEqual(len(retrieved), 2)
        result = RAGPipeline(provider=MockProvider(latency_ms=0), store=store).query("alpha", top_k=2)
        self.assertEqual(len(result.retrieved), 2)
        self.assertTrue(result.answer)

    def test_memory_strategies_are_bounded_and_no_memory_is_empty(self):
        self.assertEqual(NoMemory().retrieve("anything"), [])
        memory = make_memory("sliding_window", capacity=3)
        for i in range(5):
            memory.add(MemoryItem(str(i), f"event {i}", created_step=i))
        self.assertEqual([item.memory_id for item in memory.retrieve(limit=2)], ["4", "3"])
        summary = SummaryMemory(capacity=30)
        for i in range(12):
            summary.add(MemoryItem(str(i), f"event {i}", importance=0.1, created_step=i))
        summary.consolidate(12)
        self.assertLessEqual(len(summary.items), 10)


if __name__ == "__main__":
    unittest.main()
