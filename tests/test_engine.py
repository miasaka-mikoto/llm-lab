import json
import tempfile
import unittest
from pathlib import Path

from llmlab import (
    Agent,
    Document,
    EvaluatorSuite,
    Experiment,
    ExperimentMatrix,
    ExperimentRunner,
    MockProvider,
    RAGPipeline,
    VectorStore,
    chunk_document,
    load_dataset,
    make_memory,
    parse_json_output,
    sample_dataset,
    summarize,
)


class EngineTests(unittest.TestCase):
    def test_mock_is_deterministic(self):
        provider = MockProvider()
        first = provider.generate("", "hello", seed=7)
        second = provider.generate("", "hello", seed=7)
        self.assertEqual(first.text, second.text)
        self.assertEqual(len(provider.embed(["a"])[0]), 32)

    def test_dataset_formats_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps({"input": "a", "expected": "b"}) + "\n", encoding="utf-8")
            dataset = load_dataset(path)
            self.assertEqual(dataset.size, 1)
            self.assertTrue(dataset.sha256)

    def test_matrix_runner(self):
        dataset = sample_dataset(3)
        experiment = Experiment(prompt="Classify sentiment: {input}", metrics=["exact_match"])
        runs = ExperimentRunner().run(experiment, dataset, matrix=ExperimentMatrix({"model": ["mock-balanced", "mock-concise"], "temperature": [0, 0.5]}))
        self.assertEqual(len(runs), 12)
        self.assertTrue(all(run.provider == "mock" for run in runs))

    def test_evaluators_and_structured(self):
        suite = EvaluatorSuite(["exact_match", "json_validity"])
        result = suite.evaluate('{"ok": true}', '{"ok": true}')
        self.assertEqual(result.metrics["exact_match"], 1.0)
        self.assertTrue(parse_json_output('{"x":1}', {"type": "object", "required": ["x"]}).valid)

    def test_rag_memory_agent(self):
        pipeline = RAGPipeline()
        chunks = pipeline.index([Document("d1", "The library contains research papers.")], chunk_size=20, overlap=2)
        self.assertGreaterEqual(len(chunks), 2)
        answer = pipeline.query("library", top_k=1)
        self.assertEqual(len(answer.retrieved), 1)
        memory = make_memory("sliding_window")
        self.assertEqual(memory.name, "sliding_window")
        self.assertEqual(Agent().run("2 + 3").final, 5)

    def test_statistics(self):
        summary = summarize([1, 2, 3], success=[True, True, False])
        self.assertEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

