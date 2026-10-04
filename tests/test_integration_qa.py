"""End-to-end offline checks for persistence, matrix execution and exports."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llmlab.agent import Agent
from llmlab.benchmark import BenchmarkSuite, CATEGORIES
from llmlab.datasets import sample_dataset
from llmlab.evaluators import EvaluatorSuite
from llmlab.experiments import ExperimentMatrix, ExperimentRunner
from llmlab.models import Experiment, RunResult, TraceEvent
from llmlab.report import build_manifest, export_report
from llmlab.storage import SQLiteStore
from llmlab.structured import parse_json_output


class IntegrationQATest(unittest.TestCase):
    def test_matrix_runner_resume_and_sqlite_round_trip(self):
        dataset = sample_dataset(2)
        experiment = Experiment(name="QA", prompt="Answer {input}", metrics=["contains"])
        matrix = ExperimentMatrix({"model": ["mock-balanced", "mock-concise"], "temperature": [0.0, 0.5]})
        runner = ExperimentRunner()
        first = runner.run(experiment, dataset, matrix=matrix, evaluator=EvaluatorSuite(["contains"]))
        self.assertEqual(len(first), 8)
        self.assertEqual(runner.state.status, "completed")
        # Re-running the same configuration should use the persisted-success
        # keys and avoid adding duplicates.
        second = runner.run(experiment, dataset, matrix=matrix, evaluator=EvaluatorSuite(["contains"]))
        self.assertEqual(second, [])
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "qa.sqlite3"
            with SQLiteStore(db) as store:
                store.save_dataset(dataset)
                store.save_experiment(experiment)
                manifest = build_manifest(experiment, dataset_hash=dataset.sha256)
                store.save_manifest(manifest)
                self.assertEqual(store.load_manifest(experiment.experiment_id).dataset_hash, dataset.sha256)
                store.save_snapshot("snap-1", experiment.experiment_id, "after-run", {"runs": len(experiment.runs)})
                loaded = store.load_experiment(experiment.experiment_id)
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(len(loaded.runs), 8)
                self.assertEqual(store.load_snapshot("snap-1")["runs"], 8)

            # Incremental writes must also be visible through load_experiment,
            # not only through the indexed list_runs view.
            incremental_db = Path(directory) / "incremental.sqlite3"
            with SQLiteStore(incremental_db) as store:
                empty = Experiment(name="incremental")
                store.save_experiment(empty)
                store.save_run(RunResult(experiment_id=empty.experiment_id, sample_id="s1", output="ok"))
                loaded = store.load_experiment(empty.experiment_id)
                self.assertIsNotNone(loaded)
                self.assertEqual(len(loaded.runs), 1)

    def test_report_exports_markdown_and_html_and_preserves_trace(self):
        experiment = Experiment(name="Report QA", research_question="Does export work?")
        run = ExperimentRunner().run(experiment, sample_dataset(1), evaluator=EvaluatorSuite(["contains"]))[0]
        run.trace = [TraceEvent(0, "observation", observation="public input", thought_summary="rule summary")]
        with tempfile.TemporaryDirectory() as directory:
            paths = export_report(experiment, directory, formats=("md", "html"))
            self.assertEqual({path.suffix for path in paths}, {".md", ".html"})
            markdown = paths[0].read_text(encoding="utf-8")
            self.assertIn("Report QA", markdown)
            self.assertIn("Success rate", markdown)
            self.assertIn("Contains mean", markdown)
            self.assertNotIn("chain-of-thought", markdown.casefold())

    def test_agent_trace_is_public_and_structured_diagnostics_are_actionable(self):
        result = Agent().run("calculate 2 + 3")
        self.assertEqual(result.final, 5)
        self.assertTrue(all(event.thought_summary for event in result.trace))
        self.assertFalse(any("chain" in event.thought_summary.casefold() for event in result.trace))
        diag = parse_json_output('{"x":"wrong","extra":true}', {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}, "additionalProperties": False})
        self.assertTrue(diag.valid)
        self.assertIn("x", diag.wrong_types)
        self.assertIn("extra", diag.hallucinated_fields)

    def test_cli_self_check_and_headless_are_offline(self):
        root = Path(__file__).resolve().parents[1]
        env = {"PYTHONPATH": str(root), "PATH": __import__("os").environ.get("PATH", "")}
        check = subprocess.run([sys.executable, "-m", "llmlab", "self-check"], cwd=root, env=env, text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(check.stdout)["provider"])
        data_path = root / "sample_data" / "prompt_robustness.jsonl"
        headless = subprocess.run([sys.executable, "-m", "llmlab", "headless", "--dataset", str(data_path)], cwd=root, env=env, text=True, capture_output=True, check=True)
        payload = json.loads(headless.stdout)
        self.assertEqual(payload["runs"], 20)
        self.assertEqual(payload["state"]["status"], "completed")

    def test_benchmark_dashboard_is_measured_and_provenance_aware(self):
        suite = BenchmarkSuite.sample()
        results = suite.run(["mock-balanced", "mock-concise"])
        self.assertEqual(len(results), len(CATEGORIES) * 2)
        dashboard = suite.dashboard()
        self.assertEqual(list(dashboard["categories"]), list(CATEGORIES))
        self.assertIsNone(dashboard["ranking"])
        provenance = {
            p
            for category in dashboard["categories"].values()
            for metrics in category["models"].values()
            for p in metrics["provenance"]
        }
        self.assertIn("sample-mock", provenance)


if __name__ == "__main__":
    unittest.main()
