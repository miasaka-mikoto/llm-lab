import tempfile
import unittest
from pathlib import Path

from llmlab import (
    EvaluatorSuite,
    Experiment,
    ExperimentMatrix,
    ExperimentRunner,
    SQLiteStore,
    dashboard_svg,
    export_report,
    load_dataset,
)
from llmlab.report import build_manifest


class IntegrationTests(unittest.TestCase):
    def test_matrix_storage_report_and_trace(self):
        root = Path(__file__).resolve().parents[1]
        dataset = load_dataset(root / "sample_data" / "prompt_robustness.jsonl")
        experiment = Experiment(name="integration", prompt="Answer {input}", metrics=["contains", "latency"])
        runner = ExperimentRunner()
        runner.run(experiment, dataset, matrix=ExperimentMatrix({"model": ["mock-balanced", "mock-concise"]}), evaluator=EvaluatorSuite(["contains", "latency"]))
        self.assertEqual(len(experiment.runs), 40)
        self.assertTrue(all(run.trace for run in experiment.runs))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = tmp_path / "lab.sqlite3"
            with SQLiteStore(db) as store:
                store.save_dataset(dataset)
                store.save_experiment(experiment)
                store.save_manifest(build_manifest(experiment, dataset_hash=dataset.sha256))
                loaded = store.load_experiment(experiment.experiment_id)
                self.assertIsNotNone(loaded)
                self.assertEqual(len(loaded.runs), 40)
                self.assertEqual(len(store.list_runs(experiment.experiment_id)), 40)
            paths = export_report(experiment, tmp_path, formats=("md", "html"))
            self.assertTrue(all(path.exists() for path in paths))
            svg = dashboard_svg(experiment, tmp_path / "dashboard.svg")
            self.assertIn("<svg", svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
