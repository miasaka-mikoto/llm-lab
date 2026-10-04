"""CLI entrypoint for LLM Lab (offline by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .datasets import load_dataset, sample_dataset
from .experiments import ExperimentMatrix, ExperimentRunner
from .models import Experiment, PromptVersion
from .prompts import PromptTemplate
from .providers import default_registry
from .report import dashboard_svg, export_report, build_manifest
from .storage import SQLiteStore
from .evaluators import EvaluatorSuite


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "prompt_robustness.jsonl"


def _demo(args: argparse.Namespace) -> int:
    dataset_path = args.dataset or SAMPLE
    dataset = load_dataset(dataset_path, name="prompt_robustness") if Path(dataset_path).exists() else sample_dataset(20, name="prompt_robustness")
    experiment = Experiment(
        name="Prompt Robustness Test",
        research_question="Which prompt variant stays reliable across small paraphrased tasks?",
        hypothesis="Structured concise prompts improve exact-match stability.",
        dataset=dataset.name,
        model="mock-balanced",
        prompt="Answer the task precisely: {input}",
        prompt_version="v1",
        parameters={"max_tokens": 64, "seed": 42},
        evaluator="exact_match",
        metrics=["contains", "latency", "token_count"],
        notes="Offline demo: all model outputs are deterministic MockProvider responses.",
    )
    # Four prompt variants and three local model personalities.  Twenty rows
    # in the sample dataset therefore produce the requested 3×4×20 matrix.
    # Keep all four matrix variants as real, persisted prompt versions.  A
    # missing version should be a configuration error in normal experiments,
    # so the demo explicitly defines every version it schedules.
    prompt = PromptTemplate(name="prompt_robustness", active_version="v1")
    prompt.add_version("v1", system="You are an offline test model.", user="Answer precisely: {input}", template="Answer precisely: {input}", notes="baseline")
    prompt.add_version("v2", system="You are an offline test model.", user="Return only the short answer for: {input}", template="Return only the short answer for: {input}", notes="concise")
    prompt.add_version("v3", system="You are an offline test model.", user="Check the task, then answer precisely: {input}", template="Check the task, then answer precisely: {input}", notes="verification")
    prompt.add_version("v4", system="You are an offline test model.", user="Answer the following with a stable label: {input}", template="Answer the following with a stable label: {input}", notes="label framing")
    matrix = ExperimentMatrix({"model": ["mock-balanced", "mock-concise", "mock-noisy"], "prompt_version": ["v1", "v2", "v3", "v4"], "temperature": [0.0]})
    runner = ExperimentRunner(default_registry())
    runs = runner.run(experiment, dataset, prompt=prompt, matrix=matrix, evaluator=EvaluatorSuite(["contains", "latency", "token_count"]), provider_name="mock", retries=1)
    out_dir = Path(args.output or "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    with SQLiteStore(args.db or str(out_dir / "llmlab_demo.sqlite3")) as store:
        store.save_dataset(dataset)
        store.save_experiment(experiment)
        store.save_manifest(build_manifest(experiment, dataset_hash=dataset.sha256, environment={"python": sys.version, "platform": platform.platform()}))
    report_paths = export_report(experiment, out_dir, formats=("md", "html"))
    dash = dashboard_svg(experiment, out_dir / "dashboard.svg", metric="contains")
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_paths[0].read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"experiment_id": experiment.experiment_id, "runs": len(runs), "status": runner.state.status, "report": [str(p) for p in report_paths], "dashboard": str(dash), "database": args.db or str(out_dir / "llmlab_demo.sqlite3"), "offline": True}, ensure_ascii=False, indent=2))
    return 0


def _headless(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    experiment = Experiment(name="Headless Local Experiment", dataset=dataset.name, model="mock-balanced", prompt="Answer: {input}", metrics=["contains", "latency", "token_count"])
    runner = ExperimentRunner(default_registry())
    runner.run(experiment, dataset, evaluator=EvaluatorSuite(experiment.metrics), provider_name="mock")
    print(json.dumps({"runs": len(experiment.runs), "state": runner.state.__dict__, "summary": experiment.results}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-lab", description="Offline LLM Lab research workbench")
    sub = parser.add_subparsers(dest="command")
    demo = sub.add_parser("demo", help="run the 3-model × 4-prompt × 20-sample offline demo")
    demo.add_argument("--dataset", type=Path, default=None)
    demo.add_argument("--db", default=None)
    demo.add_argument("--output", default="reports")
    demo.add_argument("--report", default=None)
    demo.set_defaults(func=_demo)
    headless = sub.add_parser("headless", help="run a local dataset without opening the GUI")
    headless.add_argument("--dataset", type=Path, required=True)
    headless.set_defaults(func=_headless)
    sub.add_parser("gui", help="open the Tk research workbench").set_defaults(func=lambda _args: _gui())
    sub.add_parser("self-check", help="run a fast offline smoke check").set_defaults(func=_self_check)
    return parser


def _gui() -> int:
    from .gui import GuiAdapter, run
    # Persist normal desktop sessions while retaining a fallback for a
    # read-only or unusually minimal environment.
    try:
        with SQLiteStore(Path.cwd() / "llmlab.sqlite3") as store:
            run(adapter=GuiAdapter(store=store, project_dir=Path.cwd()), project_dir=Path.cwd())
    except Exception:
        run(project_dir=Path.cwd())
    return 0


def _self_check(_args: argparse.Namespace) -> int:
    from .providers import MockProvider
    from .structured import parse_json_output
    from .rag import Document, RAGPipeline
    provider = MockProvider()
    result = provider.generate("", "hello", seed=1)
    rag = RAGPipeline(provider=provider)
    rag.index([Document("d1", "Local experiments are reproducible.")], chunk_size=20, overlap=2)
    checks = {"provider": bool(result.text), "structured": parse_json_output('{"ok": true}', {"type": "object", "required": ["ok"]}).valid, "rag": bool(rag.query("reproducible").retrieved)}
    print(json.dumps(checks, indent=2))
    return 0 if all(checks.values()) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "command", None):
        return _gui()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
