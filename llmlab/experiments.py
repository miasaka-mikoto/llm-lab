"""Experiment matrix generation and offline batch execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

from .datasets import Dataset
from .evaluators import EvaluatorSuite
from .models import Experiment, RunResult, TraceEvent
from .models import PromptVersion
from .providers import LLMProvider, ProviderRegistry, default_registry


@dataclass
class MatrixPoint:
    values: Dict[str, Any]
    index: int = 0


class ExperimentMatrix:
    """Cartesian product of model/prompt/parameter dimensions."""

    def __init__(self, dimensions: Mapping[str, Iterable[Any]]):
        self.dimensions = {key: list(values) for key, values in dimensions.items()}

    def points(self) -> List[MatrixPoint]:
        if not self.dimensions:
            return [MatrixPoint({}, 0)]
        keys = list(self.dimensions)
        return [MatrixPoint(dict(zip(keys, values)), i) for i, values in enumerate(itertools.product(*(self.dimensions[k] for k in keys)))]

    def __len__(self) -> int:
        size = 1
        for values in self.dimensions.values():
            size *= len(values)
        return size


@dataclass
class RunnerState:
    status: str = "idle"  # idle/running/paused/completed/cancelled
    completed: int = 0
    failed: int = 0
    total: int = 0
    errors: List[str] = field(default_factory=list)


class ExperimentRunner:
    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self.registry = registry or default_registry()
        self.state = RunnerState()
        self._pause_requested = False
        self._cancel_requested = False
        self._current_jobs: List[tuple[Dict[str, Any], Any]] = []

    def pause(self) -> None:
        self._pause_requested = True
        if self.state.status == "running":
            self.state.status = "paused"

    def resume(self) -> None:
        self._pause_requested = False
        if self.state.status == "paused":
            self.state.status = "running"

    def cancel(self) -> None:
        self._cancel_requested = True
        self.state.status = "cancelled"

    def _jobs(self, dataset: Dataset, matrix: ExperimentMatrix) -> Iterator[tuple[Dict[str, Any], Any]]:
        points = matrix.points()
        for point in points:
            for record in dataset.records:
                yield point.values, record

    def run(
        self,
        experiment: Experiment,
        dataset: Dataset,
        *,
        prompt: Optional[PromptVersion] = None,
        matrix: Optional[ExperimentMatrix] = None,
        evaluator: Optional[EvaluatorSuite] = None,
        retries: int = 1,
        provider_name: str = "mock",
        on_run: Optional[Callable[[RunResult], None]] = None,
    ) -> List[RunResult]:
        matrix = matrix or ExperimentMatrix({})
        evaluator = evaluator or EvaluatorSuite(experiment.metrics or [experiment.evaluator or "exact_match"])
        jobs = list(self._jobs(dataset, matrix))
        # Resume support: leave existing successful sample IDs untouched.
        # Parameter values may include structured-output dictionaries/lists;
        # canonical JSON keeps resume keys hashable and deterministic.
        import json
        def _key_params(values: Mapping[str, Any]) -> str:
            return json.dumps(dict(values), ensure_ascii=False, sort_keys=True, default=str)
        existing = {(run.sample_id, _key_params(run.parameters)) for run in experiment.runs if run.status == "success"}
        self.state = RunnerState("running", 0, 0, len(jobs))
        self._pause_requested = False
        self._cancel_requested = False
        results: List[RunResult] = []
        for point, record in jobs:
            if self._cancel_requested:
                break
            while self._pause_requested:
                self.state.status = "paused"
                time.sleep(0.01)
            self.state.status = "running"
            params = {**experiment.parameters, **point}
            key = (record.record_id, _key_params(params))
            if key in existing:
                continue
            model_name = str(point.get("model", experiment.model or provider_name))
            provider = self.registry.get(model_name if model_name in self.registry.names() else provider_name)
            version = str(point.get("prompt_version", experiment.prompt_version))
            if prompt:
                # Accept either PromptTemplate.render(values, version) or a
                # single PromptVersion.render(values), keeping the public API
                # convenient for notebooks and the CLI.
                try:
                    system, user_template = prompt.render({"input": record.input, "context": record.context or ""}, version)
                except TypeError:
                    system, user_template = prompt.render({"input": record.input, "context": record.context or ""})
            else:
                system, user_template = "", (experiment.prompt or "{input}")
            if not prompt:
                try:
                    user_template = user_template.format(input=record.input, context=record.context or "")
                except (KeyError, ValueError):
                    user_template = user_template
            result: Optional[RunResult] = None
            last_error = ""
            for attempt in range(max(0, retries) + 1):
                try:
                    trace = [TraceEvent(0, "request", action="generate", observation=user_template, thought_summary="Provider request recorded for reproducibility.")]
                    generated = provider.generate(system, user_template, temperature=float(point.get("temperature", params.get("temperature", 0.0))), top_p=float(point.get("top_p", params.get("top_p", 1.0))), max_tokens=int(point.get("max_tokens", params.get("max_tokens", 256))), seed=point.get("seed", params.get("seed")), structured_output=point.get("structured_output", params.get("structured_output")))
                    evaluation = evaluator.evaluate(generated.text, record.expected, latency_ms=generated.latency_ms, token_count=generated.input_tokens + generated.output_tokens, input_tokens=generated.input_tokens, output_tokens=generated.output_tokens)
                    trace.append(TraceEvent(1, "response", action="generate", result=generated.text, thought_summary="Provider response and evaluation recorded."))
                    result = RunResult(experiment_id=experiment.experiment_id, sample_id=record.record_id, model=generated.model, provider=generated.provider, prompt_version=version, input=user_template, output=generated.text, expected=record.expected, parameters=params, metrics=evaluation.metrics, latency_ms=generated.latency_ms, input_tokens=generated.input_tokens, output_tokens=generated.output_tokens, status="success" if not evaluation.errors else "error", error="; ".join(evaluation.errors))
                    result.trace = trace
                    break
                except Exception as exc:
                    last_error = str(exc)
            if result is None:
                result = RunResult(experiment_id=experiment.experiment_id, sample_id=record.record_id, model=model_name, provider=provider_name, prompt_version=version, input=user_template, expected=record.expected, parameters=params, status="error", error=last_error, trace=[TraceEvent(0, "error", action="generate", result=last_error, thought_summary="Provider request failed after retries.")])
            experiment.add_run(result)
            results.append(result)
            self.state.completed += 1
            if result.status != "success":
                self.state.failed += 1
                self.state.errors.append(result.error)
            if on_run:
                on_run(result)
        if self.state.status != "cancelled":
            self.state.status = "completed"
        experiment.results = {"summary": {"total": len(experiment.runs), "success": sum(r.status == "success" for r in experiment.runs), "errors": sum(r.status != "success" for r in experiment.runs)}}
        return results


def side_by_side(runs: Iterable[RunResult], *, group_key: str = "sample_id") -> List[Dict[str, Any]]:
    """Build UI-ready rows comparing model outputs for the same sample."""
    groups: Dict[str, List[RunResult]] = {}
    for run in runs:
        groups.setdefault(str(getattr(run, group_key, "")), []).append(run)
    rows: List[Dict[str, Any]] = []
    for sample_id, values in groups.items():
        rows.append({"sample_id": sample_id, "variants": [{"model": run.model, "prompt_version": run.prompt_version, "output": run.output, "score": run.metrics.get("exact_match", run.metrics.get("score", 0.0)), "latency_ms": run.latency_ms, "tokens": run.total_tokens, "error": run.error} for run in sorted(values, key=lambda item: (item.model, item.prompt_version))]})
    return rows
