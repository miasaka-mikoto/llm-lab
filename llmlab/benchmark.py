"""Offline model benchmark suite and provenance-aware dashboard.

The benchmark layer deliberately records measurements; it never ships a
pre-computed leaderboard or claims that a model is better without experiment
data.  The default cases are labelled ``sample-mock`` so a dashboard can make
that distinction obvious.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .evaluators import EvaluatorSuite
from .providers import LLMProvider, MockProvider, ProviderRegistry, UnavailableProvider, default_registry
from .statistics import confidence_interval, percentile


CATEGORIES: tuple[str, ...] = (
    "Reasoning",
    "Coding",
    "Extraction",
    "Classification",
    "RAG",
    "Tool Use",
    "Long Context",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkCase:
    """One benchmark prompt and optional expected answer.

    ``expected`` may be omitted for exploratory cases.  In that situation
    quality is reported as ``None`` rather than a fabricated zero score.
    """

    case_id: str
    category: str
    prompt: str
    system: str = ""
    expected: Any = None
    metric_names: List[str] = field(default_factory=lambda: ["exact_match"])
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown benchmark category: {self.category}; use one of {', '.join(CATEGORIES)}")


@dataclass
class BenchmarkResult:
    case_id: str
    category: str
    model: str
    provider: str
    output: Any = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    score: Optional[float] = None
    latency_ms: float = 0.0
    tokens: int = 0
    error: str = ""
    status: str = "success"
    provenance: str = "mock"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkSuite:
    """Run benchmark cases against explicitly selected providers/models."""

    categories = CATEGORIES

    def __init__(
        self,
        cases: Optional[Iterable[BenchmarkCase]] = None,
        *,
        registry: Optional[ProviderRegistry] = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.cases: List[BenchmarkCase] = list(cases or [])
        self.results: List[BenchmarkResult] = []

    @classmethod
    def sample(cls, *, registry: Optional[ProviderRegistry] = None) -> "BenchmarkSuite":
        """Return seven labelled sample cases for an offline first-run view."""
        cases = [
            BenchmarkCase("reasoning-01", "Reasoning", "Classify this reasoning result as positive: excellent evidence", expected="positive", metadata={"provenance": "sample-mock"}),
            BenchmarkCase("coding-01", "Coding", "Write a short Python function that adds two numbers.", metadata={"provenance": "sample-mock"}),
            BenchmarkCase("extraction-01", "Extraction", "Extract the person name as JSON from: Ada Lovelace", metadata={"provenance": "sample-mock"}),
            BenchmarkCase("classification-01", "Classification", "Classify sentiment: I love this", expected="positive", metadata={"provenance": "sample-mock"}),
            BenchmarkCase("rag-01", "RAG", "Answer using context: Paris is the capital of France. What is the capital?", metadata={"provenance": "sample-mock"}),
            BenchmarkCase("tool-use-01", "Tool Use", "Calculate 2 + 3 using the available tool.", metadata={"provenance": "sample-mock"}),
            BenchmarkCase("long-context-01", "Long Context", "Summarize this long context: " + ("local research note " * 40), metadata={"provenance": "sample-mock"}),
        ]
        return cls(cases, registry=registry)

    def add_case(self, case: BenchmarkCase) -> BenchmarkCase:
        if any(existing.case_id == case.case_id for existing in self.cases):
            raise ValueError(f"Duplicate benchmark case: {case.case_id}")
        self.cases.append(case)
        return case

    @staticmethod
    def _provenance(provider: LLMProvider, case: BenchmarkCase) -> str:
        requested = str(case.metadata.get("provenance", ""))
        if requested:
            return requested
        if isinstance(provider, UnavailableProvider):
            return "unavailable"
        if isinstance(provider, MockProvider):
            return "sample-mock" if case.metadata.get("sample", False) else "mock"
        return "experiment"

    @staticmethod
    def _quality_score(metrics: Mapping[str, float]) -> Optional[float]:
        quality_keys = [key for key in metrics if key not in {"latency", "tokens", "token_count", "length", "cost", "cost_estimate"}]
        if not quality_keys:
            return None
        return float(mean(float(metrics[key]) for key in quality_keys))

    def run(
        self,
        models: Sequence[str] = ("mock-balanced",),
        *,
        samples: int = 1,
        cases: Optional[Iterable[BenchmarkCase]] = None,
        clear: bool = True,
    ) -> List[BenchmarkResult]:
        """Execute cases locally and return measurements.

        ``models`` are registry names, not implicit network adapters.  An
        unavailable provider becomes an explicit error row and is never called
        over the network.
        """
        selected_cases = list(cases or self.cases)
        if clear:
            self.results = []
        results: List[BenchmarkResult] = []
        for model_name in models:
            try:
                provider = self.registry.get(model_name)
            except KeyError as exc:
                for case in selected_cases:
                    results.append(BenchmarkResult(case.case_id, case.category, model_name, "unknown", error=str(exc), status="error", provenance="unavailable", parameters=dict(case.parameters)))
                continue
            for case in selected_cases:
                evaluator = EvaluatorSuite(case.metric_names)
                for sample_index in range(max(1, int(samples))):
                    params = dict(case.parameters)
                    if "seed" in params:
                        params["seed"] = int(params["seed"]) + sample_index
                    try:
                        generated = provider.generate(case.system, case.prompt, **params)
                        metrics: Dict[str, float] = {"latency": float(generated.latency_ms), "tokens": float(generated.input_tokens + generated.output_tokens)}
                        score: Optional[float] = None
                        if case.expected is not None:
                            evaluation = evaluator.evaluate(generated.text, case.expected, latency_ms=generated.latency_ms, token_count=generated.input_tokens + generated.output_tokens, input_tokens=generated.input_tokens, output_tokens=generated.output_tokens)
                            metrics.update(evaluation.metrics)
                            score = self._quality_score(evaluation.metrics)
                            status = "success" if not evaluation.errors else "error"
                            error = "; ".join(evaluation.errors)
                        else:
                            status, error = "success", ""
                        results.append(BenchmarkResult(case.case_id, case.category, generated.model, generated.provider, generated.text, metrics, score, generated.latency_ms, generated.input_tokens + generated.output_tokens, error, status, self._provenance(provider, case), params))
                    except Exception as exc:
                        results.append(BenchmarkResult(case.case_id, case.category, model_name, getattr(provider, "name", model_name), error=str(exc), status="error", provenance=self._provenance(provider, case), parameters=params))
        self.results.extend(results)
        return results

    def dashboard(self, results: Optional[Iterable[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Build a category/model aggregate without ranking models."""
        rows = list(results if results is not None else self.results)
        groups: Dict[str, Dict[str, List[BenchmarkResult]]] = {}
        for row in rows:
            groups.setdefault(row.category, {}).setdefault(row.model, []).append(row)
        categories: Dict[str, Any] = {}
        for category in self.categories:
            models: Dict[str, Any] = {}
            for model, group in sorted(groups.get(category, {}).items()):
                scores = [r.score for r in group if r.score is not None]
                latencies = [r.latency_ms for r in group if r.status == "success"]
                tokens = [r.tokens for r in group if r.status == "success"]
                successes = sum(r.status == "success" for r in group)
                metrics: Dict[str, Any] = {
                    "runs": len(group),
                    "success_rate": successes / len(group) if group else 0.0,
                    "error_rate": 1.0 - (successes / len(group) if group else 0.0),
                    "provenance": sorted({r.provenance for r in group}),
                    "mean_score": mean(scores) if scores else None,
                    "mean_latency_ms": mean(latencies) if latencies else 0.0,
                    "p95_latency_ms": percentile(latencies, 95) if latencies else 0.0,
                    "mean_tokens": mean(tokens) if tokens else 0.0,
                    "confidence_interval": confidence_interval(scores) if len(scores) > 1 else None,
                }
                models[model] = metrics
            categories[category] = {"models": models}
        return {
            "title": "LLM Lab Model Benchmark Dashboard",
            "categories": categories,
            "results": [row.to_dict() for row in rows],
            "ranking": None,
            "note": "No model ranking is pre-populated; compare only measured runs and provenance labels.",
            "generated_at": _now(),
        }

    def export(self, path: str | Path, results: Optional[Iterable[BenchmarkResult]] = None) -> Path:
        """Export dashboard as JSON, Markdown, HTML or SVG-like text."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dashboard = self.dashboard(results)
        suffix = path.suffix.lower()
        if suffix == ".json":
            text = json.dumps(dashboard, ensure_ascii=False, indent=2)
        elif suffix in {".md", ".markdown"}:
            lines = [f"# {dashboard['title']}", "", dashboard["note"], "", f"Generated: {dashboard['generated_at']}", "", "| Category | Model | Runs | Mean score | Mean latency (ms) | Provenance |", "|---|---|---:|---:|---:|---|"]
            for category in self.categories:
                for model, values in dashboard["categories"][category]["models"].items():
                    score = "n/a" if values["mean_score"] is None else f"{values['mean_score']:.3f}"
                    lines.append(f"| {category} | {model} | {values['runs']} | {score} | {values['mean_latency_ms']:.2f} | {', '.join(values['provenance'])} |")
            text = "\n".join(lines) + "\n"
        elif suffix == ".html":
            rows = []
            for category in self.categories:
                for model, values in dashboard["categories"][category]["models"].items():
                    score = "n/a" if values["mean_score"] is None else f"{values['mean_score']:.3f}"
                    rows.append(f"<tr><td>{html.escape(category)}</td><td>{html.escape(model)}</td><td>{values['runs']}</td><td>{score}</td><td>{values['mean_latency_ms']:.2f}</td><td>{html.escape(', '.join(values['provenance']))}</td></tr>")
            text = "<!doctype html><meta charset='utf-8'><title>LLM Lab Benchmark</title><style>body{background:#10141c;color:#e8edf5;font:15px system-ui;max-width:1100px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #39485c;padding:8px;text-align:left}</style><h1>" + html.escape(dashboard["title"]) + "</h1><p>" + html.escape(dashboard["note"]) + "</p><table><tr><th>Category</th><th>Model</th><th>Runs</th><th>Mean score</th><th>Mean latency (ms)</th><th>Provenance</th></tr>" + "".join(rows) + "</table>"
        else:
            # Keep a dependency-free, inspectable SVG dashboard.  Bars show
            # measured mean scores only; cases without expected labels are
            # intentionally omitted rather than treated as zero.
            bars: List[str] = []
            measured = [(category, model, values["mean_score"]) for category in self.categories for model, values in dashboard["categories"][category]["models"].items() if values["mean_score"] is not None]
            width, height = 1000, 560
            if measured:
                bar_width = max(10.0, min(80.0, 820.0 / len(measured)))
                for index, (category, model, score) in enumerate(measured):
                    x = 100 + index * bar_width
                    y = 430 - max(0.0, min(1.0, float(score))) * 300
                    bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width-6:.1f}' height='{430-y:.1f}' fill='#63a4ff'/><text x='{x+bar_width/2:.1f}' y='455' text-anchor='middle' fill='#c9d7e8' font-size='10'>{html.escape(category)}</text><text x='{x+bar_width/2:.1f}' y='{y-6:.1f}' text-anchor='middle' fill='#fff' font-size='11'>{float(score):.2f}</text>")
            text = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><rect width='100%' height='100%' fill='#10141c'/><text x='40' y='44' fill='#f0f4fb' font-size='24'>LLM Lab Benchmark Dashboard</text><text x='40' y='72' fill='#94a8be' font-size='13'>Measured runs only · no pre-populated ranking · see provenance</text><line x1='90' y1='430' x2='930' y2='430' stroke='#53677d'/>{''.join(bars)}</svg>"
        path.write_text(text, encoding="utf-8")
        return path


# ``BenchmarkRecord`` was used by an early notebook prototype; retain the
# alias so existing saved notebooks/imports remain compatible while exposing
# the clearer ``BenchmarkResult`` name for new code.
BenchmarkRecord = BenchmarkResult

__all__ = ["CATEGORIES", "BenchmarkCase", "BenchmarkResult", "BenchmarkRecord", "BenchmarkSuite"]
