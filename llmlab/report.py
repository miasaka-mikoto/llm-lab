"""Offline research report and dashboard export helpers."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .models import Experiment, ReproducibilityManifest, utc_now
from .statistics import group_summary, summarize_runs


def _rows(experiment: Experiment) -> list[dict[str, Any]]:
    rows = []
    for run in experiment.runs:
        rows.append({
            "run_id": run.run_id,
            "sample_id": run.sample_id,
            "model": run.model,
            "prompt_version": run.prompt_version,
            "output": run.output,
            "expected": run.expected,
            "status": run.status,
            "latency_ms": run.latency_ms,
            "tokens": run.total_tokens,
            "error": run.error,
            **{f"metric:{k}": v for k, v in run.metrics.items()},
        })
    return rows


def _primary_score_metric(experiment: Experiment) -> str:
    """Pick a meaningful quality metric present in the run payloads.

    Reports used to hard-code ``exact_match``.  That made perfectly valid
    contains/regex/schema experiments look like all-zero results.  Keep
    exact-match as the preferred metric when available, then fall back to the
    other built-in quality metrics (and finally ``score`` for GUI rows).
    """
    preferred = ("exact_match", "contains", "regex", "regex_match", "schema_compliance", "json_validity", "score")
    available = {key for run in experiment.runs for key in getattr(run, "metrics", {})}
    return next((name for name in preferred if name in available), "exact_match")


def build_manifest(experiment: Experiment, *, dataset_hash: str = "", environment: Optional[Mapping[str, Any]] = None, dependency_versions: Optional[Mapping[str, str]] = None) -> ReproducibilityManifest:
    return ReproducibilityManifest(
        experiment_id=experiment.experiment_id,
        config=experiment.to_dict(),
        environment=dict(environment or {"python": __import__("sys").version, "platform": __import__("platform").platform()}),
        dependency_versions=dict(dependency_versions or {"stdlib": "python-standard-library"}),
        dataset_hash=dataset_hash,
        prompt_version=experiment.prompt_version,
        provider=experiment.model,
        parameters=dict(experiment.parameters),
    )


def markdown_report(experiment: Experiment, *, title: Optional[str] = None, method: str = "", interpretation: str = "", limitations: str = "", next_experiment: str = "") -> str:
    title = title or experiment.name
    score_metric = _primary_score_metric(experiment)
    summary = summarize_runs(experiment.runs, metric=score_metric)
    groups = group_summary(experiment.runs, group_by="model", metric=score_metric)
    lines = [f"# {title}", "", f"- **Experiment ID:** `{experiment.experiment_id}`", f"- **Created:** {experiment.created_at}", f"- **Prompt version:** `{experiment.prompt_version}`", f"- **Provider/model:** `{experiment.model}`", "", "## Question and hypothesis", "", experiment.research_question or "_(not specified)_", "", f"**Hypothesis:** {experiment.hypothesis or '_(not specified)_'}", "", "## Method", "", method or "Offline deterministic provider; see experiment configuration.", "", "## Results", "", f"Runs: **{len(experiment.runs)}**  ", f"Success rate: **{summary.get('success_rate', 0):.1%}**  ", f"{score_metric.replace('_', ' ').title()} mean: **{summary.get('mean', 0):.3f}**  ", f"Median: **{summary.get('median', 0):.3f}**  ", f"P95: **{summary.get('p95', 0):.3f}**", "", "### By model", "", "| Model | N | Mean | P95 | Error rate |", "|---|---:|---:|---:|---:|"]
    for model, stats in groups.items():
        lines.append(f"| {model} | {int(stats.get('count', 0))} | {stats.get('mean', 0):.3f} | {stats.get('p95', 0):.3f} | {stats.get('error_rate', 0):.1%} |")
    lines += ["", "### Sample outputs", "", "| Sample | Model | Output | Score | Latency (ms) |", "|---|---|---|---:|---:"]
    for row in _rows(experiment)[:50]:
        output = str(row["output"]).replace("|", "\\|").replace("\n", " ")[:160]
        score = row.get(f"metric:{score_metric}", row.get("metric:score", ""))
        lines.append(f"| {row['sample_id']} | {row['model']} | {output} | {score} | {row['latency_ms']:.2f} |")
    lines += ["", "## Interpretation", "", interpretation or "_(not specified)_", "", "## Limitations", "", limitations or "Mock outputs are deterministic simulations and are not evidence of real-model capability.", "", "## Next experiment", "", next_experiment or "Compare a local model under the same frozen dataset and prompt versions.", "", "## Reproducibility", "", "```json", json.dumps({"experiment": experiment.to_dict(), "offline": True}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def html_report(experiment: Experiment, **kwargs: Any) -> str:
    md = markdown_report(experiment, **kwargs)
    # Keep HTML export dependency-free while preserving readable headings/tables.
    body = escape(md)
    for level in range(1, 4):
        body = body.replace(f"#" * level + " ", f"<h{level}>", 1)
    body = body.replace("\n\n", "</p><p>").replace("\n", "<br>\n")
    return "<!doctype html><meta charset='utf-8'><title>LLM Lab report</title><style>body{background:#10141c;color:#e8edf5;font:15px system-ui;max-width:1100px;margin:40px auto;padding:0 24px}h1,h2,h3{color:#79c7ff}code{color:#ffd479}table{border-collapse:collapse}</style><p>" + body + "</p>"


def export_report(experiment: Experiment, directory: str | Path = "reports", *, formats: Iterable[str] = ("md", "html"), **kwargs: Any) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment.name).strip("_") or "experiment"
    paths: list[Path] = []
    for fmt in formats:
        path = directory / f"{safe}_{experiment.experiment_id}.{fmt}"
        path.write_text(markdown_report(experiment, **kwargs) if fmt == "md" else html_report(experiment, **kwargs), encoding="utf-8")
        paths.append(path)
    return paths


def dashboard_svg(experiment: Experiment, path: str | Path = "reports/dashboard.svg", *, metric: str = "exact_match") -> Path:
    """Write a compact self-contained SVG dashboard (no plotting dependency)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_summary(experiment.runs, group_by="model", metric=metric)
    width, height = 900, 520
    chart_x, chart_y, chart_w, chart_h = 70, 130, 780, 300
    bars = []
    n = max(1, len(groups))
    bar_w = min(140, chart_w / n * 0.55)
    for i, (model, stats) in enumerate(groups.items()):
        value = max(0.0, min(1.0, float(stats.get("mean", 0))))
        x = chart_x + (i + 0.5) * chart_w / n - bar_w / 2
        y = chart_y + chart_h * (1 - value)
        bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{chart_y+chart_h-y:.1f}' rx='6' fill='#62c6ff'/><text x='{x+bar_w/2:.1f}' y='{chart_y+chart_h+26}' text-anchor='middle' fill='#c8d5e5' font-size='14'>{escape(model)}</text><text x='{x+bar_w/2:.1f}' y='{y-8:.1f}' text-anchor='middle' fill='#fff' font-size='14'>{value:.2f}</text>")
    summary = summarize_runs(experiment.runs, metric=metric)
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='#10141c'/><text x='40' y='48' fill='#f0f4fb' font-size='26' font-family='system-ui'>LLM Lab · {escape(experiment.name)}</text><text x='40' y='78' fill='#94a8be' font-size='14' font-family='system-ui'>Offline dashboard · metric: {escape(metric)} · {len(experiment.runs)} runs</text>
<line x1='{chart_x}' y1='{chart_y}' x2='{chart_x}' y2='{chart_y+chart_h}' stroke='#53677d'/><line x1='{chart_x}' y1='{chart_y+chart_h}' x2='{chart_x+chart_w}' y2='{chart_y+chart_h}' stroke='#53677d'/><text x='28' y='{chart_y+8}' fill='#94a8be' font-size='12'>1.0</text><text x='28' y='{chart_y+chart_h}' fill='#94a8be' font-size='12'>0.0</text>{''.join(bars)}
<text x='40' y='485' fill='#c8d5e5' font-size='14' font-family='system-ui'>Mean {summary.get('mean', 0):.3f} · Median {summary.get('median', 0):.3f} · P95 {summary.get('p95', 0):.3f} · Success {summary.get('success_rate', 0):.1%}</text></svg>"""
    path.write_text(svg, encoding="utf-8")
    return path
