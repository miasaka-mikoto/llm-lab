"""Small dependency-free statistical summaries and chart-ready data."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import NormalDist, mean, median, stdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    values = sorted(float(v) for v in values)
    index = (len(values) - 1) * p / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def confidence_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    avg = mean(values)
    if len(values) < 2:
        return (avg, avg)
    # Standard-library NormalDist keeps this dependency-free while honoring
    # arbitrary confidence levels (rather than silently falling back to 95%).
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = z * stdev(values) / math.sqrt(len(values))
    return (avg - margin, avg + margin)


def summarize(values: Iterable[float], *, success: Iterable[bool] | None = None) -> Dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "success_rate": 0.0, "error_rate": 0.0}
    successes = list(success) if success is not None else []
    low, high = confidence_interval(values)
    result = {
        "count": float(len(values)),
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "ci95_low": low,
        "ci95_high": high,
    }
    if successes:
        result["success_rate"] = sum(bool(s) for s in successes) / len(successes)
        result["error_rate"] = 1.0 - result["success_rate"]
    else:
        result["success_rate"] = 0.0
        result["error_rate"] = 0.0
    return {key: round(value, 8) for key, value in result.items()}


def summarize_runs(runs: Iterable[Any], metric: str = "score") -> Dict[str, float]:
    runs = list(runs)
    values = []
    success = []
    for run in runs:
        metrics = getattr(run, "metrics", {})
        values.append(float(metrics.get(metric, metrics.get("exact_match", 0.0))))
        success.append(getattr(run, "status", "success") == "success" and not getattr(run, "error", ""))
    return summarize(values, success=success)


def group_summary(runs: Iterable[Any], group_by: str = "model", metric: str = "score") -> Dict[str, Dict[str, float]]:
    groups: Dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        key = str(getattr(run, group_by, ""))
        groups[key].append(run)
    return {key: summarize_runs(group, metric) for key, group in sorted(groups.items())}


def chart_data(runs: Iterable[Any], *, kind: str = "bar", x: str = "model", y: str = "score") -> Dict[str, Any]:
    """Return a simple Vega/Plotly-independent chart specification."""
    rows = []
    for run in runs:
        metric_value = getattr(run, y, None)
        if metric_value is None:
            metric_value = getattr(run, "metrics", {}).get(y, 0)
        rows.append({"x": getattr(run, x, ""), "y": metric_value})
    return {"kind": kind, "x": x, "y": y, "data": rows}
