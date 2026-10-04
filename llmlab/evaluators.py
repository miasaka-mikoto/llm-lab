"""Deterministic, local evaluation metrics for generated outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from statistics import mean
from typing import Any, Callable, Dict, Iterable, Mapping, Optional
from pathlib import Path
import importlib.util


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).strip()).casefold()


def exact_match(output: Any, expected: Any) -> float:
    return 1.0 if _norm(output) == _norm(expected) else 0.0


def contains(output: Any, expected: Any) -> float:
    return 1.0 if _norm(expected) in _norm(output) else 0.0


def regex_match(output: Any, expected: Any) -> float:
    try:
        return 1.0 if re.search(_text(expected), _text(output), flags=re.I | re.M) else 0.0
    except re.error:
        return 0.0


def json_validity(output: Any, expected: Any = None) -> float:
    if not isinstance(output, (str, bytes, bytearray)):
        return 1.0 if isinstance(output, (dict, list, int, float, bool)) or output is None else 0.0
    try:
        json.loads(output)
        return 1.0
    except (ValueError, TypeError):
        return 0.0


def _type_ok(value: Any, type_name: str) -> bool:
    return {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str), "number": isinstance(value, (int, float)) and not isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool), "boolean": isinstance(value, bool), "null": value is None}.get(type_name, True)


def schema_compliance(output: Any, schema: Mapping[str, Any]) -> float:
    try:
        value = json.loads(output) if isinstance(output, (str, bytes, bytearray)) else output
    except (ValueError, TypeError):
        return 0.0
    if not _type_ok(value, schema.get("type", "object")):
        return 0.0
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if any(key not in value for key in required):
            return 0.0
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            return 0.0
        for key, spec in properties.items():
            if key in value and isinstance(spec, Mapping) and not _type_ok(value[key], spec.get("type", "")):
                return 0.0
    return 1.0


def length_metric(output: Any, expected: Any = None) -> float:
    return float(len(_text(output)))


def latency_metric(output: Any, expected: Any = None, *, latency_ms: float = 0.0) -> float:
    return float(latency_ms)


def token_count_metric(output: Any, expected: Any = None, *, token_count: int = 0) -> float:
    return float(token_count)


def cost_estimate(
    output: Any,
    expected: Any = None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> float:
    return (input_tokens * input_price_per_million + output_tokens * output_price_per_million) / 1_000_000.0


@dataclass
class EvaluationResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    errors: list[str] = field(default_factory=list)


class LLMJudgeEvaluator:
    """Opt-in judge evaluator backed by an explicitly supplied provider.

    Passing no provider uses ``MockProvider``; a real provider can only be used
    when a caller explicitly constructs and supplies one.  The judge receives
    a compact rubric and returns a structured 0..1 score.  No hidden reasoning
    or judge trace is persisted by this helper.
    """

    def __init__(self, provider: Any = None, rubric: str = "Does the output answer the expected target correctly?") -> None:
        if provider is None:
            from .providers import MockProvider
            provider = MockProvider("mock-balanced")
        self.provider = provider
        self.rubric = rubric

    def __call__(self, *, output: Any, expected: Any, context: Optional[Mapping[str, Any]] = None) -> float:
        prompt = f"Rubric: {self.rubric}\nExpected: {_text(expected)}\nOutput: {_text(output)}\nReturn a score from 0 to 1."
        response = self.provider.generate("You are an evaluation utility.", prompt, temperature=0.0, max_tokens=64, structured_output={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"], "additionalProperties": False})
        try:
            parsed = json.loads(response.text)
            return max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        except (ValueError, TypeError, AttributeError):
            return 1.0 if exact_match(output, expected) else 0.0


class EvaluatorSuite:
    """Composable evaluator; no network calls and no hidden judge model."""

    def __init__(self, metric_names: Optional[Iterable[str]] = None, *, schema: Optional[Mapping[str, Any]] = None, custom: Optional[Callable[..., float]] = None, judge: Optional[Callable[..., float]] = None) -> None:
        self.metric_names = list(metric_names or ["exact_match"])
        self.schema = schema
        self.custom = custom
        self.judge = judge

    def evaluate(self, output: Any, expected: Any = None, **context: Any) -> EvaluationResult:
        results: Dict[str, float] = {}
        errors: list[str] = []
        for name in self.metric_names:
            try:
                if name in {"exact", "exact_match"}:
                    value = exact_match(output, expected)
                elif name in {"contains", "contain"}:
                    value = contains(output, expected)
                elif name in {"regex", "regex_match"}:
                    value = regex_match(output, expected)
                elif name in {"json", "json_validity", "parse"}:
                    value = json_validity(output, expected)
                elif name in {"schema", "schema_compliance"}:
                    value = schema_compliance(output, self.schema or {})
                elif name == "length":
                    value = length_metric(output, expected)
                elif name == "latency":
                    value = latency_metric(output, expected, latency_ms=float(context.get("latency_ms", 0)))
                elif name in {"tokens", "token_count"}:
                    value = token_count_metric(output, expected, token_count=int(context.get("token_count", 0)))
                elif name in {"cost", "cost_estimate"}:
                    value = cost_estimate(output, expected, input_tokens=int(context.get("input_tokens", 0)), output_tokens=int(context.get("output_tokens", 0)), input_price_per_million=float(context.get("input_price_per_million", 0)), output_price_per_million=float(context.get("output_price_per_million", 0)))
                elif name == "custom" and self.custom:
                    value = float(self.custom(output=output, expected=expected, context=context))
                elif name in {"llm_judge", "judge"} and self.judge:
                    value = float(self.judge(output=output, expected=expected, context=context))
                elif name in {"llm_judge", "judge"}:
                    errors.append("LLM judge metric requires an explicitly supplied judge/provider")
                    continue
                else:
                    errors.append(f"Unknown metric: {name}")
                    continue
                results[name] = float(value)
            except Exception as exc:  # Evaluator errors are data, not crashes.
                errors.append(f"{name}: {exc}")
                results[name] = 0.0
        score_metrics = [v for key, v in results.items() if key not in {"length", "latency", "tokens", "token_count", "cost", "cost_estimate"}]
        return EvaluationResult(results, bool(score_metrics and all(v >= 1.0 for v in score_metrics)) and not errors, errors)


def evaluate_run(run: Any, suite: EvaluatorSuite) -> EvaluationResult:
    return suite.evaluate(run.output, run.expected, latency_ms=run.latency_ms, token_count=run.total_tokens, input_tokens=run.input_tokens, output_tokens=run.output_tokens)


def load_custom_evaluator(path: str | Path, function_name: str = "evaluate") -> Callable[..., float]:
    """Load a user-owned local evaluator module without executing anything else.

    The module must expose ``evaluate(output, expected, context=...)`` (or a
    compatible callable).  This is intentionally an explicit opt-in helper;
    the engine never discovers or runs arbitrary files automatically.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("llmlab_custom_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = getattr(module, function_name, None)
    if not callable(evaluator):
        raise AttributeError(f"Evaluator module must define callable {function_name}()")
    return evaluator
