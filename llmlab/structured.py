"""Structured-output parsing and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class StructuredDiagnostics:
    parsed: Any = None
    valid: bool = False
    parse_error: str = ""
    missing_fields: List[str] = field(default_factory=list)
    wrong_types: Dict[str, str] = field(default_factory=dict)
    hallucinated_fields: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.valid:
            return 0.0
        penalty = len(self.missing_fields) + len(self.wrong_types) + len(self.hallucinated_fields)
        return 1.0 if penalty == 0 else max(0.0, 1.0 - penalty * 0.2)


def _type_ok(value: Any, name: str) -> bool:
    return {"string": isinstance(value, str), "number": isinstance(value, (int, float)) and not isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool), "boolean": isinstance(value, bool), "array": isinstance(value, list), "object": isinstance(value, dict), "null": value is None}.get(name, True)


def parse_json_output(output: Any, schema: Optional[Mapping[str, Any]] = None) -> StructuredDiagnostics:
    diagnostics = StructuredDiagnostics()
    try:
        if isinstance(output, (str, bytes, bytearray)):
            text = output.decode() if isinstance(output, bytes) else str(output)
            # Permit a JSON fenced block while still reporting truly malformed output.
            match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
            if match:
                text = match.group(1).strip()
            diagnostics.parsed = json.loads(text)
        else:
            diagnostics.parsed = output
    except (ValueError, TypeError) as exc:
        diagnostics.parse_error = str(exc)
        return diagnostics
    diagnostics.valid = True
    if schema:
        if schema.get("type") and not _type_ok(diagnostics.parsed, schema["type"]):
            diagnostics.wrong_types["$"] = schema["type"]
            return diagnostics
        if isinstance(diagnostics.parsed, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", list(properties))
            diagnostics.missing_fields = [key for key in required if key not in diagnostics.parsed]
            if schema.get("additionalProperties") is False:
                diagnostics.hallucinated_fields = [key for key in diagnostics.parsed if key not in properties]
            for key, spec in properties.items():
                if key in diagnostics.parsed and isinstance(spec, Mapping) and spec.get("type") and not _type_ok(diagnostics.parsed[key], spec["type"]):
                    diagnostics.wrong_types[key] = spec["type"]
    return diagnostics


def extract_json(text: str) -> Any:
    diagnostics = parse_json_output(text)
    if diagnostics.valid:
        return diagnostics.parsed
    match = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
    if match:
        return json.loads(match.group(0))
    raise ValueError(diagnostics.parse_error or "No JSON object found")


def classify(label: str, labels: Iterable[str]) -> Dict[str, Any]:
    labels = list(labels)
    found = next((candidate for candidate in labels if candidate.casefold() == str(label).strip().casefold()), None)
    return {"label": found or str(label).strip(), "valid": found is not None, "allowed": labels}


def function_style(name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return {"name": name, "arguments": dict(arguments)}
