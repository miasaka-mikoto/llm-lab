"""Data models shared by the offline LLM Lab engine.

The models intentionally contain only JSON-serialisable values.  This keeps
experiments portable between the desktop UI, the CLI and exported reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex[:12]
    return f"{prefix}{value}" if prefix else value


@dataclass
class PromptVersion:
    version: str
    system: str = ""
    user: str = ""
    template: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    notes: str = ""

    def render(self, values: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        """Render this prompt without evaluating arbitrary Python expressions."""
        values = {**self.variables, **(values or {})}
        source = self.template or self.user
        try:
            rendered_user = source.format_map(_SafeFormatDict(values))
        except (KeyError, ValueError):
            rendered_user = source
        return self.system, rendered_user


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass
class DatasetRecord:
    record_id: str
    input: str = ""
    expected: Any = None
    context: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    split: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Dataset:
    name: str
    records: List[DatasetRecord] = field(default_factory=list)
    source_path: str = ""
    format: str = "jsonl"
    sha256: str = ""
    field_mapping: Dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.records)

    def split_records(self, split: str) -> List[DatasetRecord]:
        return [r for r in self.records if r.split == split]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class TraceEvent:
    """A public, safe-to-display event in a provider/agent trace.

    ``thought_summary`` is an intentionally short, rule-generated explanation;
    hidden chain-of-thought is never stored by the engine.
    """

    step: int
    kind: str
    action: str = ""
    tool: str = ""
    observation: str = ""
    result: Any = None
    thought_summary: str = ""
    timestamp: str = field(default_factory=utc_now)


@dataclass
class RunResult:
    run_id: str = field(default_factory=lambda: new_id("run_"))
    experiment_id: str = ""
    sample_id: str = ""
    model: str = ""
    provider: str = ""
    prompt_version: str = ""
    input: str = ""
    output: Any = ""
    expected: Any = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    status: str = "success"
    timestamp: str = field(default_factory=utc_now)
    trace: List[TraceEvent] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value


@dataclass
class Experiment:
    experiment_id: str = field(default_factory=lambda: new_id("exp_"))
    name: str = "Untitled Experiment"
    research_question: str = ""
    hypothesis: str = ""
    dataset: str = ""
    model: str = "mock"
    prompt: str = ""
    prompt_version: str = "v1"
    parameters: Dict[str, Any] = field(default_factory=dict)
    evaluator: str = "exact_match"
    metrics: List[str] = field(default_factory=list)
    runs: List[RunResult] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    artifacts: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def add_run(self, run: RunResult) -> None:
        self.runs.append(run)
        self.updated_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value


@dataclass
class ReproducibilityManifest:
    experiment_id: str
    config: Dict[str, Any]
    environment: Dict[str, Any]
    dependency_versions: Dict[str, str]
    dataset_hash: str
    prompt_version: str
    provider: str
    parameters: Dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

