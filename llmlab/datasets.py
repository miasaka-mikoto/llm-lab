"""Dataset loading, field mapping and local dataset inspection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .models import Dataset, DatasetRecord


def _record_from_mapping(
    item: Mapping[str, Any], index: int, mapping: Optional[Mapping[str, str]] = None
) -> DatasetRecord:
    mapping = mapping or {}
    input_key = mapping.get("input", "input")
    expected_key = mapping.get("expected", "expected")
    context_key = mapping.get("context", "context")
    metadata_key = mapping.get("metadata", "metadata")
    split_key = mapping.get("split", "split")
    metadata = item.get(metadata_key, {})
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}
    reserved = {input_key, expected_key, context_key, metadata_key, split_key, "id", "record_id"}
    metadata = {**{k: v for k, v in item.items() if k not in reserved}, **metadata}
    return DatasetRecord(
        record_id=str(item.get("record_id", item.get("id", index))),
        input=str(item.get(input_key, item.get("text", ""))),
        expected=item.get(expected_key),
        context=item.get(context_key),
        metadata=metadata,
        split=str(item.get(split_key, "")),
    )


def from_records(
    records: Sequence[Mapping[str, Any] | DatasetRecord],
    *,
    name: str = "dataset",
    field_mapping: Optional[Mapping[str, str]] = None,
) -> Dataset:
    converted = [
        item if isinstance(item, DatasetRecord) else _record_from_mapping(item, i, field_mapping)
        for i, item in enumerate(records)
    ]
    canonical = json.dumps([r.to_dict() for r in converted], ensure_ascii=False, sort_keys=True).encode()
    return Dataset(
        name=name,
        records=converted,
        format="memory",
        sha256=hashlib.sha256(canonical).hexdigest(),
        field_mapping=dict(field_mapping or {}),
    )


def load_dataset(
    path: str | Path,
    *,
    name: Optional[str] = None,
    field_mapping: Optional[Mapping[str, str]] = None,
    encoding: str = "utf-8",
) -> Dataset:
    path = Path(path)
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    text = raw.decode(encoding)
    records: List[DatasetRecord] = []
    if suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
        records = [_record_from_mapping(v, i, field_mapping) if isinstance(v, dict) else DatasetRecord(str(i), str(v)) for i, v in enumerate(values)]
        fmt = "jsonl"
    elif suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            value = value["records"]
        if not isinstance(value, list):
            value = [value]
        records = [_record_from_mapping(v, i, field_mapping) if isinstance(v, dict) else DatasetRecord(str(i), str(v)) for i, v in enumerate(value)]
        fmt = "json"
    elif suffix == ".csv":
        rows = csv.DictReader(text.splitlines())
        records = [_record_from_mapping(dict(row), i, field_mapping) for i, row in enumerate(rows)]
        fmt = "csv"
    elif suffix in {".txt", ".text"}:
        records = [DatasetRecord(str(i), line) for i, line in enumerate(text.splitlines()) if line.strip()]
        fmt = "txt"
    else:
        raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}; use JSON, JSONL, CSV or TXT")
    return Dataset(
        name=name or path.stem,
        records=records,
        source_path=str(path),
        format=fmt,
        sha256=hashlib.sha256(raw).hexdigest(),
        field_mapping=dict(field_mapping or {}),
    )


class DatasetViewer:
    """Small view-model used by both CLI and GUI dataset tables."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def page(self, page: int = 1, page_size: int = 50, *, split: str = "") -> List[Dict[str, Any]]:
        rows = self.dataset.records if not split else self.dataset.split_records(split)
        start = max(page - 1, 0) * page_size
        return [row.to_dict() for row in rows[start : start + page_size]]

    def search(self, query: str, *, fields: Iterable[str] = ("input", "expected", "context")) -> List[DatasetRecord]:
        query = query.casefold()
        found = []
        for row in self.dataset.records:
            values = [str(getattr(row, field, "")) for field in fields]
            if any(query in value.casefold() for value in values):
                found.append(row)
        return found


def sample_dataset(size: int = 20, *, name: str = "sample_sentiment") -> Dataset:
    labels = [("I love this product", "positive"), ("This is excellent", "positive"), ("I dislike this", "negative"), ("This is terrible", "negative"), ("It is on the table", "neutral")]
    rows = [{"input": text, "expected": label, "metadata": {"source": "sample", "index": i}} for i in range(size) for text, label in [labels[i % len(labels)]]]
    return from_records(rows, name=name)

