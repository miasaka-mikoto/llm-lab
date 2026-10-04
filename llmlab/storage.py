"""SQLite persistence for experiments, runs, datasets and reproducibility data.

The store is intentionally small and dependency-free.  JSON payloads preserve
the complete dataclass shape while indexed columns keep common list/filter
operations fast enough for a desktop research workbench.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Optional

from .models import Dataset, Experiment, ReproducibilityManifest, RunResult, utc_now


class SQLiteStore:
    def __init__(self, path: str | Path = "llmlab.sqlite3") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
              experiment_id TEXT PRIMARY KEY, name TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
              sample_id TEXT, model TEXT, status TEXT, timestamp TEXT,
              payload TEXT NOT NULL,
              FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
            CREATE TABLE IF NOT EXISTS datasets (
              name TEXT PRIMARY KEY, source_path TEXT, sha256 TEXT,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS manifests (
              experiment_id TEXT PRIMARY KEY, dataset_hash TEXT,
              created_at TEXT, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS traces (
              run_id TEXT PRIMARY KEY, experiment_id TEXT, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
              snapshot_id TEXT PRIMARY KEY, experiment_id TEXT, label TEXT,
              created_at TEXT, payload TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def save_experiment(self, experiment: Experiment) -> None:
        payload = experiment.to_dict()
        self.connection.execute(
            "INSERT INTO experiments(experiment_id,name,created_at,updated_at,payload) VALUES(?,?,?,?,?) "
            "ON CONFLICT(experiment_id) DO UPDATE SET name=excluded.name,updated_at=excluded.updated_at,payload=excluded.payload",
            (experiment.experiment_id, experiment.name, experiment.created_at, experiment.updated_at, json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.execute("DELETE FROM runs WHERE experiment_id=?", (experiment.experiment_id,))
        for run in experiment.runs:
            self._insert_run(run)
        self.connection.commit()

    def _insert_run(self, run: RunResult) -> None:
        payload = run.to_dict()
        self.connection.execute(
            "INSERT OR REPLACE INTO runs(run_id,experiment_id,sample_id,model,status,timestamp,payload) VALUES(?,?,?,?,?,?,?)",
            (run.run_id, run.experiment_id, run.sample_id, run.model, run.status, run.timestamp, json.dumps(payload, ensure_ascii=False)),
        )
        if run.trace:
            self.save_trace(run, commit=False)

    def save_run(self, run: RunResult) -> None:
        self._insert_run(run)
        # Keep the denormalised experiment payload in sync when callers append
        # runs incrementally (for example a streaming batch runner).  Without
        # this, ``load_experiment`` after ``save_run`` returned an old run list
        # even though the indexed ``runs`` table was current.
        row = self.connection.execute("SELECT payload FROM experiments WHERE experiment_id=?", (run.experiment_id,)).fetchone()
        if row:
            experiment_payload = json.loads(row["payload"])
            stored_runs = list(experiment_payload.get("runs", []))
            run_payload = run.to_dict()
            replaced = False
            for index, existing in enumerate(stored_runs):
                if existing.get("run_id") == run.run_id:
                    stored_runs[index] = run_payload
                    replaced = True
                    break
            if not replaced:
                stored_runs.append(run_payload)
            experiment_payload["runs"] = stored_runs
            experiment_payload["updated_at"] = utc_now()
            self.connection.execute(
                "UPDATE experiments SET updated_at=?, payload=? WHERE experiment_id=?",
                (experiment_payload["updated_at"], json.dumps(experiment_payload, ensure_ascii=False), run.experiment_id),
            )
        self.connection.commit()

    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        row = self.connection.execute("SELECT payload FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if not row:
            return None
        data = json.loads(row["payload"])
        # The runs table is authoritative for incremental writes.  Older
        # databases may have a stale denormalised payload, so merge the index
        # whenever rows are available.
        indexed_runs = self.connection.execute("SELECT payload FROM runs WHERE experiment_id=? ORDER BY timestamp", (experiment_id,)).fetchall()
        if indexed_runs:
            data["runs"] = [json.loads(item["payload"]) for item in indexed_runs]
        # Nested dataclass reconstruction is kept explicit for compatibility
        # with old exported JSON files.
        from .models import TraceEvent
        runs = []
        for item in data.get("runs", []):
            item = dict(item)
            item["trace"] = [TraceEvent(**event) for event in item.get("trace", [])]
            runs.append(RunResult(**item))
        data["runs"] = runs
        return Experiment(**data)

    def list_experiments(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT experiment_id,name,created_at,updated_at FROM experiments ORDER BY updated_at DESC").fetchall()
        return [{**dict(row), "id": row["experiment_id"]} for row in rows]

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """GUI-friendly mapping view of :meth:`load_experiment`."""
        experiment = self.load_experiment(experiment_id)
        if experiment is None:
            return None
        value = experiment.to_dict()
        value["id"] = value.get("experiment_id", experiment_id)
        return value

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT payload FROM runs WHERE experiment_id=? ORDER BY timestamp", (experiment_id,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT name,source_path,sha256,payload FROM datasets ORDER BY name").fetchall()
        result = []
        for row in rows:
            value = json.loads(row["payload"])
            value.update({"id": row["name"], "rows": len(value.get("records", []))})
            result.append(value)
        return result

    def save_dataset(self, dataset: Dataset) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO datasets(name,source_path,sha256,payload) VALUES(?,?,?,?)",
            (dataset.name, dataset.source_path, dataset.sha256, json.dumps(dataset.to_dict(), ensure_ascii=False)),
        )
        self.connection.commit()

    def load_dataset(self, name: str) -> Optional[Dataset]:
        row = self.connection.execute("SELECT payload FROM datasets WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        from .models import DatasetRecord
        data = json.loads(row["payload"])
        data["records"] = [DatasetRecord(**record) for record in data.get("records", [])]
        return Dataset(**data)

    def save_manifest(self, manifest: ReproducibilityManifest) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO manifests(experiment_id,dataset_hash,created_at,payload) VALUES(?,?,?,?)",
            (manifest.experiment_id, manifest.dataset_hash, manifest.created_at, json.dumps(manifest.to_dict(), ensure_ascii=False)),
        )
        self.connection.commit()

    def load_manifest(self, experiment_id: str) -> Optional[ReproducibilityManifest]:
        row = self.connection.execute("SELECT payload FROM manifests WHERE experiment_id=?", (experiment_id,)).fetchone()
        if not row:
            return None
        return ReproducibilityManifest(**json.loads(row["payload"]))

    def save_trace(self, run: RunResult, *, commit: bool = True) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO traces(run_id,experiment_id,payload) VALUES(?,?,?)",
            (run.run_id, run.experiment_id, json.dumps([asdict(event) for event in run.trace], ensure_ascii=False)),
        )
        if commit:
            self.connection.commit()

    def save_snapshot(self, snapshot_id: str, experiment_id: str, label: str, payload: Any, created_at: str = "") -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO snapshots(snapshot_id,experiment_id,label,created_at,payload) VALUES(?,?,?,?,?)",
            (snapshot_id, experiment_id, label, created_at, json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.commit()

    def load_snapshot(self, snapshot_id: str) -> Optional[Any]:
        row = self.connection.execute("SELECT payload FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
