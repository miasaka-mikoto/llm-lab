"""Tkinter desktop workbench for LLM Lab.

The GUI is deliberately adapter based.  The rest of the project can expose a
store/runner later without coupling the presentation layer to a particular
database or provider implementation.  In a clean checkout it falls back to a
fully local MockProvider-like demo, so every screen remains usable offline.

Run directly with::

    python -m llmlab.gui

or::

    python LLMLab/run_gui.py
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import os
import queue
import random
import statistics
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


__all__ = ["GuiAdapter", "LLMLabApp", "capture_screenshot", "run"]


# ---------------------------------------------------------------------------
# Local fallback data and adapter
# ---------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(value: str) -> str:
    chars = [c.lower() if c.isalnum() else "-" for c in value.strip()]
    return "".join(chars).strip("-") or "experiment"


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{digest}"


def _mock_text(model: str, prompt: str, sample: int, temperature: float) -> str:
    """Return deterministic-looking local output; it never contacts a model."""
    seed = hashlib.sha256(f"{model}|{prompt}|{sample}|{temperature}".encode()).hexdigest()
    vocab = ["clear", "measured", "robust", "concise", "structured", "useful", "stable"]
    words = [vocab[int(seed[i : i + 2], 16) % len(vocab)] for i in range(0, 14, 2)]
    return (
        f"[{model} / mock sample {sample}] "
        f"The response is {words[0]}, {words[1]}, and {words[2]}. "
        f"Prompt signal: {prompt[:88].replace(chr(10), ' ')}. "
        f"Local-only evaluation path; no external API was called."
    )


def demo_experiments() -> list[dict[str, Any]]:
    return [
        {
            "id": "exp-prompt-robustness",
            "name": "Prompt Robustness Test",
            "research_question": "Which prompt framing remains stable across paraphrased inputs?",
            "hypothesis": "A concise structured prompt will outperform a terse baseline.",
            "dataset": "sample_robustness_20",
            "model": "Mock-Reasoner",
            "prompt": "Answer the task precisely. Return a short rationale and a final label.",
            "system_prompt": "You are a careful local evaluation model.",
            "user_prompt": "Classify this input: {{input}}",
            "variables": "input",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 256,
            "seed": 42,
            "structured_output": False,
            "metrics": ["Exact Match", "Latency", "Token Count"],
            "created_at": "2026-10-04 10:00:00",
            "updated_at": "2026-10-04 10:00:00",
        },
        {
            "id": "exp-json-extraction",
            "name": "Structured Extraction Smoke Test",
            "research_question": "Can a schema prompt reduce malformed extraction?",
            "hypothesis": "Explicit JSON constraints improve validity in the mock matrix.",
            "dataset": "sample_entities_12",
            "model": "Mock-Structured",
            "prompt": "Extract entities as JSON with keys: name, type, confidence.",
            "system_prompt": "Return JSON only.",
            "user_prompt": "Extract entities from: {{input}}",
            "variables": "input",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "seed": 7,
            "structured_output": True,
            "metrics": ["JSON Validity", "Schema Compliance", "Latency"],
            "created_at": "2026-10-04 10:01:00",
            "updated_at": "2026-10-04 10:01:00",
        },
    ]


def demo_datasets() -> list[dict[str, Any]]:
    return [
        {"id": "sample_robustness_20", "name": "Sample Robustness (20)", "format": "JSONL", "rows": 20, "split": "dev", "hash": "sha256:demo-robustness"},
        {"id": "sample_entities_12", "name": "Sample Entities (12)", "format": "JSON", "rows": 12, "split": "test", "hash": "sha256:demo-entities"},
        {"id": "sample_rag_8", "name": "Sample RAG Notes (8)", "format": "TXT", "rows": 8, "split": "train", "hash": "sha256:demo-rag"},
    ]


class GuiAdapter:
    """A tiny compatibility layer around whichever core the project provides.

    Core implementations can implement any of ``list_experiments``,
    ``get_experiment``, ``save_experiment``, ``list_datasets`` and
    ``run_batch``.  Missing methods gracefully use the local demo store.
    """

    def __init__(self, store: Any = None, runner: Any = None, project_dir: str | Path | None = None):
        self.store = store
        self.runner = runner or store
        self.project_dir = Path(project_dir or Path.cwd())
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._experiments = demo_experiments()
        self._datasets = demo_datasets()
        self._runs: dict[str, list[dict[str, Any]]] = {}

    def _call(self, obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(obj, name, None) if obj is not None else None
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                # Some cores expose a no-argument getter.
                if args or kwargs:
                    try:
                        return fn()
                    except Exception:
                        return None
            except Exception:
                return None
        return None

    @staticmethod
    def _normalise_run(value: Any) -> dict[str, Any]:
        """Flatten canonical RunResult fields for Treeview/report widgets."""
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        row = dict(value or {})
        metrics = row.get("metrics") or {}
        row.setdefault("run_id", row.get("id", ""))
        row.setdefault("score", metrics.get("exact_match", metrics.get("score", 0)))
        row.setdefault("latency_ms", row.get("latency", 0))
        row.setdefault("tokens", row.get("total_tokens", (row.get("input_tokens", 0) or 0) + (row.get("output_tokens", 0) or 0)))
        row.setdefault("prompt", row.get("prompt_version", ""))
        row.setdefault("status", "success" if not row.get("error") else "error")
        return row

    def list_experiments(self) -> list[dict[str, Any]]:
        value = self._call(self.store, "list_experiments")
        if value:
            records: list[dict[str, Any]] = []
            for item in value:
                if hasattr(item, "to_dict"):
                    item = item.to_dict()
                row = dict(item)
                # SQLiteStore uses the dataclass field name while the GUI uses
                # the shorter id consistently in widgets and report paths.
                row.setdefault("id", row.get("experiment_id", ""))
                row.setdefault("name", row.get("experiment_id", "Untitled"))
                records.append(row)
            return records
        return [dict(x) for x in self._experiments]

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        value = self._call(self.store, "get_experiment", exp_id)
        if isinstance(value, Mapping):
            return dict(value)
        # SQLiteStore's public API is load_experiment; accepting it here keeps
        # the UI independent of storage naming and dataclass implementation.
        value = self._call(self.store, "load_experiment", exp_id)
        if value is not None:
            if hasattr(value, "to_dict"):
                data = dict(value.to_dict())
                data.setdefault("id", data.get("experiment_id", exp_id))
                # Expose common parameter controls as top-level values for
                # the editor while retaining the canonical nested dictionary.
                data.update({k: v for k, v in (data.get("parameters") or {}).items() if k in {"temperature", "top_p", "max_tokens", "seed", "structured_output", "temperature_sweep"}})
                return data
            if isinstance(value, Mapping):
                data = dict(value)
                data.setdefault("id", data.get("experiment_id", exp_id))
                return data
        return next((dict(x) for x in self.list_experiments() if x.get("id") == exp_id), None)

    def save_experiment(self, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload["updated_at"] = _now()
        saved = self._call(self.store, "save_experiment", payload)
        if isinstance(saved, Mapping):
            return dict(saved)
        # Convert to the project's dataclass when a SQLiteStore is supplied.
        # The conversion is optional so this module remains importable as a
        # standalone GUI in a minimal Python environment.
        if self.store is not None and hasattr(self.store, "save_experiment"):
            try:
                from .models import Experiment, RunResult

                allowed = {field for field in Experiment.__dataclass_fields__}
                # Pull UI-only scalar controls from the original payload
                # before filtering to dataclass fields.
                ui_parameters = {key: payload[key] for key in ("temperature", "top_p", "max_tokens", "seed", "structured_output", "temperature_sweep") if key in payload}
                clean = {key: value for key, value in payload.items() if key in allowed}
                clean["experiment_id"] = payload.get("id", clean.get("experiment_id", ""))
                # GUI parameters are stored in individual controls; fold them
                # back into the canonical experiment parameter dictionary.
                params = dict(clean.get("parameters") or {})
                params.update(ui_parameters)
                clean["parameters"] = params
                runs = []
                for run in clean.get("runs") or []:
                    if isinstance(run, RunResult):
                        runs.append(run)
                    elif isinstance(run, Mapping):
                        run_copy = dict(run)
                        run_copy.pop("trace", None)  # old GUI rows do not have TraceEvent objects
                        runs.append(RunResult(**{k: v for k, v in run_copy.items() if k in RunResult.__dataclass_fields__}))
                clean["runs"] = runs
                obj = Experiment(**{k: v for k, v in clean.items() if k in allowed})
                self.store.save_experiment(obj)
                payload = obj.to_dict()
                payload["id"] = payload.get("experiment_id", "")
                # Keep the UI's friendly parameter fields available after the
                # round-trip through the canonical storage model.
                payload.update(params)
                return payload
            except Exception:
                # A partially implemented core must not make the editor fail;
                # local fallback persistence below remains available.
                pass
        for i, old in enumerate(self._experiments):
            if old.get("id") == payload.get("id"):
                self._experiments[i] = payload
                break
        else:
            self._experiments.append(payload)
        return payload

    def create_experiment(self, name: str = "Untitled Experiment") -> dict[str, Any]:
        record = {
            "id": _stable_id("exp", f"{name}-{time.time_ns()}"),
            "name": name,
            "research_question": "",
            "hypothesis": "",
            "dataset": self._datasets[0]["id"],
            "model": "Mock-Reasoner",
            "prompt": "",
            "system_prompt": "You are a careful local evaluation model.",
            "user_prompt": "{{input}}",
            "variables": "input",
            "temperature": 0.2,
            "temperature_sweep": "0.0,0.5,1.0",
            "top_p": 0.9,
            "max_tokens": 256,
            "seed": 42,
            "structured_output": False,
            "metrics": ["Exact Match", "Latency"],
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._experiments.append(record)
        return record

    def list_datasets(self) -> list[dict[str, Any]]:
        value = self._call(self.store, "list_datasets")
        if value:
            return [dict(x.to_dict() if hasattr(x, "to_dict") else x) for x in value]
        return [dict(x) for x in self._datasets]

    def runs(self, exp_id: str) -> list[dict[str, Any]]:
        value = self._call(self.store, "list_runs", exp_id)
        if value:
            return [self._normalise_run(x) for x in value]
        loaded = self._call(self.store, "load_experiment", exp_id)
        if loaded is not None:
            try:
                data = loaded.to_dict() if hasattr(loaded, "to_dict") else dict(loaded)
                return [self._normalise_run(x) for x in data.get("runs", [])]
            except Exception:
                pass
        return [self._normalise_run(x) for x in self._runs.get(exp_id, [])]

    def run_batch(self, experiment: Mapping[str, Any], matrix: list[dict[str, Any]], stop_event: threading.Event | None = None, progress: Callable[[dict[str, Any]], None] | None = None, pause_event: threading.Event | None = None) -> list[dict[str, Any]]:
        """Run a local matrix, or delegate to a compatible core runner."""
        delegated = self._call(self.runner, "run_batch", experiment, matrix, stop_event=stop_event, progress=progress)
        if delegated is not None:
            return [self._normalise_run(x) for x in delegated]
        delegated = self._call(self.runner, "run_experiment", experiment, matrix)
        if delegated is not None:
            return [self._normalise_run(x) for x in delegated]
        results: list[dict[str, Any]] = []
        stop_event = stop_event or threading.Event()
        pause_event = pause_event or threading.Event()
        for index, item in enumerate(matrix, 1):
            if stop_event.is_set():
                break
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.04)
            started = time.perf_counter()
            model = str(item.get("model") or experiment.get("model") or "Mock-Reasoner")
            prompt = str(item.get("prompt") or experiment.get("prompt") or experiment.get("user_prompt") or "")
            sample = int(item.get("sample", index))
            temperature = float(item.get("temperature", experiment.get("temperature", 0.2)))
            output = _mock_text(model, prompt, sample, temperature)
            elapsed = (time.perf_counter() - started) * 1000 + 8.0 + ((index * 7) % 37)
            tokens = max(1, len(output.split()))
            score = round(0.68 + ((index * 17) % 28) / 100, 3)
            row = {
                "run_id": _stable_id("run", f"{experiment.get('id')}:{index}:{time.time_ns()}"),
                "experiment_id": experiment.get("id"),
                "model": model,
                "prompt": item.get("prompt_name", f"Prompt {index}"),
                "sample": sample,
                "output": output,
                "score": score,
                "latency_ms": round(elapsed, 2),
                "tokens": tokens,
                "error": "",
                "status": "success",
                "timestamp": _now(),
                "parameters": {"temperature": temperature, "top_p": item.get("top_p", experiment.get("top_p", 0.9))},
            }
            results.append(self._normalise_run(row))
            if progress:
                progress({"index": index, "total": len(matrix), "row": row})
        self._runs.setdefault(str(experiment.get("id")), []).extend(results)
        return results

    def export_report(self, experiment: Mapping[str, Any], rows: Iterable[Mapping[str, Any]], destination: str | Path) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = list(rows)
        scores = [float(r.get("score", 0) or 0) for r in rows]
        mean = statistics.mean(scores) if scores else 0.0
        markdown = [
            f"# {experiment.get('name', 'LLM Lab Experiment')}",
            "",
            f"- **Experiment ID:** `{experiment.get('id', '')}`",
            f"- **Generated:** {_now()}",
            "- **Provider mode:** Local MockProvider (no paid API calls)",
            "",
            "## Research question",
            str(experiment.get("research_question", "")),
            "",
            "## Hypothesis",
            str(experiment.get("hypothesis", "")),
            "",
            "## Summary",
            f"- Runs: {len(rows)}",
            f"- Mean score: {mean:.3f}",
            f"- Success rate: {(sum(1 for r in rows if r.get('status') == 'success') / len(rows) * 100) if rows else 0:.1f}%",
            "",
            "## Results",
            "| Run | Model | Score | Latency (ms) | Tokens | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in rows:
            markdown.append(f"| {row.get('run_id', '')[:12]} | {row.get('model', '')} | {float(row.get('score', 0) or 0):.3f} | {row.get('latency_ms', 0)} | {row.get('tokens', 0)} | {row.get('status', '')} |")
        markdown += ["", "## Limitations", "These results use deterministic mock outputs and are for workflow validation only.", ""]
        dest.write_text("\n".join(markdown), encoding="utf-8")
        return dest


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#10141d")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))


class LLMLabApp(tk.Tk):
    """The desktop research workbench."""

    COLORS = {
        "bg": "#0b0f16",
        "surface": "#111722",
        "surface2": "#171f2d",
        "surface3": "#202b3b",
        "border": "#2a3547",
        "text": "#e7edf7",
        "muted": "#93a1b5",
        "accent": "#63a4ff",
        "accent2": "#7ee2c3",
        "warning": "#f6c667",
        "danger": "#ff7a90",
    }

    def __init__(self, adapter: GuiAdapter | None = None, project_dir: str | Path | None = None):
        super().__init__()
        self.title("LLM Lab · 大语言模型实验室")
        self.geometry("1480x900")
        self.minsize(1120, 700)
        self.configure(bg=self.COLORS["bg"])
        self.adapter = adapter or GuiAdapter(project_dir=project_dir)
        self.project_dir = Path(project_dir or self.adapter.project_dir)
        self.current: dict[str, Any] | None = None
        self.current_rows: list[dict[str, Any]] = []
        self._run_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._running = False
        self._vars: dict[str, tk.Variable] = {}
        self._setup_style()
        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._refresh_sidebar()
        self.after(100, self._drain_events)

    # ---- style and chrome -------------------------------------------------
    def _setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        c = self.COLORS
        style.configure("TFrame", background=c["bg"])
        style.configure("Surface.TFrame", background=c["surface"])
        style.configure("Card.TFrame", background=c["surface2"], relief="flat")
        style.configure("TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=c["bg"], foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabel", background=c["surface2"], foreground=c["text"], font=("Segoe UI", 10, "bold"))
        style.configure("Badge.TLabel", background="#19314e", foreground="#9ec8ff", padding=(8, 3), font=("Segoe UI", 9, "bold"))
        style.configure("TButton", background=c["surface3"], foreground=c["text"], bordercolor=c["border"], padding=(10, 6), font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#2d405b"), ("disabled", c["surface"])], foreground=[("disabled", c["muted"])])
        style.configure("Accent.TButton", background=c["accent"], foreground="#07101d", padding=(12, 7), font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton", background=[("active", "#8bbcff"), ("disabled", "#40526c")])
        style.configure("Danger.TButton", background="#512b39", foreground="#ffb4c1", padding=(10, 6))
        style.configure("TEntry", fieldbackground=c["surface"], foreground=c["text"], insertcolor=c["text"], bordercolor=c["border"], padding=6)
        style.configure("TCombobox", fieldbackground=c["surface"], background=c["surface3"], foreground=c["text"], arrowcolor=c["muted"], padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", c["surface"])], foreground=[("readonly", c["text"])])
        style.configure("TCheckbutton", background=c["surface2"], foreground=c["text"])
        style.configure("TRadiobutton", background=c["surface2"], foreground=c["text"])
        style.configure("TNotebook", background=c["bg"], bordercolor=c["border"])
        style.configure("TNotebook.Tab", background=c["surface2"], foreground=c["muted"], padding=(13, 7))
        style.map("TNotebook.Tab", background=[("selected", c["surface3"])], foreground=[("selected", c["text"])])
        style.configure("Treeview", background=c["surface"], fieldbackground=c["surface"], foreground=c["text"], bordercolor=c["border"], rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=c["surface3"], foreground=c["muted"], relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#24476d")], foreground=[("selected", "#ffffff")])
        style.configure("Vertical.TScrollbar", background=c["surface3"], troughcolor=c["surface"], arrowcolor=c["muted"])

    def _build_menu(self) -> None:
        menu = tk.Menu(self, tearoff=False, bg=self.COLORS["surface"], fg=self.COLORS["text"], activebackground="#24476d", activeforeground="#fff")
        file_menu = tk.Menu(menu, tearoff=False, bg=self.COLORS["surface"], fg=self.COLORS["text"], activebackground="#24476d")
        file_menu.add_command(label="New experiment", command=self._new_experiment)
        file_menu.add_command(label="Save experiment", command=self._save_experiment)
        file_menu.add_separator()
        file_menu.add_command(label="Export Markdown report", command=self._export_report)
        file_menu.add_command(label="Import dataset…", command=self._import_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        view_menu = tk.Menu(menu, tearoff=False, bg=self.COLORS["surface"], fg=self.COLORS["text"], activebackground="#24476d")
        view_menu.add_command(label="Refresh", command=self._refresh_sidebar)
        view_menu.add_command(label="Focus prompt editor", command=lambda: self.prompt_text.focus_set())
        menu.add_cascade(label="View", menu=view_menu)
        help_menu = tk.Menu(menu, tearoff=False, bg=self.COLORS["surface"], fg=self.COLORS["text"], activebackground="#24476d")
        help_menu.add_command(label="About LLM Lab", command=lambda: messagebox.showinfo("LLM Lab", "Local-first LLM experiment workbench\nMockProvider enabled · no paid API calls"))
        menu.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menu)

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=self.COLORS["surface"], height=50, highlightthickness=1, highlightbackground=self.COLORS["border"])
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(9, weight=1)
        tk.Label(bar, text="LLM LAB", bg=self.COLORS["surface"], fg=self.COLORS["accent"], font=("Segoe UI", 13, "bold")).grid(row=0, column=0, padx=(18, 6), pady=10)
        tk.Label(bar, text="大语言模型实验室", bg=self.COLORS["surface"], fg=self.COLORS["muted"], font=("Microsoft YaHei UI", 10)).grid(row=0, column=1, padx=5)
        ttk.Label(bar, text="LOCAL · MOCK ONLY", style="Badge.TLabel").grid(row=0, column=2, padx=18)
        ttk.Button(bar, text="＋ New", command=self._new_experiment).grid(row=0, column=3, padx=4)
        self.run_button = ttk.Button(bar, text="▶ Run matrix", style="Accent.TButton", command=self._run_matrix)
        self.run_button.grid(row=0, column=4, padx=4)
        ttk.Button(bar, text="Ⅱ Pause", command=self._pause_matrix).grid(row=0, column=5, padx=2)
        ttk.Button(bar, text="▶ Resume", command=self._resume_matrix).grid(row=0, column=6, padx=2)
        ttk.Button(bar, text="↻ Retry", command=self._retry_matrix).grid(row=0, column=7, padx=2)
        ttk.Button(bar, text="■ Stop", style="Danger.TButton", command=self._stop_matrix).grid(row=0, column=8, padx=4, sticky="w")
        self.progress = ttk.Progressbar(bar, orient="horizontal", mode="determinate", length=180)
        self.progress.grid(row=0, column=9, padx=(4, 16))

    def _build_body(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self._build_sidebar(body).grid(row=0, column=0, sticky="nsew")
        self._build_center(body).grid(row=0, column=1, sticky="nsew")
        self._build_inspector(body).grid(row=0, column=2, sticky="nsew")

    def _build_sidebar(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=12)
        frame.configure(width=225)
        frame.grid_propagate(False)
        ttk.Label(frame, text="RESEARCH SPACE", style="Muted.TLabel").pack(anchor="w", pady=(4, 8))
        for label in ("▦  Projects", "◈  Experiments", "▤  Datasets", "◌  Templates"):
            btn = ttk.Button(frame, text=label, command=lambda: self._log(f"Opened {label.strip('▦◈▤◌ ')}"))
            btn.pack(fill="x", pady=2)
        ttk.Separator(frame).pack(fill="x", pady=12)
        ttk.Label(frame, text="EXPERIMENTS", style="Muted.TLabel").pack(anchor="w")
        list_frame = ttk.Frame(frame, style="Surface.TFrame")
        list_frame.pack(fill="both", expand=True, pady=(6, 8))
        self.exp_list = tk.Listbox(list_frame, bg=self.COLORS["surface"], fg=self.COLORS["text"], selectbackground="#24476d", selectforeground="#fff", relief="flat", borderwidth=0, highlightthickness=0, activestyle="none", font=("Segoe UI", 10))
        exp_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.exp_list.yview)
        self.exp_list.configure(yscrollcommand=exp_scroll.set)
        self.exp_list.pack(side="left", fill="both", expand=True)
        exp_scroll.pack(side="right", fill="y")
        self.exp_list.bind("<<ListboxSelect>>", self._select_experiment)
        ttk.Button(frame, text="＋  New experiment", command=self._new_experiment).pack(fill="x")
        ttk.Label(frame, text="DATASETS", style="Muted.TLabel").pack(anchor="w", pady=(18, 4))
        self.dataset_list = tk.Listbox(frame, height=5, bg=self.COLORS["surface"], fg=self.COLORS["muted"], selectbackground="#24476d", relief="flat", borderwidth=0, highlightthickness=0, font=("Segoe UI", 9))
        self.dataset_list.pack(fill="x")
        return frame

    def _build_center(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=(12, 10, 12, 8))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.grid_columnconfigure(0, weight=1)
        self.workspace_title = ttk.Label(header, text="Experiment workspace", style="Title.TLabel")
        self.workspace_title.grid(row=0, column=0, sticky="w")
        self.workspace_meta = ttk.Label(header, text="Select an experiment to begin", style="Muted.TLabel")
        self.workspace_meta.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.editor_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.charts_tab = ttk.Frame(self.notebook)
        self.report_tab = ttk.Frame(self.notebook)
        self.labs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.editor_tab, text="  Editor  ")
        self.notebook.add(self.results_tab, text="  Results  ")
        self.notebook.add(self.charts_tab, text="  Charts  ")
        self.notebook.add(self.report_tab, text="  Report  ")
        self.notebook.add(self.labs_tab, text="  Labs  ")
        self._build_editor(self.editor_tab)
        self._build_results(self.results_tab)
        self._build_charts(self.charts_tab)
        self._build_report(self.report_tab)
        self._build_labs(self.labs_tab)
        self._build_bottom(frame)
        return frame

    def _build_labs(self, parent: ttk.Frame) -> None:
        """Small offline probes for RAG, Agent, Memory and structured labs."""
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        head = ttk.Frame(parent, style="Card.TFrame", padding=12)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(head, text="SPECIALISED LABS", style="Section.TLabel").pack(side="left")
        ttk.Label(head, text="All probes use local deterministic components", style="Muted.TLabel").pack(side="left", padx=14)
        self.lab_choice = ttk.Combobox(head, state="readonly", values=["RAG Lab", "Agent Lab", "Memory Lab", "Structured Output Lab", "Benchmark Dashboard"], width=24)
        self.lab_choice.set("RAG Lab")
        self.lab_choice.pack(side="right", padx=(8, 0))
        ttk.Button(head, text="Run local probe", style="Accent.TButton", command=self._run_lab_probe).pack(side="right")
        body = ttk.Frame(parent, style="Card.TFrame", padding=14)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.lab_description = ttk.Label(body, text="RAG pipeline · document → chunk → fake embedding → retrieve → answer", style="Muted.TLabel")
        self.lab_description.grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.lab_output = tk.Text(body, wrap="word", bg=self.COLORS["surface"], fg=self.COLORS["text"], relief="flat", state="disabled", padx=12, pady=10, font=("Cascadia Code", 10))
        self.lab_output.grid(row=1, column=0, sticky="nsew")

    def _run_lab_probe(self) -> None:
        choice = self.lab_choice.get()
        descriptions = {
            "RAG Lab": "RAG pipeline · document → chunk → fake embedding → retrieve → answer",
            "Agent Lab": "Public trace only · planner → tool → observation → final (no hidden chain-of-thought)",
            "Memory Lab": "Compare bounded No Memory / Sliding Window / Summary / Vector strategies",
            "Structured Output Lab": "Parse, validate and diagnose JSON/schema output",
            "Benchmark Dashboard": "Aggregate current local runs by model and metric",
        }
        self.lab_description.configure(text=descriptions.get(choice, "Local probe"))
        result: Any
        try:
            if choice == "RAG Lab":
                from .labs import RAGLab
                from .rag import Document

                lab = RAGLab()
                indexed = lab.index([Document("demo", "A local retrieval note about reproducible prompt experiments."), Document("guide", "Chunking and overlap affect recall." )])
                answer = lab.query("What affects recall?", top_k=2)
                result = {"indexed_documents": indexed, "answer": answer.answer, "retrieved": [item.chunk.text for item in answer.retrievals]}
            elif choice == "Agent Lab":
                from .labs import AgentLab

                run = AgentLab().run("Inspect a local experiment")
                result = {"status": run.status, "final": run.final, "public_trace": [event.__dict__ if hasattr(event, "__dict__") else str(event) for event in run.trace]}
            elif choice == "Memory Lab":
                from .labs import MemoryLab

                lab = MemoryLab(strategy="sliding_window", capacity=4)
                for i in range(6):
                    lab.remember(f"local event {i}", memory_id=f"m{i}", importance=i / 6, step=i)
                result = {"strategy": "sliding_window", "stored": 6, "retrieved": [item.text for item in lab.recall("event", limit=4, step=6)]}
            elif choice == "Structured Output Lab":
                from .structured import parse_json_output

                diagnostics = parse_json_output('{"label":"positive","confidence":0.91}', {"type": "object", "required": ["label", "confidence"]})
                result = diagnostics.__dict__ if hasattr(diagnostics, "__dict__") else str(diagnostics)
            else:
                rows = self.current_rows
                grouped: dict[str, list[float]] = {}
                for row in rows:
                    grouped.setdefault(str(row.get("model", "unknown")), []).append(float(row.get("score", 0) or 0))
                result = {"runs": len(rows), "models": {model: round(statistics.mean(scores), 3) for model, scores in grouped.items()}}
        except Exception as exc:
            result = {"error": str(exc)}
        self._set_text(self.lab_output, json.dumps(result, indent=2, ensure_ascii=False, default=str))
        self._log(f"Ran {choice} offline probe")

    def _build_bottom(self, parent: ttk.Frame) -> None:
        """Build the persistent Runs / Logs / Trace strip below the tabs."""
        bottom = ttk.Notebook(parent, height=142)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        runs_tab = ttk.Frame(bottom)
        logs_tab = ttk.Frame(bottom)
        trace_tab = ttk.Frame(bottom)
        bottom.add(runs_tab, text="  Runs  ")
        bottom.add(logs_tab, text="  Logs  ")
        bottom.add(trace_tab, text="  Trace  ")
        runs_tab.grid_rowconfigure(0, weight=1)
        runs_tab.grid_columnconfigure(0, weight=1)
        self.run_history_tree = ttk.Treeview(runs_tab, columns=("run", "status", "model", "score", "latency"), show="headings", height=4)
        for col, title, width in (("run", "Run", 140), ("status", "Status", 90), ("model", "Model", 160), ("score", "Score", 80), ("latency", "Latency", 100)):
            self.run_history_tree.heading(col, text=title)
            self.run_history_tree.column(col, width=width, anchor="w")
        self.run_history_tree.grid(row=0, column=0, sticky="nsew")
        logs_tab.grid_rowconfigure(0, weight=1)
        logs_tab.grid_columnconfigure(0, weight=1)
        self.log_text = tk.Text(logs_tab, height=5, wrap="word", bg=self.COLORS["surface"], fg=self.COLORS["muted"], relief="flat", state="normal", padx=8, pady=5, font=("Cascadia Code", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        trace_tab.grid_rowconfigure(0, weight=1)
        trace_tab.grid_columnconfigure(0, weight=1)
        self.trace_text = tk.Text(trace_tab, height=5, wrap="word", bg=self.COLORS["surface"], fg=self.COLORS["muted"], relief="flat", state="disabled", padx=8, pady=5, font=("Cascadia Code", 9))
        self.trace_text.grid(row=0, column=0, sticky="nsew")

    def _build_editor(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        top = ttk.Frame(parent, style="Card.TFrame", padding=12)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            top.grid_columnconfigure(col, weight=1 if col in (1, 3) else 0)
        self._vars["name"] = tk.StringVar()
        self._vars["research_question"] = tk.StringVar()
        self._vars["hypothesis"] = tk.StringVar()
        self._vars["dataset"] = tk.StringVar()
        self._vars["model"] = tk.StringVar(value="Mock-Reasoner")
        self._field(top, 0, 0, "Name", self._vars["name"])
        self._field(top, 0, 2, "Dataset", self._vars["dataset"], combo=True)
        self._field(top, 1, 0, "Research question", self._vars["research_question"])
        self._field(top, 1, 2, "Model / Provider", self._vars["model"], combo=True, values=["Mock-Reasoner", "Mock-Structured", "Mock-Fast"])
        self._field(top, 2, 0, "Hypothesis", self._vars["hypothesis"])
        ttk.Button(top, text="Save", command=self._save_experiment).grid(row=2, column=3, sticky="e", padx=(8, 0), pady=5)

        prompt_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        prompt_card.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        prompt_card.grid_columnconfigure(0, weight=1)
        prompt_card.grid_rowconfigure(2, weight=1)
        ttk.Label(prompt_card, text="PROMPT PLAYGROUND", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        prompt_info = ttk.Frame(prompt_card, style="Card.TFrame")
        prompt_info.grid(row=1, column=0, sticky="ew", pady=(7, 4))
        prompt_info.grid_columnconfigure(1, weight=1)
        prompt_info.grid_columnconfigure(3, weight=1)
        ttk.Label(prompt_info, text="Version", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.prompt_version = ttk.Combobox(prompt_info, values=["v1", "v2", "v3"], state="readonly", width=8)
        self.prompt_version.set("v1")
        self.prompt_version.grid(row=0, column=1, sticky="w", padx=(6, 20))
        ttk.Label(prompt_info, text="Variables", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.variables_entry = ttk.Entry(prompt_info)
        self.variables_entry.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        self.prompt_text = tk.Text(prompt_card, height=8, wrap="word", undo=True, bg=self.COLORS["surface"], fg=self.COLORS["text"], insertbackground=self.COLORS["text"], selectbackground="#24476d", relief="flat", padx=10, pady=8, font=("Cascadia Code", 10))
        self.prompt_text.grid(row=2, column=0, sticky="nsew")
        prompt_scroll = ttk.Scrollbar(prompt_card, orient="vertical", command=self.prompt_text.yview)
        prompt_scroll.grid(row=2, column=1, sticky="ns")
        self.prompt_text.configure(yscrollcommand=prompt_scroll.set)

        matrix_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        matrix_card.grid(row=2, column=0, sticky="nsew")
        matrix_card.grid_rowconfigure(1, weight=1)
        matrix_card.grid_columnconfigure(0, weight=1)
        matrix_head = ttk.Frame(matrix_card, style="Card.TFrame")
        matrix_head.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        ttk.Label(matrix_head, text="BATCH MATRIX", style="Section.TLabel").pack(side="left")
        ttk.Label(matrix_head, text="Models × prompts × temperatures × samples", style="Muted.TLabel").pack(side="left", padx=15)
        ttk.Button(matrix_head, text="Build matrix", command=self._build_matrix).pack(side="right")
        cols = ("model", "prompt", "temperature", "samples", "status")
        self.matrix_tree = ttk.Treeview(matrix_card, columns=cols, show="headings", height=5)
        for col, title, width in (("model", "Model", 150), ("prompt", "Prompt variant", 180), ("temperature", "Temperature", 90), ("samples", "Samples", 70), ("status", "Status", 100)):
            self.matrix_tree.heading(col, text=title)
            self.matrix_tree.column(col, width=width, anchor="w")
        self.matrix_tree.grid(row=1, column=0, sticky="nsew")

    def _field(self, parent: ttk.Frame, row: int, label_col: int, label: str, variable: tk.Variable, combo: bool = False, values: list[str] | None = None) -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=label_col, sticky="w", padx=(0, 7), pady=5)
        col = label_col + 1
        if combo:
            widget = ttk.Combobox(parent, textvariable=variable, values=values or [x["id"] for x in self.adapter.list_datasets()], state="normal")
        else:
            widget = ttk.Entry(parent, textvariable=variable)
        widget.grid(row=row, column=col, sticky="ew", padx=(0, 18), pady=3)

    def _build_results(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        summary = ttk.Frame(parent, style="Card.TFrame", padding=10)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.summary_labels: dict[str, ttk.Label] = {}
        for i, (key, title) in enumerate((("runs", "Runs"), ("mean", "Mean score"), ("latency", "Median latency"), ("success", "Success rate"))):
            card = ttk.Frame(summary, style="Card.TFrame", padding=(12, 6))
            card.pack(side="left", fill="x", expand=True, padx=3)
            ttk.Label(card, text=title, style="Muted.TLabel").pack(anchor="w")
            label = ttk.Label(card, text="—", font=("Segoe UI", 15, "bold"), background=self.COLORS["surface2"], foreground=self.COLORS["text"])
            label.pack(anchor="w", pady=(3, 0))
            self.summary_labels[key] = label
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        cols = ("run", "model", "prompt", "score", "latency", "tokens", "status")
        self.results_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="extended")
        for col, title, width in (("run", "Run", 120), ("model", "Model", 125), ("prompt", "Prompt", 140), ("score", "Score", 70), ("latency", "Latency ms", 90), ("tokens", "Tokens", 70), ("status", "Status", 90)):
            self.results_tree.heading(col, text=title)
            self.results_tree.column(col, width=width, anchor="w")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        ys.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=ys.set)
        self.results_tree.bind("<<TreeviewSelect>>", self._show_selected_trace)
        compare = ttk.Frame(parent, style="Card.TFrame", padding=8)
        compare.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        compare.grid_columnconfigure(0, weight=1)
        compare.grid_columnconfigure(1, weight=1)
        ttk.Label(compare, text="SIDE-BY-SIDE OUTPUT COMPARE  · select two runs", style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.compare_left = tk.Text(compare, height=5, wrap="word", bg=self.COLORS["surface"], fg=self.COLORS["text"], relief="flat", state="disabled", padx=8, pady=6)
        self.compare_right = tk.Text(compare, height=5, wrap="word", bg=self.COLORS["surface"], fg=self.COLORS["text"], relief="flat", state="disabled", padx=8, pady=6)
        self.compare_left.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(5, 0))
        self.compare_right.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(5, 0))

    def _build_charts(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.chart_canvas = tk.Canvas(parent, bg=self.COLORS["surface"], highlightthickness=0)
        self.chart_canvas.grid(row=0, column=0, sticky="nsew")
        self.chart_canvas.bind("<Configure>", lambda _e: self._draw_charts())

    def _build_report(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.report_text = tk.Text(parent, wrap="word", bg=self.COLORS["surface"], fg=self.COLORS["text"], insertbackground=self.COLORS["text"], relief="flat", padx=18, pady=15, font=("Cascadia Code", 10))
        self.report_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.report_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.report_text.configure(yscrollcommand=scroll.set)

    def _build_inspector(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=12)
        frame.configure(width=285)
        frame.grid_propagate(False)
        ttk.Label(frame, text="INSPECTOR", style="Muted.TLabel").pack(anchor="w", pady=(4, 10))
        param_card = ttk.Frame(frame, style="Card.TFrame", padding=10)
        param_card.pack(fill="x", pady=(0, 10))
        ttk.Label(param_card, text="PARAMETERS", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self._param_vars: dict[str, tk.StringVar] = {}
        for key, label, default in (("temperature", "Temperature", "0.2"), ("top_p", "Top P", "0.9"), ("max_tokens", "Max tokens", "256"), ("seed", "Seed", "42")):
            row = ttk.Frame(param_card, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Muted.TLabel").pack(side="left")
            var = tk.StringVar(value=default)
            self._param_vars[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side="right")
        sweep_row = ttk.Frame(param_card, style="Card.TFrame")
        sweep_row.pack(fill="x", pady=3)
        ttk.Label(sweep_row, text="Temperature sweep", style="Muted.TLabel").pack(side="left")
        self.temperature_sweep_var = tk.StringVar(value="0.2")
        ttk.Entry(sweep_row, textvariable=self.temperature_sweep_var, width=14).pack(side="right")
        ttk.Label(param_card, text="comma-separated values · e.g. 0.0, 0.5, 1.0", style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        self.structured_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_card, text="Structured output", variable=self.structured_var).pack(anchor="w", pady=(7, 0))
        eval_card = ttk.Frame(frame, style="Card.TFrame", padding=10)
        eval_card.pack(fill="x", pady=(0, 10))
        ttk.Label(eval_card, text="EVALUATOR", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.metric_vars: dict[str, tk.BooleanVar] = {}
        for metric in ("Exact Match", "Contains", "Regex", "JSON Validity", "Schema Compliance", "Length", "Latency", "Token Count", "Cost Estimate"):
            var = tk.BooleanVar(value=metric in ("Exact Match", "Latency", "Token Count"))
            self.metric_vars[metric] = var
            ttk.Checkbutton(eval_card, text=metric, variable=var).pack(anchor="w", pady=1)
        meta_card = ttk.Frame(frame, style="Card.TFrame", padding=10)
        meta_card.pack(fill="both", expand=True)
        ttk.Label(meta_card, text="METADATA", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.metadata_text = tk.Text(meta_card, height=8, bg=self.COLORS["surface"], fg=self.COLORS["muted"], relief="flat", wrap="word", state="disabled", padx=7, pady=6, font=("Cascadia Code", 9))
        self.metadata_text.pack(fill="both", expand=True)
        return frame

    def _build_statusbar(self) -> None:
        self.status_var = tk.StringVar(value="Ready · offline mode · MockProvider")
        bar = ttk.Frame(self, style="Surface.TFrame", padding=(14, 5))
        bar.grid(row=2, column=0, sticky="ew")
        ttk.Label(bar, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")
        ttk.Label(bar, text="No paid API calls", style="Muted.TLabel").pack(side="right")

    # ---- data and interaction --------------------------------------------
    def _refresh_sidebar(self) -> None:
        exps = self.adapter.list_experiments()
        self.exp_list.delete(0, "end")
        for exp in exps:
            self.exp_list.insert("end", f"{exp.get('name', 'Untitled')}\n  {exp.get('id', '')[:20]}")
        self._exp_records = exps
        self.dataset_list.delete(0, "end")
        for ds in self.adapter.list_datasets():
            self.dataset_list.insert("end", f"{ds.get('name', ds.get('id', ''))}  ·  {ds.get('rows', 0)} rows")
        if exps and self.current is None:
            self.exp_list.selection_set(0)
            self._select_experiment()

    def _select_experiment(self, _event: tk.Event | None = None) -> None:
        selection = self.exp_list.curselection()
        if not selection:
            return
        exp = self._exp_records[selection[0]]
        self.current = self.adapter.get_experiment(str(exp.get("id"))) or dict(exp)
        self._load_current()

    def _load_current(self) -> None:
        exp = self.current or {}
        self.workspace_title.configure(text=str(exp.get("name", "Experiment workspace")))
        self.workspace_meta.configure(text=f"{exp.get('id', 'new')}  ·  updated {exp.get('updated_at', '—')}  ·  local mock provider")
        for key in ("name", "research_question", "hypothesis", "dataset", "model"):
            self._vars[key].set(str(exp.get(key, "")))
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", str(exp.get("prompt") or exp.get("user_prompt") or ""))
        self.variables_entry.delete(0, "end")
        self.variables_entry.insert(0, str(exp.get("variables", "input")))
        self.prompt_version.set(str(exp.get("prompt_version", "v1")))
        for key in self._param_vars:
            self._param_vars[key].set(str(exp.get(key, self._param_vars[key].get())))
        self.temperature_sweep_var.set(str(exp.get("temperature_sweep", exp.get("temperature", "0.2"))))
        self.structured_var.set(bool(exp.get("structured_output", False)))
        metrics = set(exp.get("metrics", []))
        for key, var in self.metric_vars.items():
            var.set(key in metrics)
        self._set_metadata(exp)
        self._build_matrix()
        self.current_rows = self.adapter.runs(str(exp.get("id")))
        self._refresh_results()
        self._log(f"Loaded {exp.get('name', 'experiment')}")

    def _set_metadata(self, exp: Mapping[str, Any]) -> None:
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", "end")
        metadata = {
            "experiment_id": exp.get("id", ""),
            "provider": "MockProvider",
            "mode": "offline / deterministic",
            "dataset_hash": next((d.get("hash", "") for d in self.adapter.list_datasets() if d.get("id") == exp.get("dataset")), ""),
            "prompt_version": exp.get("prompt_version", "v1"),
            "created_at": exp.get("created_at", ""),
        }
        self.metadata_text.insert("1.0", json.dumps(metadata, indent=2, ensure_ascii=False))
        self.metadata_text.configure(state="disabled")

    def _new_experiment(self) -> None:
        self.current = self.adapter.create_experiment("Untitled Experiment")
        self._refresh_sidebar()
        for i, exp in enumerate(self._exp_records):
            if exp.get("id") == self.current.get("id"):
                self.exp_list.selection_clear(0, "end")
                self.exp_list.selection_set(i)
                self.exp_list.see(i)
                break
        self._load_current()
        self._log("Created new experiment")

    def _save_experiment(self) -> None:
        if not self.current:
            self._new_experiment()
        assert self.current is not None
        exp = dict(self.current)
        for key in ("name", "research_question", "hypothesis", "dataset", "model"):
            exp[key] = self._vars[key].get()
        exp["prompt"] = self.prompt_text.get("1.0", "end-1c")
        exp["user_prompt"] = exp["prompt"]
        exp["variables"] = self.variables_entry.get()
        exp["prompt_version"] = self.prompt_version.get()
        for key, var in self._param_vars.items():
            try:
                exp[key] = float(var.get()) if key in ("temperature", "top_p") else int(var.get())
            except ValueError:
                pass
        exp["temperature_sweep"] = self.temperature_sweep_var.get()
        exp["structured_output"] = bool(self.structured_var.get())
        exp["metrics"] = [key for key, var in self.metric_vars.items() if var.get()]
        self.current = self.adapter.save_experiment(exp)
        self._refresh_sidebar()
        self._set_metadata(self.current)
        self.workspace_title.configure(text=self.current.get("name", "Experiment workspace"))
        self.status_var.set("Saved · configuration is reproducible")
        self._log("Saved experiment configuration")

    def _build_matrix(self) -> None:
        self.matrix_tree.delete(*self.matrix_tree.get_children())
        exp = self.current or {}
        models = [str(exp.get("model") or "Mock-Reasoner"), "Mock-Structured", "Mock-Fast"]
        prompts = ["Baseline", "Structured", "Chain-of-verification", "Minimal"]
        if str(exp.get("name", "")) != "Prompt Robustness Test":
            # Generic experiments expose the documented 5-prompt sweep; the
            # bundled demo intentionally stays at four variants.
            prompts.append("Few-shot")
        try:
            sweep_text = self.temperature_sweep_var.get() if hasattr(self, "temperature_sweep_var") else str(exp.get("temperature", 0.2))
            temperatures = [float(value.strip()) for value in str(sweep_text).split(",") if value.strip()]
        except ValueError:
            temperatures = [float(exp.get("temperature", 0.2))]
        temperatures = temperatures or [float(exp.get("temperature", 0.2))]
        self._matrix: list[dict[str, Any]] = []
        for model in models:
            for prompt in prompts:
                # Each listed temperature becomes a separate matrix cell.
                # The bundled demo defaults to one temperature (240 runs); a
                # new experiment can enter 3 values for the canonical sweep.
                for temperature in temperatures:
                    item = {"model": model, "prompt_name": prompt, "prompt": f"{prompt}: {exp.get('prompt') or exp.get('user_prompt') or '{{input}}'}", "temperature": temperature, "samples": 20}
                    self._matrix.append(item)
                    self.matrix_tree.insert("", "end", values=(model, prompt, f"{temperature:.2f}", 20, "queued"))
        self.status_var.set(f"Matrix ready · {len(self._matrix)} cells · local mock")

    def _run_matrix(self) -> None:
        if self._running:
            return
        if not self.current:
            self._new_experiment()
        self._save_experiment()
        matrix = []
        for item in getattr(self, "_matrix", []):
            for sample in range(1, int(item.get("samples", 1)) + 1):
                row = dict(item)
                row["sample"] = sample
                matrix.append(row)
        if not matrix:
            self._build_matrix()
            matrix = list(getattr(self, "_matrix", []))
        self._running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self.progress.configure(maximum=max(1, len(matrix)), value=0)
        self.run_button.configure(state="disabled")
        self.status_var.set(f"Running {len(matrix)} cells · MockProvider")
        self._log(f"Started batch matrix ({len(matrix)} runs)")

        def worker() -> None:
            try:
                rows = self.adapter.run_batch(self.current or {}, matrix, stop_event=self._stop_event, pause_event=self._pause_event, progress=lambda p: self._events.put({"kind": "progress", **p}))
                self._events.put({"kind": "done", "rows": rows})
            except Exception as exc:  # keep GUI alive and expose trace
                self._events.put({"kind": "error", "error": f"{exc}\n{traceback.format_exc()}"})

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _stop_matrix(self) -> None:
        if self._running:
            self._stop_event.set()
            self.status_var.set("Stopping after current run…")
            self._log("Stop requested")

    def _pause_matrix(self) -> None:
        if self._running:
            self._pause_event.set()
            pause = getattr(self.adapter.runner, "pause", None)
            if callable(pause):
                try:
                    pause()
                except Exception:
                    pass
            self.status_var.set("Paused · resume keeps completed runs")
            self._log("Pause requested")

    def _resume_matrix(self) -> None:
        if self._running:
            self._pause_event.clear()
            resume = getattr(self.adapter.runner, "resume", None)
            if callable(resume):
                try:
                    resume()
                except Exception:
                    pass
            self.status_var.set("Running · resumed")
            self._log("Resume requested")

    def _retry_matrix(self) -> None:
        """Retry failed cells without requiring a new experiment."""
        if self._running:
            return
        failed = [row for row in self.current_rows if row.get("status") != "success"]
        if failed:
            self._log(f"Retrying {len(failed)} failed runs")
        else:
            self._log("No failed runs; starting the current matrix")
        self._run_matrix()

    def _drain_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                kind = event.get("kind")
                if kind == "progress":
                    self.progress.configure(value=event.get("index", 0))
                    row = event.get("row") or {}
                    self._log(f"Run {event.get('index')}/{event.get('total')} · {row.get('model')} · score {row.get('score')}")
                elif kind == "done":
                    self._running = False
                    self.run_button.configure(state="normal")
                    self.current_rows = list(event.get("rows", []))
                    self._refresh_results()
                    self._update_report()
                    self.status_var.set(f"Completed · {len(self.current_rows)} local runs · no paid API calls")
                    self.notebook.select(self.results_tab)
                elif kind == "error":
                    self._running = False
                    self.run_button.configure(state="normal")
                    self.status_var.set("Run failed · see Logs")
                    self._log(event.get("error", "Unknown error"))
                    messagebox.showerror("Batch run failed", str(event.get("error", "Unknown error")).split("Traceback", 1)[0])
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _refresh_results(self) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        if hasattr(self, "run_history_tree"):
            self.run_history_tree.delete(*self.run_history_tree.get_children())
        rows = self.current_rows
        for index, row in enumerate(rows):
            iid = str(index)
            self.results_tree.insert("", "end", iid=iid, values=(str(row.get("run_id", ""))[:12], row.get("model", ""), row.get("prompt", ""), f"{float(row.get('score', 0) or 0):.3f}", row.get("latency_ms", ""), row.get("tokens", ""), row.get("status", "")))
            if hasattr(self, "run_history_tree"):
                self.run_history_tree.insert("", "end", values=(str(row.get("run_id", ""))[:16], row.get("status", ""), row.get("model", ""), f"{float(row.get('score', 0) or 0):.3f}", f"{float(row.get('latency_ms', 0) or 0):.1f} ms"))
        scores = [float(r.get("score", 0) or 0) for r in rows]
        latencies = [float(r.get("latency_ms", 0) or 0) for r in rows]
        self.summary_labels["runs"].configure(text=str(len(rows)))
        self.summary_labels["mean"].configure(text=f"{statistics.mean(scores):.3f}" if scores else "—")
        self.summary_labels["latency"].configure(text=f"{statistics.median(latencies):.1f} ms" if latencies else "—")
        self.summary_labels["success"].configure(text=f"{sum(1 for r in rows if r.get('status') == 'success') / len(rows) * 100:.1f}%" if rows else "—")
        self._draw_charts()

    def _show_selected_trace(self, _event: tk.Event | None = None) -> None:
        selected = self.results_tree.selection()
        if selected and hasattr(self, "trace_text"):
            row = self.current_rows[int(selected[0])]
            trace = row.get("trace", [])
            if hasattr(trace, "to_dict"):
                trace = trace.to_dict()
            trace_payload = {
                "run_id": row.get("run_id", ""),
                "experiment_id": row.get("experiment_id", self.current.get("id", "") if self.current else ""),
                "input": row.get("input", ""),
                "model": row.get("model", ""),
                "parameters": row.get("parameters", {}),
                "output": row.get("output", ""),
                "latency_ms": row.get("latency_ms", 0),
                "tokens": row.get("tokens", row.get("total_tokens", 0)),
                "error": row.get("error", ""),
                "timestamp": row.get("timestamp", ""),
                "public_trace": trace,
            }
            self._set_text(self.trace_text, json.dumps(trace_payload, indent=2, ensure_ascii=False, default=str))
        if len(selected) >= 2:
            left, right = (self.current_rows[int(selected[0])], self.current_rows[int(selected[1])])
            self._set_text(self.compare_left, f"{left.get('model')} · {left.get('prompt')}\n\n{left.get('output', '')}")
            self._set_text(self.compare_right, f"{right.get('model')} · {right.get('prompt')}\n\n{right.get('output', '')}")
        elif selected:
            row = self.current_rows[int(selected[0])]
            self._log(f"Trace selected: {row.get('run_id')} · {row.get('latency_ms')} ms · {row.get('tokens')} tokens")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _draw_charts(self) -> None:
        if not hasattr(self, "chart_canvas"):
            return
        canvas = self.chart_canvas
        canvas.delete("all")
        width = max(500, canvas.winfo_width())
        height = max(300, canvas.winfo_height())
        canvas.create_text(24, 24, text="RESULTS OVERVIEW", anchor="w", fill=self.COLORS["muted"], font=("Segoe UI", 10, "bold"))
        rows = self.current_rows
        if not rows:
            canvas.create_text(width / 2, height / 2, text="Run the matrix to populate local statistics", fill=self.COLORS["muted"], font=("Segoe UI", 12))
            return
        # left: score bars (first 16 runs), right: latency trend
        left_x, base_y = 45, height - 52
        chart_w = width * 0.53
        chart_h = height - 105
        canvas.create_text(left_x, 58, text="Score by run", anchor="w", fill=self.COLORS["text"], font=("Segoe UI", 10, "bold"))
        canvas.create_line(left_x, base_y, left_x + chart_w, base_y, fill=self.COLORS["border"])
        vals = [float(r.get("score", 0) or 0) for r in rows[:16]]
        bar_w = max(7, (chart_w - 16) / max(1, len(vals)) - 3)
        for i, val in enumerate(vals):
            x = left_x + 8 + i * (bar_w + 3)
            y = base_y - chart_h * min(1, max(0, val))
            canvas.create_rectangle(x, y, x + bar_w, base_y, fill=self.COLORS["accent"], outline="")
            canvas.create_text(x + bar_w / 2, base_y + 12, text=str(i + 1), fill=self.COLORS["muted"], font=("Segoe UI", 8))
        rx = left_x + chart_w + 60
        canvas.create_text(rx, 58, text="Latency trend (ms)", anchor="w", fill=self.COLORS["text"], font=("Segoe UI", 10, "bold"))
        lats = [float(r.get("latency_ms", 0) or 0) for r in rows[:24]]
        max_lat = max(lats) if lats else 1
        rw = width - rx - 35
        canvas.create_line(rx, base_y, rx + rw, base_y, fill=self.COLORS["border"])
        points: list[float] = []
        for i, val in enumerate(lats):
            x = rx + (i / max(1, len(lats) - 1)) * rw
            y = base_y - (val / max_lat) * chart_h
            points.extend((x, y))
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=self.COLORS["accent2"], outline="")
        if len(points) > 3:
            canvas.create_line(*points, fill=self.COLORS["accent2"], width=2, smooth=True)

    def _update_report(self) -> None:
        if not self.current:
            return
        rows = self.current_rows
        scores = [float(r.get("score", 0) or 0) for r in rows]
        mean = statistics.mean(scores) if scores else 0
        report = [
            f"# {self.current.get('name', 'LLM Lab Experiment')}",
            "",
            f"**Experiment ID:** `{self.current.get('id', '')}`  ",
            f"**Provider:** `MockProvider` (offline; no paid API calls)  ",
            f"**Generated:** {_now()}",
            "",
            "## Question",
            self.current.get("research_question", ""),
            "",
            "## Hypothesis",
            self.current.get("hypothesis", ""),
            "",
            "## Method",
            f"Prompt version `{self.current.get('prompt_version', 'v1')}`, dataset `{self.current.get('dataset', '')}`, matrix size {len(rows)}.",
            "",
            "## Result",
            f"Mean score: **{mean:.3f}**; runs: **{len(rows)}**; success rate: **{sum(1 for r in rows if r.get('status') == 'success') / len(rows) * 100 if rows else 0:.1f}%**.",
            "",
            "## Interpretation",
            "Use the side-by-side output and trace panel to inspect individual runs before drawing conclusions.",
            "",
            "## Limitations",
            "This demo uses deterministic mock responses. Replace the provider only when an explicit, configured local or remote provider is desired.",
        ]
        self._set_text(self.report_text, "\n".join(report))

    def _export_report(self) -> None:
        if not self.current:
            messagebox.showinfo("Export report", "Select an experiment first.")
            return
        default = _slug(str(self.current.get("name", "experiment"))) + ".md"
        path = filedialog.asksaveasfilename(title="Export Markdown report", initialdir=str(self.project_dir), initialfile=default, defaultextension=".md", filetypes=[("Markdown", "*.md"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.adapter.export_report(self.current, self.current_rows, path)
            self._log(f"Exported report: {path}")
            self.status_var.set(f"Report exported · {path}")
        except Exception as exc:
            messagebox.showerror("Export report", str(exc))

    def _import_dataset(self) -> None:
        path = filedialog.askopenfilename(title="Import dataset", filetypes=[("JSON/JSONL/CSV/TXT", "*.json *.jsonl *.csv *.txt"), ("All files", "*.*")])
        if not path:
            return
        file_path = Path(path)
        try:
            if file_path.suffix.lower() == ".csv":
                with file_path.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            elif file_path.suffix.lower() in (".json", ".jsonl"):
                raw = file_path.read_text(encoding="utf-8")
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if file_path.suffix.lower() == ".jsonl" else json.loads(raw)
                if isinstance(rows, Mapping):
                    rows = rows.get("data", [rows])
            else:
                rows = [{"input": line} for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
            self.adapter._datasets.append({"id": _slug(file_path.stem), "name": file_path.name, "format": file_path.suffix.lstrip("."), "rows": len(rows), "split": "unspecified", "hash": f"sha256:{digest}"})
            self._refresh_sidebar()
            self._log(f"Imported dataset {file_path.name} ({len(rows)} rows)")
        except Exception as exc:
            messagebox.showerror("Import dataset", str(exc))

    def _log(self, message: str) -> None:
        if hasattr(self, "log_text"):
            self.log_text.insert("end", f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {message}\n")
            self.log_text.see("end")


def run(
    adapter: GuiAdapter | None = None,
    project_dir: str | Path | None = None,
    *,
    store: Any = None,
    runner: Any = None,
) -> None:
    """Launch the GUI; kept as a small function for console-script entrypoints."""
    if adapter is None and (store is not None or runner is not None):
        adapter = GuiAdapter(store=store, runner=runner, project_dir=project_dir)
    app = LLMLabApp(adapter=adapter, project_dir=project_dir)
    app.mainloop()


def capture_screenshot(widget: tk.Misc, destination: str | Path) -> Path:
    """Capture a widget for QA/screenshots without adding a GUI dependency.

    On Windows/macOS/Linux desktops with Pillow installed this writes PNG/JPEG
    pixels.  In minimal environments it falls back to Tk's vector PostScript
    export (use a ``.ps`` or ``.eps`` destination).  The helper is intentionally
    explicit and never runs on import, which keeps headless CI safe.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    widget.update_idletasks()
    suffix = destination.suffix.lower()
    if suffix in {".ps", ".eps"} and hasattr(widget, "postscript"):
        widget.postscript(file=str(destination), colormode="color")
        return destination
    try:
        from PIL import ImageGrab  # type: ignore

        left = widget.winfo_rootx()
        top = widget.winfo_rooty()
        right = left + widget.winfo_width()
        bottom = top + widget.winfo_height()
        image = ImageGrab.grab(bbox=(left, top, right, bottom))
        image.save(destination)
        return destination
    except Exception:
        # Canvas widgets support PostScript even when Pillow/ImageGrab do not.
        canvas = getattr(widget, "chart_canvas", None)
        if canvas is not None:
            fallback = destination.with_suffix(".ps")
            canvas.postscript(file=str(fallback), colormode="color")
            return fallback
        raise RuntimeError("PNG capture requires Pillow/ImageGrab; use a .ps destination for Tk vector capture")


if __name__ == "__main__":  # pragma: no cover - manual launch
    run()
