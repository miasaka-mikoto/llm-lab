# LLM Lab / 大语言模型实验室

LLM Lab is an offline-first research workbench for learning, comparing and reproducing LLM-system experiments. It is deliberately useful without an API key: the included `MockProvider`, fake embeddings, sample dataset and demo run exercise the same provider, evaluator, statistics, trace and report paths used by future local or hosted providers.

> **Safety and reproducibility.** No paid model API is called by default. API keys are read only from environment variables by optional adapters and are never stored in experiment data. Agent traces contain public action summaries and tool results, never hidden chain-of-thought.

## Start

Windows: double-click `run_llm_lab.bat` (or run `python run.py`). Other platforms: `python -m llmlab` from this directory. The GUI starts in dark mode and creates `llmlab.sqlite3` locally. It can be used entirely offline.

The command line demo is useful on a server or in CI:

```bash
python -m llmlab demo --db demo.sqlite3 --report reports/demo.md
python -m llmlab headless --dataset sample_data/prompt_robustness.jsonl
python -m unittest discover -s tests -v
```

## What is included

- Experiment records with question, hypothesis, dataset, prompt versions, provider/parameters, evaluator, metrics, runs, results, notes and artifacts.
- A common provider protocol (`generate`, `stream`, `embed`, `count_tokens`) and deterministic `MockProvider`; adapters can be added later for Ollama, vLLM or HTTP without changing experiment code.
- Prompt Playground with variables, temperature/top-p/max-tokens/seed and structured-output settings; version comparison is persisted.
- Dataset readers for JSON, JSONL, CSV and TXT with input/expected/context/metadata mapping.
- Batch matrices with pause/resume/retry/cancel and deterministic sample seeds.
- Evaluators: exact match, contains, regex, JSON validity, schema compliance, length, latency, token count and cost estimate; custom Python evaluators are supported through the documented hook.
- RAG, memory, agent and structured-output lab helpers; traces and reproducibility manifests are stored in SQLite.
- Model Benchmark dashboard data for Reasoning, Coding, Extraction, Classification, RAG, Tool Use and Long Context; it only displays real experiment rows or explicitly labelled `sample_mock` rows and never ships a ranking.
- Statistics (mean, median, standard deviation, percentiles, success/error rates and confidence intervals) plus lightweight Canvas charts.
- Markdown/HTML research reports and paper-reproduction template.

## Demo

The first-run **Prompt Robustness Test** uses three mock models, four prompt variants and twenty sample records. It creates 240 runs, computes metrics, shows charts and exports a report. Select **Run Demo** in the GUI or use `python -m llmlab demo`.

## Layout

```text
LLMLab/
  llmlab/              # engine, providers, storage, labs and Tk GUI
  sample_data/         # offline datasets
  templates/           # experiment and paper-reproduction templates
  tests/               # stdlib unittest suite
  run.py               # launcher
  run_gui.py           # persistent SQLite GUI launcher
  run_llm_lab.bat      # Windows launcher
  build_windows.bat    # optional PyInstaller build
  reports/             # generated demo Markdown/HTML/SVG dashboard
```

## Optional providers

The development path intentionally uses `MockProvider` only. Future adapters should implement the `LLMProvider` protocol and obtain credentials from process environment (for example `OPENAI_API_KEY`); do not put keys in source, YAML, SQLite, reports or screenshots.

## License / scope

This is a local research prototype for files and datasets you are authorised to inspect. It does not implement DRM bypass, account authentication, payment, anti-cheat circumvention or other security bypasses.
