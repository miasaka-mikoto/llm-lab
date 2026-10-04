# LLM Lab test report

Last local run: 2026-10-04 (offline, Python standard library).

```text
python -m unittest discover -s tests -v
Ran 21 tests ... OK
```

Coverage includes provider determinism and streaming, dataset formats and
hashes, prompt rendering/versioning, matrix execution, retries and traces,
evaluators, statistics/confidence intervals, RAG retrieval, all bounded memory
strategies, structured diagnostics, SQLite save/load, report export and SVG
dashboard generation. `python -m llmlab self-check` additionally validates the
provider, structured-output and RAG paths. The demo was run as a 240-run
3-model × 4-prompt × 20-sample matrix and completed successfully.
