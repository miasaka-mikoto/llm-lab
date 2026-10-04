# LLM Lab architecture

The project is a standard-library Python application with a strict offline
boundary. `MockProvider` is the only provider registered for development; the
named cloud/local adapters are explicit `UnavailableProvider` placeholders and
cannot make a network request by accident.

```text
Dataset -> PromptTemplate -> ExperimentMatrix -> ExperimentRunner
                                              |-> LLMProvider (MockProvider)
                                              |-> EvaluatorSuite
                                              |-> RunResult + public TraceEvent
                                              |-> Statistics / SVG dashboard
                                              |-> SQLiteStore / Markdown + HTML report
```

Specialised labs are independent, reusable modules: `rag.py` provides chunking,
fake embeddings, vector retrieval and reranking; `memory.py` provides bounded
memory strategies; `agent.py` provides a rule planner, safe tools and public
trace summaries; `structured.py` diagnoses JSON/schema output. `labs.py` adds
small facades useful in notebooks.

The Tkinter GUI is a presentation layer. It can use `SQLiteStore`, but retains
an in-memory sample adapter so the interface opens even on a clean offline
machine. The same engine is available headlessly for CI and long matrices.

All experiment configuration, dataset hashes, prompt versions, provider names,
parameters, environment and dependency information can be saved as a
`ReproducibilityManifest`. No hidden chain-of-thought is persisted or shown;
agent/provider traces contain only observable inputs, outputs, action names and
short rule-generated summaries.
