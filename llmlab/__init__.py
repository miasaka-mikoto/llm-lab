"""LLM Lab offline research engine.

The package is deliberately usable without API keys, network access or paid
model calls.  Import ``default_registry`` or ``MockProvider`` for local runs.
"""

from .models import Dataset, DatasetRecord, Experiment, PromptVersion, ReproducibilityManifest, RunResult, TraceEvent
from .providers import AnthropicProvider, GenerationResult, GoogleProvider, HTTPProvider, LLMProvider, MockProvider, OllamaProvider, OpenAIProvider, ProviderRegistry, UnavailableProvider, VLLMProvider, default_registry
from .datasets import DatasetViewer, from_records, load_dataset, sample_dataset
from .prompts import PromptRegistry, PromptTemplate, make_prompt
from .evaluators import EvaluationResult, EvaluatorSuite, LLMJudgeEvaluator, exact_match, contains, regex_match, json_validity, load_custom_evaluator, schema_compliance
from .statistics import chart_data, confidence_interval, group_summary, percentile, summarize, summarize_runs
from .experiments import ExperimentMatrix, ExperimentRunner, RunnerState, side_by_side
from .rag import Chunk, Document, RAGPipeline, RAGResult, Reranker, Retrieval, VectorStore, chunk_document
from .memory import EpisodicMemory, MemoryItem, MemoryStrategy, NoMemory, SemanticMemory, SlidingWindow, SummaryMemory, VectorMemory, make_memory
from .agent import Agent, AgentEnvironment, AgentPlan, AgentRun, RuleBasedPlanner, Tool, ToolRegistry
from .structured import StructuredDiagnostics, classify, extract_json, function_style, parse_json_output
from .storage import SQLiteStore
from .report import build_manifest, dashboard_svg, export_report, html_report, markdown_report
from .labs import AgentLab, MemoryLab, PromptPlayground, RAGLab, StructuredOutputLab
from .benchmark import BenchmarkRecord, BenchmarkSuite, CATEGORIES

__all__ = [
    "Agent", "AgentEnvironment", "AgentLab", "AgentPlan", "AgentRun", "AnthropicProvider", "BenchmarkRecord", "BenchmarkSuite", "CATEGORIES", "Chunk", "Dataset", "DatasetRecord", "DatasetViewer", "Document", "EpisodicMemory", "EvaluationResult", "EvaluatorSuite", "Experiment", "ExperimentMatrix", "ExperimentRunner", "GenerationResult", "GoogleProvider", "HTTPProvider", "LLMJudgeEvaluator", "LLMProvider", "MemoryItem", "MemoryLab", "MemoryStrategy", "MockProvider", "NoMemory", "OllamaProvider", "OpenAIProvider", "PromptPlayground", "PromptRegistry", "PromptTemplate", "PromptVersion", "ProviderRegistry", "RAGLab", "RAGPipeline", "RAGResult", "ReproducibilityManifest", "Reranker", "Retrieval", "RuleBasedPlanner", "RunResult", "RunnerState", "SQLiteStore", "SemanticMemory", "side_by_side", "SlidingWindow", "StructuredDiagnostics", "StructuredOutputLab", "SummaryMemory", "Tool", "ToolRegistry", "TraceEvent", "UnavailableProvider", "VLLMProvider", "VectorMemory", "VectorStore", "build_manifest", "chart_data", "chunk_document", "classify", "confidence_interval", "contains", "dashboard_svg", "default_registry", "exact_match", "export_report", "extract_json", "from_records", "function_style", "group_summary", "html_report", "json_validity", "load_custom_evaluator", "load_dataset", "make_memory", "make_prompt", "markdown_report", "parse_json_output", "percentile", "regex_match", "sample_dataset", "schema_compliance", "summarize", "summarize_runs"
]
