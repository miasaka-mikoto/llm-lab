"""Small, inspectable Agent Lab runtime with public trace summaries only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .models import TraceEvent


@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, function: Callable[..., Any], description: str = "") -> None:
        self._tools[name] = Tool(name, description, function)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].function(**kwargs)


@dataclass
class AgentPlan:
    action: str
    tool: str = ""
    rationale: str = ""
    scores: Dict[str, float] = field(default_factory=dict)


class RuleBasedPlanner:
    """Rule planner that records rationale summaries, never hidden thoughts."""

    def plan(self, observation: Mapping[str, Any], tools: Iterable[str]) -> AgentPlan:
        text = str(observation.get("input", observation.get("goal", ""))).casefold()
        scores = {"answer": 0.35, "search": 0.20, "calculate": 0.10, "wait": 0.05}
        if any(word in text for word in ("search", "find", "lookup", "查找")):
            scores["search"] = 0.90
        if any(word in text for word in ("calculate", "sum", "math", "计算")) or all(ch in "0123456789+-*/(). %" for ch in text.strip()) and any(ch.isdigit() for ch in text):
            scores["calculate"] = 0.90
        available = set(tools)
        action = max(scores, key=scores.get)
        tool = action if action in available else ""
        return AgentPlan(action, tool, f"Rule match selected '{action}' from observable task keywords.", scores)


class AgentEnvironment:
    def observe(self, input_value: Any) -> Dict[str, Any]:
        return {"input": input_value}


@dataclass
class AgentRun:
    final: Any
    trace: List[TraceEvent]
    plan: AgentPlan
    error: str = ""


class Agent:
    def __init__(self, *, planner: Optional[RuleBasedPlanner] = None, tools: Optional[ToolRegistry] = None, environment: Optional[AgentEnvironment] = None):
        self.planner = planner or RuleBasedPlanner()
        self.tools = tools or ToolRegistry()
        self.environment = environment or AgentEnvironment()
        self._register_defaults()

    def _register_defaults(self) -> None:
        if "calculate" not in self.tools.names():
            self.tools.register("calculate", lambda expression="": _safe_calculate(expression), "Evaluate a basic arithmetic expression")
        if "search" not in self.tools.names():
            self.tools.register("search", lambda query="": f"No external search in offline mode; indexed query: {query}", "Search local/offline index")

    def run(self, input_value: Any) -> AgentRun:
        observation = self.environment.observe(input_value)
        plan = self.planner.plan(observation, self.tools.names())
        trace = [TraceEvent(0, "observation", observation=str(observation), thought_summary="Observation captured from the environment.")]
        try:
            if plan.tool:
                if plan.tool == "calculate":
                    # Planner keywords are routing hints, not part of the
                    # arithmetic expression.  Strip a natural-language
                    # prefix so inputs such as ``calculate: 2 + 3`` remain
                    # executable while the safe AST evaluator still rejects
                    # arbitrary code.
                    import re
                    expression = re.sub(r"^\s*(?:calculate|calc|sum|math)\s*[:\-]?\s*", "", str(input_value), flags=re.I)
                    kwargs = {"expression": expression}
                else:
                    kwargs = {"query": input_value}
                tool_result = self.tools.call(plan.tool, **kwargs)
                trace.append(TraceEvent(1, "action", action=plan.action, tool=plan.tool, result=tool_result, thought_summary=plan.rationale))
                final = tool_result
            else:
                final = str(input_value)
                trace.append(TraceEvent(1, "final", result=final, thought_summary="No tool was required by the rule planner."))
            trace.append(TraceEvent(len(trace), "final", result=final, thought_summary="Run completed."))
            return AgentRun(final, trace, plan)
        except Exception as exc:
            trace.append(TraceEvent(len(trace), "error", result=str(exc), thought_summary="Tool execution failed."))
            return AgentRun("", trace, plan, str(exc))


def _safe_calculate(expression: str) -> Any:
    # Deliberately narrow arithmetic parser; no eval and no arbitrary code.
    import ast
    import operator
    tree = ast.parse(str(expression), mode="eval")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops: return ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)): return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        raise ValueError("Only basic arithmetic is allowed")
    return visit(tree)
