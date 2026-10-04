"""Provider abstraction and deterministic offline providers.

No provider in this module makes a network request implicitly.  ``MockProvider``
is the default and is sufficient for every demo and test in local mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import random
import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Stable interface implemented by local and future model providers."""

    name = "provider"

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        structured_output: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        raise NotImplementedError

    def stream(self, system: str, user: str, **kwargs: Any) -> Iterator[str]:
        result = self.generate(system, user, **kwargs)
        # Deliberately small chunks make the API useful for a UI without
        # requiring a real streaming backend.
        words = result.text.split(" ")
        for index, word in enumerate(words):
            yield word + (" " if index < len(words) - 1 else "")

    @abstractmethod
    def embed(self, texts: Iterable[str], *, dimensions: int = 32, **kwargs: Any) -> List[List[float]]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        # A predictable approximation that works offline and for non-ASCII
        # text.  It is intentionally called a count estimate in the UI.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class MockProvider(LLMProvider):
    """Deterministic model simulator with a few useful model personalities.

    ``model`` can be ``mock-balanced``, ``mock-concise`` or ``mock-noisy``.
    The same prompt/seed produces the same output, which makes experiments
    reproducible and avoids spending real API quota during development.
    """

    name = "mock"

    def __init__(self, model: str = "mock-balanced", *, latency_ms: float = 2.0) -> None:
        self.model = model
        self.latency_ms = float(latency_ms)

    def _rng(self, system: str, user: str, seed: Optional[int]) -> random.Random:
        material = f"{self.model}|{seed if seed is not None else 0}|{system}|{user}".encode()
        digest = hashlib.sha256(material).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _structured(self, user: str, schema: Optional[Dict[str, Any]], rng: random.Random) -> str:
        if not schema:
            return json.dumps({"answer": self._normalise(user)}, ensure_ascii=False)
        properties = schema.get("properties", {}) if schema.get("type") == "object" else {}
        required = schema.get("required", list(properties))
        result: Dict[str, Any] = {}
        for key in required:
            spec = properties.get(key, {})
            typ = spec.get("type", "string")
            if typ == "number":
                result[key] = round(rng.random(), 3)
            elif typ == "integer":
                result[key] = int(rng.random() * 10)
            elif typ == "boolean":
                result[key] = True
            elif typ == "array":
                result[key] = []
            else:
                result[key] = self._normalise(user)[:120]
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        structured_output: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        started = time.perf_counter()
        rng = self._rng(system, user, seed)
        if structured_output is not None:
            text = self._structured(user, structured_output, rng)
        else:
            clean = self._normalise(user)
            # A tiny deterministic task heuristic gives demo experiments
            # meaningful variation without pretending to be a real LLM.
            lower = clean.lower()
            if any(word in lower for word in ("classify", "classification", "sentiment")):
                if any(word in lower for word in ("good", "great", "excellent", "happy", "love", "like", "喜欢", "好")):
                    answer = "positive"
                elif any(word in lower for word in ("bad", "terrible", "sad", "讨厌", "差")):
                    answer = "negative"
                else:
                    answer = "neutral"
                text = answer
            elif any(word in lower for word in ("extract", "json")):
                text = json.dumps({"text": clean[:120]}, ensure_ascii=False)
            else:
                text = f"Mock answer: {clean[: max(1, min(max_tokens * 4, 180))]}"
            if self.model.endswith("concise"):
                text = text.split(": ", 1)[-1][:80]
            elif self.model.endswith("noisy") and temperature > 0:
                suffixes = [" (simulated)", " [mock]", " — variation"]
                text += suffixes[rng.randrange(len(suffixes))]
        max_tokens = max(1, int(max_tokens))
        words = text.split()
        if len(words) > max_tokens:
            text = " ".join(words[:max_tokens])
            finish_reason = "length"
        else:
            finish_reason = "stop"
        elapsed = (time.perf_counter() - started) * 1000.0 + self.latency_ms
        return GenerationResult(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=self.count_tokens(system + " " + user),
            output_tokens=self.count_tokens(text),
            latency_ms=round(elapsed, 3),
            finish_reason=finish_reason,
            raw={"offline": True, "temperature": temperature, "top_p": top_p, "seed": seed},
        )

    def embed(self, texts: Iterable[str], *, dimensions: int = 32, **kwargs: Any) -> List[List[float]]:
        vectors: List[List[float]] = []
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dimensions)]
            norm = sum(v * v for v in values) ** 0.5 or 1.0
            vectors.append([round(v / norm, 8) for v in values])
        return vectors


class UnavailableProvider(LLMProvider):
    """Explicit placeholder for future network providers.

    It prevents accidental network/API calls while making configuration errors
    clear to the user.
    """

    def __init__(self, provider_name: str) -> None:
        self.name = provider_name

    def generate(self, system: str, user: str, **kwargs: Any) -> GenerationResult:
        raise RuntimeError(
            f"Provider '{self.name}' is not enabled in offline development mode. "
            "Use MockProvider or explicitly configure a future adapter."
        )

    def embed(self, texts: Iterable[str], **kwargs: Any) -> List[List[float]]:
        raise RuntimeError(f"Provider '{self.name}' is unavailable in offline mode")


# Named placeholders make the future adapter boundary explicit while keeping
# accidental network calls impossible during development.
class OpenAIProvider(UnavailableProvider):
    def __init__(self, **kwargs: Any): super().__init__("openai")


class AnthropicProvider(UnavailableProvider):
    def __init__(self, **kwargs: Any): super().__init__("anthropic")


class GoogleProvider(UnavailableProvider):
    def __init__(self, **kwargs: Any): super().__init__("google")


class OllamaProvider(UnavailableProvider):
    def __init__(self, **kwargs: Any): super().__init__("ollama")


class VLLMProvider(UnavailableProvider):
    def __init__(self, **kwargs: Any): super().__init__("vllm")


class HTTPProvider(UnavailableProvider):
    def __init__(self, **kwargs: Any): super().__init__("http")


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider, alias: Optional[str] = None) -> None:
        self._providers[alias or provider.name] = provider

    def get(self, name: str = "mock") -> LLMProvider:
        if name not in self._providers:
            raise KeyError(f"Unknown provider: {name}")
        return self._providers[name]

    def names(self) -> List[str]:
        return sorted(self._providers)


def default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    for model in ("mock-balanced", "mock-concise", "mock-noisy"):
        registry.register(MockProvider(model), alias=model)
    registry.register(registry.get("mock-balanced"), alias="mock")
    # Future providers are visible in configuration, but cannot accidentally
    # make network requests.
    for name in ("openai", "anthropic", "google", "ollama", "vllm", "http"):
        registry.register(UnavailableProvider(name))
    return registry
