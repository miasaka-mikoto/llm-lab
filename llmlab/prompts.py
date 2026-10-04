"""Prompt templates and lightweight version management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .models import PromptVersion


@dataclass
class PromptTemplate:
    name: str
    versions: Dict[str, PromptVersion] = field(default_factory=dict)
    active_version: str = "v1"

    def add_version(
        self,
        version: str,
        *,
        system: str = "",
        user: str = "",
        template: str = "",
        variables: Optional[Mapping[str, Any]] = None,
        notes: str = "",
    ) -> PromptVersion:
        prompt = PromptVersion(version, system, user, template, dict(variables or {}), notes=notes)
        self.versions[version] = prompt
        if not self.active_version:
            self.active_version = version
        return prompt

    def get(self, version: Optional[str] = None) -> PromptVersion:
        version = version or self.active_version
        if version not in self.versions:
            raise KeyError(f"Unknown prompt version: {version}")
        return self.versions[version]

    def render(self, values: Optional[Mapping[str, Any]] = None, version: Optional[str] = None) -> tuple[str, str]:
        return self.get(version).render(dict(values or {}))

    def compare_versions(self, versions: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
        selected = list(versions or self.versions)
        return [{"version": v, "system": self.get(v).system, "user": self.get(v).template or self.get(v).user} for v in selected]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "active_version": self.active_version, "versions": {k: vars(v) for k, v in self.versions.items()}}


class PromptRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> PromptTemplate:
        self._templates[template.name] = template
        return template

    def get(self, name: str) -> PromptTemplate:
        return self._templates[name]

    def names(self) -> List[str]:
        return sorted(self._templates)


def make_prompt(name: str, system: str, user: str, *, version: str = "v1", variables: Optional[Mapping[str, Any]] = None) -> PromptTemplate:
    template = PromptTemplate(name=name, active_version=version)
    template.add_version(version, system=system, user=user, template=user, variables=variables)
    return template

