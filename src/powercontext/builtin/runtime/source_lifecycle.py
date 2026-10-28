# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persisted, replayable evidence eligibility for Memory and Topic Memory."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.source_lifecycle import SourceLifecycleStore
from powercontext.builtin.sources.journal import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """An invalid graph binding, event, or generation request."""


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: StrictInt = Field(gt=0)
    sequence: StrictInt = Field(gt=0)
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("event text must not be blank")  # noqa: TRY003
        return value


class _Node(BaseModel):
    kind: Literal["source", "memory", "topic"]
    source: SourceRef | None = None
    citation: MemoryCitation | None = None
    artifact: ArtifactRef | None = None
    revision: int | None = None
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    available: bool = True
    synced: bool | None = None
    event: SourceLifecycleEvent | None = None


class _Graph(BaseModel):
    nodes: dict[str, _Node] = Field(default_factory=dict)


class SourceLifecycleApplication:
    """Select a Scope's durable evidence graph."""

    def __init__(self, runtime: BuiltinRuntime, store: SourceLifecycleStore | None) -> None:
        self.runtime = runtime
        self.store = store
        self.locks: dict[str, asyncio.Lock] = {}

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self, validate_scope_id(scope_id))


class ScopedSourceLifecycle:
    """Apply source events through native Memory transitions and read filters."""

    def __init__(self, owner: SourceLifecycleApplication, scope_id: str) -> None:
        self.owner = owner
        self.scope_id = scope_id

    @property
    def _store(self) -> SourceLifecycleStore:
        if self.owner.store is None:
            raise SourceLifecycleError("Lifecycle persistence is unavailable")  # noqa: TRY003
        return self.owner.store

    @property
    def _lock(self) -> asyncio.Lock:
        return self.owner.locks.setdefault(self.scope_id, asyncio.Lock())

    async def _load(self) -> _Graph:
        payload = await self._store.load(self.scope_id)
        return _Graph() if payload is None else _Graph.model_validate_json(payload)

    async def _save(self, graph: _Graph) -> None:
        topics: dict[str, bool] = {}
        for node in graph.nodes.values():
            if node.artifact is not None:
                artifact_id = node.artifact.artifact_id
                topics[artifact_id] = topics.get(artifact_id, False) or node.available
        await self._store.save(self.scope_id, graph.model_dump_json().encode(), topics)

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        if type(revision) is not int or revision < 1 or not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("Source registration requires a reference and positive revision")  # noqa: TRY003
        _require_node_id(node_id)
        async with self._lock:
            graph = await self._load()
            node = _Node(kind="source", source=source_ref, revision=revision, synced=True)
            if self._existing(graph, node_id, node):
                return
            try:
                await self._store.validate_source(self.scope_id, source_ref)
            except (ValueError, LookupError) as error:
                raise SourceLifecycleError("Source reference is not readable in this Scope") from error  # noqa: TRY003
            graph.nodes[node_id] = node
            await self._save(graph)

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("Memory registration requires an exact citation")  # noqa: TRY003
        await self._register_derived(node_id, _Node(kind="memory", citation=citation), parents, policy)

    async def register_topic(
        self, node_id: str, artifact_ref: ArtifactRef, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("Topic registration requires a Topic Memory reference")  # noqa: TRY003
        await self._register_derived(node_id, _Node(kind="topic", artifact=artifact_ref), parents, policy)

    def _existing(self, graph: _Graph, node_id: str, node: _Node) -> bool:
        previous = graph.nodes.get(node_id)
        if previous is None:
            return False
        binding = {"kind", "source", "citation", "artifact", "revision", "parents", "policy"}
        if previous.model_dump(include=binding) != node.model_dump(include=binding):
            raise SourceLifecycleError("Node is already registered with a different binding")  # noqa: TRY003
        return True

    async def _register_derived(self, node_id: str, node: _Node, parents: tuple[str, ...], policy: str) -> None:
        _require_node_id(node_id)
        if (
            not isinstance(policy, str)
            or policy not in {"all", "any"}
            or not isinstance(parents, (tuple, list))
            or not parents
        ):
            raise SourceLifecycleError("Derived nodes require parents and an all/any policy")  # noqa: TRY003
        if any(not isinstance(parent, str) or not parent.strip() for parent in parents):
            raise SourceLifecycleError("Parent identities must be non-blank strings")  # noqa: TRY003
        if node_id in parents or len(set(parents)) != len(parents):
            raise SourceLifecycleError("Parent identities must be unique and cannot refer to self")  # noqa: TRY003
        node = node.model_copy(update={"parents": tuple(sorted(parents)), "policy": policy})
        async with self._lock:
            graph = await self._load()
            if self._existing(graph, node_id, node):
                await self._synchronize(graph, "Registration replay")
                return
            if any(parent not in graph.nodes for parent in parents):
                raise SourceLifecycleError("Parents must be registered first")  # noqa: TRY003
            await self._validate_reference(node)
            graph.nodes[node_id] = node
            _recalculate(graph)
            # Save accepted intent first. An interrupted native write can then
            # be resumed by the exact same registration or source event.
            await self._save(graph)
            await self._synchronize(graph, "Dependency registration")

    async def _validate_reference(self, node: _Node) -> None:
        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        try:
            if node.citation is not None:
                await self.owner.runtime.memory.for_scope(self.scope_id).get(
                    GetMemoryEntryRequest(citation=node.citation)
                )
            elif node.artifact is not None:
                await self.owner.runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    node.artifact.family, node.artifact.artifact_id, node.artifact.revision
                )
        except (ValueError, LookupError) as error:
            raise SourceLifecycleError("Reference is not readable in this Scope") from error  # noqa: TRY003

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a validated lifecycle event")  # noqa: TRY003
        async with self._lock:
            graph = await self._load()
            node = graph.nodes.get(event.node_id)
            if node is None or node.kind != "source" or node.revision != event.revision:
                raise SourceLifecycleError("Unknown Source or lifecycle revision mismatch")  # noqa: TRY003
            if node.event is not None and (
                event.sequence < node.event.sequence or (event.sequence == node.event.sequence and event != node.event)
            ):
                raise SourceLifecycleError("Event is stale or conflicts with the accepted sequence")  # noqa: TRY003
            node.event = event
            node.available = event.available
            _recalculate(graph)
            await self._save(graph)
            await self._synchronize(graph, event.reason)
            return sorted(node_id for node_id, value in graph.nodes.items() if value.available)

    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        _require_node_id(node_id)
        if not isinstance(purpose, str) or purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("Unknown reference purpose")  # noqa: TRY003
        async with self._lock:
            node = (await self._load()).nodes.get(node_id)
            if node is None or (purpose == "generation" and not node.available):
                raise SourceLifecycleError("Node is unknown or unavailable for generation")  # noqa: TRY003
            reference = node.source or node.citation or node.artifact
            if reference is None:
                raise SourceLifecycleError("Stored node has no reference")  # noqa: TRY003
            return reference

    async def _synchronize(self, graph: _Graph, reason: str) -> None:
        # Aliases of one immutable entry can independently support its whole
        # body. Native eligibility is the union of their available supports.
        groups: dict[tuple[str, str], list[_Node]] = {}
        for node in graph.nodes.values():
            if node.citation is not None:
                key = (node.citation.memory_ref.artifact_id, node.citation.entry_id)
                groups.setdefault(key, []).append(node)
        for nodes in groups.values():
            available = any(node.available for node in nodes)
            if all(node.synced == available for node in nodes):
                continue
            await self._set_memory_state(nodes[0], available, reason)
            for node in nodes:
                node.synced = available
            await self._save(graph)

    async def _set_memory_state(self, node: _Node, available: bool, reason: str) -> None:
        citation = node.citation
        if citation is None:
            raise SourceLifecycleError("Memory node has no citation")  # noqa: TRY003
        runtime = self.owner.runtime
        async with runtime._context(self.scope_id) as context, runtime._locked(self.scope_id):
            service = context.artifacts.memory
            memory = await service.head(context.artifacts.memory_artifact_id)
            entries = await service.entries(memory)
            entry = next((entry for entry in entries if entry.entry_id == citation.entry_id), None)
            if entry is None or entry.entry_version_id != citation.entry_version_id:
                raise SourceLifecycleError("Bound Memory body changed or disappeared")  # noqa: TRY003
            manifest_entry = next(
                item for item in memory.content.manifest.entries if item.entry_id == citation.entry_id
            )
            if (manifest_entry.state == "active") == available:
                return
            if available:
                await service.reactivate(memory, entries=(entry,), reason=reason)
            else:
                await service.forget(memory, entries=(entry,), reason=reason)


def _require_node_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("Node identity must be a non-blank string")  # noqa: TRY003


def _recalculate(graph: _Graph) -> None:
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(node_id: str) -> bool:
        node = graph.nodes[node_id]
        if node_id in visited:
            return node.available
        if node_id in visiting:
            raise SourceLifecycleError("Dependency cycle")  # noqa: TRY003
        visiting.add(node_id)
        if node.kind != "source":
            support = [visit(parent) for parent in node.parents]
            node.available = all(support) if node.policy == "all" else any(support)
        visiting.remove(node_id)
        visited.add(node_id)
        return node.available

    for node_id in graph.nodes:
        visit(node_id)


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent"]
