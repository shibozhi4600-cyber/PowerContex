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

"""Lifecycle support follows an exact body version, including after a restart."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.records import ArtifactWrite
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    ContextAssembly,
    ContextAssemblySection,
    GetMemoryEntryRequest,
    PrepareContextRequest,
    RememberMemoryRequest,
    ReviseMemoryEntryRequest,
    SearchMemoryRequest,
    SearchTopicMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


def _event(root: str, sequence: int, available: bool) -> SourceLifecycleEvent:
    return SourceLifecycleEvent(
        node_id=root, revision=1, sequence=sequence, available=available, reason="Version support"
    )


def test_topic_history_cannot_keep_an_unsupported_current_revision_searchable(tmp_path: Path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'topic-revisions.db'}"))

    async def scenario() -> None:
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(
                    title="Exact Topic versions", summary="Independent supports", idempotency_key="topic-revisions"
                )
            )
            sources = runtime.sources.for_scope(scope.scope_id)
            records = runtime.records.for_scope(scope.scope_id)
            old = await records.create_artifact(
                "topic-memory",
                ArtifactWrite(
                    content={"title": "oldtopicneedle", "summary": "oldtopicneedle supported", "detail": "Old body"}
                ),
            )
            current = await records.replace_artifact(
                "topic-memory",
                old.artifact_id,
                '"revision:1"',
                ArtifactWrite(
                    content={"title": "newtopicneedle", "summary": "newtopicneedle current", "detail": "New body"}
                ),
            )
            # Both immutable bodies exist before binding. No body is edited during
            # registration, withdrawal, restoration, or the subsequent restart.
            historical_ref = ArtifactRef(family="topic-memory", artifact_id=old.artifact_id, revision=old.revision)
            current_ref = ArtifactRef(family="topic-memory", artifact_id=current.artifact_id, revision=current.revision)
            graph = runtime.lifecycle.for_scope(scope.scope_id)
            for root in ("historical-root", "current-root", "alias-root"):
                receipt = await sources.capture(CaptureSource(source_id=root, content=root, metadata={}))
                await graph.register_source(root, receipt.source_ref, revision=1)
            await graph.register_topic("historical", historical_ref, parents=("historical-root",), policy="all")
            await graph.register_topic("current", current_ref, parents=("current-root",), policy="all")
            await graph.register_topic("current-alias", current_ref, parents=("alias-root",), policy="all")
            await graph.apply(_event("current-root", 1, False))
            # A second support for the same exact version keeps that version usable.
            assert (
                await runtime.topic_memory.for_scope(scope.scope_id).search(
                    SearchTopicMemoryRequest(query="newtopicneedle")
                )
            ).hits
            await graph.apply(_event("alias-root", 1, False))
            assert await graph.resolve("historical", purpose="generation") == historical_ref
            with pytest.raises(SourceLifecycleError):
                await graph.resolve("current", purpose="generation")
            assert not (
                await runtime.topic_memory.for_scope(scope.scope_id).search(
                    SearchTopicMemoryRequest(query="newtopicneedle")
                )
            ).hits
            assert not await runtime.topic_memory.for_scope(scope.scope_id).browse(limit=10)

        async with open_builtin_runtime(config) as runtime:
            graph = runtime.lifecycle.for_scope(scope.scope_id)
            topics = runtime.topic_memory.for_scope(scope.scope_id)
            assert not (await topics.search(SearchTopicMemoryRequest(query="newtopicneedle"))).hits
            assert not await topics.browse(limit=10)
            prepared = await runtime.context.for_scope(scope.scope_id).prepare(
                PrepareContextRequest(
                    query="newtopicneedle",
                    assembly=ContextAssembly(sections=(ContextAssemblySection(family="topic-memory", limit=2),)),
                )
            )
            assert "newtopicneedle" not in (prepared.content or "")
            for node, ref, title in (
                ("historical", historical_ref, "oldtopicneedle"),
                ("current", current_ref, "newtopicneedle"),
            ):
                assert await graph.resolve(node, purpose="audit") == ref
                record = await runtime.records.for_scope(scope.scope_id).get_artifact_revision(
                    "topic-memory",
                    ref.artifact_id,
                    ref.revision,
                )
                assert record.content["title"] == title
            await graph.apply(_event("current-root", 2, True))
            assert await graph.resolve("current", purpose="generation") == current_ref
            assert (await topics.search(SearchTopicMemoryRequest(query="newtopicneedle"))).hits[
                0
            ].artifact_ref == current_ref
            assert (await topics.browse(limit=10))[0].artifact_ref == current_ref
            # Removing historical support cannot disable the independently supported head.
            await graph.apply(_event("historical-root", 1, False))
            assert (await topics.search(SearchTopicMemoryRequest(query="newtopicneedle"))).hits[
                0
            ].artifact_ref == current_ref

    asyncio.run(scenario())


def test_memory_historical_body_and_current_body_have_independent_eligibility(tmp_path: Path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'memory-versions.db'}"))

    async def scenario() -> None:
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(
                    title="Exact Memory bodies", summary="Independent supports", idempotency_key="memory-versions"
                )
            )
            memory = runtime.memory.for_scope(scope.scope_id)
            old = await memory.remember(
                RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text="oldmemoryneedle body"),))
            )
            assert old.entry is not None
            historical_ref = old.entry.citation
            current = await memory.revise(
                ReviseMemoryEntryRequest(citation=historical_ref, kind="fact", text="newmemoryneedle body")
            )
            assert current.entry is not None
            current_ref = current.entry.citation
            assert current_ref.entry_id == historical_ref.entry_id
            assert current_ref.entry_version_id != historical_ref.entry_version_id
            # Revision is completed before any lifecycle bindings are introduced.
            graph = runtime.lifecycle.for_scope(scope.scope_id)
            for root in ("historical-root", "current-root", "alias-root"):
                receipt = await runtime.sources.for_scope(scope.scope_id).capture(
                    CaptureSource(source_id=root, content=root, metadata={})
                )
                await graph.register_source(root, receipt.source_ref, revision=1)
            await graph.register_memory("historical", historical_ref, parents=("historical-root",), policy="all")
            await graph.register_memory("current", current_ref, parents=("current-root",), policy="all")
            await graph.register_memory("current-alias", current_ref, parents=("alias-root",), policy="all")
            await graph.apply(_event("current-root", 1, False))
            assert (await memory.search(SearchMemoryRequest(query="newmemoryneedle", mode="fts"))).hits
            await graph.apply(_event("alias-root", 1, False))
            assert await graph.resolve("historical", purpose="generation") == historical_ref
            with pytest.raises(SourceLifecycleError):
                await graph.resolve("current", purpose="generation")
            assert not (await memory.search(SearchMemoryRequest(query="newmemoryneedle", mode="fts"))).hits
            entries = await memory.list(include_inactive=True)
            assert len(entries.entries) == 1 and entries.entries[0].state == "inactive"
            assert entries.entries[0].citation.entry_version_id == current_ref.entry_version_id
            withdrawn_revision = entries.memory_ref

        async with open_builtin_runtime(config) as runtime:
            graph = runtime.lifecycle.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            assert not (await memory.search(SearchMemoryRequest(query="newmemoryneedle", mode="fts"))).hits
            assert await graph.resolve("historical", purpose="generation") == historical_ref
            for node, ref, text in (
                ("historical", historical_ref, "oldmemoryneedle body"),
                ("current", current_ref, "newmemoryneedle body"),
            ):
                assert await graph.resolve(node, purpose="audit") == ref
                original = await memory.get(GetMemoryEntryRequest(citation=ref))
                assert original.citation == ref and original.entry.text == text
            prepared = await runtime.context.for_scope(scope.scope_id).prepare(
                PrepareContextRequest(query="newmemoryneedle")
            )
            assert "newmemoryneedle" not in (prepared.content or "")
            # A successful replay does not manufacture a new Memory revision.
            await graph.apply(_event("alias-root", 1, False))
            assert (await memory.list(include_inactive=True)).memory_ref == withdrawn_revision
            await graph.apply(_event("current-root", 2, True))
            assert await graph.resolve("current", purpose="generation") == current_ref
            assert (await memory.search(SearchMemoryRequest(query="newmemoryneedle", mode="fts"))).hits[
                0
            ].entry_version_id == current_ref.entry_version_id
            await graph.apply(_event("historical-root", 1, False))
            assert (await memory.search(SearchMemoryRequest(query="newmemoryneedle", mode="fts"))).hits[
                0
            ].entry_version_id == current_ref.entry_version_id
            assert (
                await memory.get(GetMemoryEntryRequest(citation=historical_ref))
            ).entry.text == "oldmemoryneedle body"

    asyncio.run(scenario())
