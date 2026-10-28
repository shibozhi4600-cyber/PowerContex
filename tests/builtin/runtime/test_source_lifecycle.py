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

"""Native partial propagation, durable retries and stable historical bodies."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.service import MemoryService
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


@pytest.mark.parametrize(
    "change",
    [
        {"revision": True},
        {"revision": "1"},
        {"revision": 0},
        {"sequence": 1.0},
        {"sequence": False},
        {"sequence": -1},
        {"available": 1},
        {"available": "false"},
        {"node_id": " "},
        {"reason": "\t"},
        {"extra": "ignored"},
    ],
)
def test_lifecycle_events_reject_coercion_and_extra_fields(change) -> None:
    with pytest.raises(ValidationError):
        SourceLifecycleEvent.model_validate(
            {"node_id": "root", "revision": 1, "sequence": 1, "available": False, "reason": "withdraw"} | change
        )


def test_interrupted_native_transition_retries_after_restart_without_rewriting_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'retry.db'}"))
    withdraw = SourceLifecycleEvent(node_id="root", revision=1, sequence=1, available=False, reason="withdraw")
    restore = SourceLifecycleEvent(node_id="root", revision=1, sequence=2, available=True, reason="restore")

    async def scenario() -> None:
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Retry", summary="Native states", idempotency_key="retry")
            )
            lifecycle = runtime.lifecycle.for_scope(scope.scope_id)
            source = await runtime.sources.for_scope(scope.scope_id).capture(
                CaptureSource(source_id="root", content="origin", metadata={})
            )
            await lifecycle.register_source("root", source.source_ref, revision=1)
            memory = runtime.memory.for_scope(scope.scope_id)
            citations = []
            for index in range(2):
                result = await memory.remember(
                    RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"needle {index}"),))
                )
                assert result.entry is not None
                citations.append(result.entry.citation)
                await lifecycle.register_memory(str(index), result.entry.citation, parents=("root",), policy="all")
            before = (await memory.list()).memory_ref
            assert before is not None
            real_forget = MemoryService.forget
            calls = 0

            async def interrupted(service, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected native write failure")  # noqa: TRY003
                return await real_forget(service, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(MemoryService, "forget", interrupted)
                with pytest.raises(RuntimeError, match="injected"):
                    await lifecycle.apply(withdraw)
            partial = await memory.list(include_inactive=True)
            assert sorted(item.state for item in partial.entries) == ["active", "inactive"]
            for node in ("0", "1"):
                with pytest.raises(SourceLifecycleError):
                    await lifecycle.resolve(node, purpose="generation")

        async with open_builtin_runtime(config) as runtime:
            lifecycle = runtime.lifecycle.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            assert await lifecycle.apply(withdraw) == []
            inactive = await memory.list(include_inactive=True)
            assert inactive.memory_ref is not None
            assert inactive.memory_ref.revision == before.revision + 2
            assert all(item.state == "inactive" for item in inactive.entries)
            assert not (await memory.search(SearchMemoryRequest(query="needle", mode="fts"))).hits
            await lifecycle.apply(withdraw)
            assert (await memory.list()).memory_ref == inactive.memory_ref
            for index, citation in enumerate(citations):
                assert await lifecycle.resolve(str(index), purpose="audit") == citation
                assert (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.text == f"needle {index}"
            assert await lifecycle.apply(restore) == ["0", "1", "root"]
            active = await memory.list()
            assert len(active.entries) == 2
            for index, citation in enumerate(citations):
                assert await lifecycle.resolve(str(index), purpose="generation") == citation
                assert (
                    next(
                        item for item in active.entries if item.citation.entry_id == citation.entry_id
                    ).citation.entry_version_id
                    == citation.entry_version_id
                )
            await lifecycle.apply(restore)
            assert (await memory.list()).memory_ref == active.memory_ref

    asyncio.run(scenario())
