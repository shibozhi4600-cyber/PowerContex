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

"""Real Runtime preparation is reconstructed from serialized, receiver-owned data."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import TypedDict

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    ContextAssembly,
    GetMemoryEntryRequest,
    PrepareContextRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.context_pack import ContextPack, ContextPackDelta, apply_context_pack_delta
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


class _LiteralPipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


class _ReceiverPack(TypedDict):
    digest: str
    items: list[str]


def _oracle_digest(items: list[str]) -> str:
    unsigned = {"schema": "powercontext.context-pack.v1", "items": items}
    return "sha256:" + hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_serialized_native_contexts_replay_across_revision_query_and_empty_changes(tmp_path: Path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'contexts.db'}"))

    async def scenario() -> None:
        async with open_builtin_runtime(config, candidate_pipeline=_LiteralPipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(
                    title="Receiver transport",
                    summary="Literal references",
                    idempotency_key="receiver-transport",
                )
            )
            sources = runtime.sources.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            context = runtime.context.for_scope(scope.scope_id)
            await sources.capture(
                CaptureSource(source_id="one", content="Cobalt sensor calibrated at 53 degrees.", metadata={})
            )
            while (await memory.flush()).processed:
                pass
            first = await context.prepare_incremental(PrepareContextRequest(query="sensor"))
            assert isinstance(first, ContextPack)
            receiver = ContextPack.from_bytes(first.to_bytes())
            await sources.capture(CaptureSource(source_id="two", content="Amber valve opens at 29 bars.", metadata={}))
            while (await memory.flush()).processed:
                pass
            for request in (
                PrepareContextRequest(query="sensor"),
                PrepareContextRequest(query="valve"),
                PrepareContextRequest(query="valve", assembly=ContextAssembly(sections=())),
            ):
                expected = await context.prepare(request)
                delta = await context.prepare_incremental(
                    request, base_pack=ContextPack.from_bytes(receiver.to_bytes())
                )
                assert isinstance(delta, ContextPackDelta)
                receiver = apply_context_pack_delta(receiver, ContextPackDelta.from_bytes(delta.to_bytes()))
                assert b"".join(receiver.items) == (expected.content or "").encode()
            assert not receiver.items

    asyncio.run(scenario())


def test_selected_native_evidence_citations_read_back_the_exact_original(tmp_path: Path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'evidence.db'}"))
    text = "Routine preparation notes. " * 100 + "海洋 sensor “blue” records 53°C at station\\north."

    async def scenario() -> None:
        async with open_builtin_runtime(config, candidate_pipeline=_LiteralPipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Evidence", summary="Exact source", idempotency_key="evidence-native")
            )
            await runtime.sources.for_scope(scope.scope_id).capture(
                CaptureSource(source_id="literal", content=text, metadata={})
            )
            memory = runtime.memory.for_scope(scope.scope_id)
            while (await memory.flush()).processed:
                pass
            prepared = await runtime.context.for_scope(scope.scope_id).prepare(
                PrepareContextRequest(
                    query="海洋 sensor blue station",
                    max_bytes=1200,
                    evidence_selection=True,
                )
            )
            assert prepared.content and prepared.content_bytes <= 1200
            items = json.loads(prepared.content.splitlines()[-2])["items"]
            assert any("53°C" in item["content"] for item in items)
            for item in items:
                citation = item["citation"]
                assert citation["scope_id"] == scope.scope_id
                entry = await memory.get(
                    GetMemoryEntryRequest(
                        citation=MemoryCitation(
                            memory_ref=ArtifactRef.model_validate(citation["memory_ref"]),
                            entry_id=citation["entry_id"],
                            entry_version_id=citation["entry_version_id"],
                        )
                    )
                )
                assert entry.entry.text[item["start"] : item["end"]] == item["content"]

    asyncio.run(scenario())


def test_forty_native_turns_report_all_wire_costs_with_an_independent_receiver(tmp_path: Path) -> None:
    """Synthetic local schedule measures protocol costs, not the private QMSum gate."""
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'costs.db'}"))

    async def scenario() -> dict[str, int | float]:
        raw_total = wire_total = full_total = gzip_total = 0
        bodies: set[bytes] = set()
        receiver: _ReceiverPack | None = None
        sender_base: ContextPack | None = None
        async with open_builtin_runtime(config, candidate_pipeline=_LiteralPipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Costs", summary="Forty local requests", idempotency_key="costs")
            )
            sources = runtime.sources.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            context = runtime.context.for_scope(scope.scope_id)

            async def capture(index: int, term: str) -> None:
                text = f"Engineer {index}: The {term} device requires a calibration review. " + (
                    "We recorded the measurement procedure and discussed the operating tolerances. " * 5
                )
                await sources.capture(CaptureSource(source_id=f"turn-{index}", content=text, metadata={}))
                while (await memory.flush()).processed:
                    pass

            for index in range(8):
                await capture(index, "cobalt" if index < 4 else "amber")
            for turn in range(40):
                if turn == 10:
                    await capture(8, "cobalt")
                if turn == 30:
                    await capture(9, "amber")
                request = PrepareContextRequest(query="cobalt" if turn < 20 else "amber", max_bytes=8192)
                prepared = await context.prepare(request)
                assert prepared.content
                body = prepared.content.encode()
                bodies.add(body)
                raw_total += len(body)
                gzip_total += len(gzip.compress(body, compresslevel=9, mtime=0))
                full_total += ContextPack.from_content(prepared.content).byte_count
                sent = await context.prepare_incremental(request, base_pack=sender_base)
                wire = sent.to_bytes()
                wire_total += len(wire)
                payload = json.loads(wire)
                if receiver is None:
                    assert payload["schema"] == "powercontext.context-pack.v1"
                    receiver = payload
                else:
                    assert payload["schema"] == "powercontext.context-pack-delta.v1"
                    assert payload["base_digest"] == receiver["digest"]
                    items = receiver["items"][: payload["prefix_items"]] + payload["appended"]
                    assert _oracle_digest(items) == payload["target_digest"]
                    receiver = {"digest": payload["target_digest"], "items": items}
                received = b"".join(base64.b64decode(item, validate=True) for item in receiver["items"])
                assert received == body
                # Sender receives only the independently decoded complete receiver pack.
                sender_base = ContextPack.build(tuple(base64.b64decode(item) for item in receiver["items"]))
        return {
            "requests": 40,
            "different_contexts": len(bodies),
            "raw_full_bytes": raw_total,
            "incremental_bytes": wire_total,
            "full_pack_bytes": full_total,
            "gzip_bytes": gzip_total,
            "saving_vs_raw": 1 - wire_total / raw_total,
            "saving_vs_full_pack": 1 - wire_total / full_total,
            "saving_vs_gzip": 1 - wire_total / gzip_total,
        }

    observed = asyncio.run(scenario())
    assert observed["different_contexts"] >= 4
    assert observed["saving_vs_raw"] >= 0.45
    print("LOCAL_SYNTHETIC_TRANSPORT " + json.dumps(observed, sort_keys=True))
