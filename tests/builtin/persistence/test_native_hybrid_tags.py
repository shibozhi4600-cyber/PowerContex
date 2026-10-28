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

"""Execute and EXPLAIN the real tag-filtered seekdb hybrid candidate query."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from sqlalchemy import Engine, event

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryCitation, MemoryEntryInput
from powercontext.builtin.inference import EmbeddingResult
from powercontext.builtin.persistence.seekdb import SeekDBConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts
from powercontext.builtin.tags import MemoryEntryTagTarget, TagFilter

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or platform.machine().lower() not in {"x86_64", "amd64"}
    or importlib.util.find_spec("pylibseekdb") is None,
    reason="Requires optional pylibseekdb on Linux x86_64; the fixed contest environment runs this real index check.",
)


class _UnitEmbedding:
    profile = EmbeddingProfile(
        profile_id="native-tagged-hybrid-v1",
        model="offline-unit-vectors",
        dimension=3,
        distance="l2",
        normalization="unit",
    )

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return EmbeddingResult(vectors=tuple((1.0, 0.0, 0.0) for _ in texts))


@pytest.mark.asyncio
async def test_tagged_hybrid_filters_current_scope_and_uses_both_native_indexes() -> None:
    # Keep the embedded database's Unix socket path short, independently of
    # pytest's per-test directory name and the checkout's absolute path.
    with TemporaryDirectory(prefix="pc-hybrid-tags-", dir=os.environ.get("TMPDIR")) as directory:
        async with open_builtin_contexts(
            BuiltinConfig(database=SeekDBConfig(path=Path(directory) / "db")), embedding_model=_UnitEmbedding()
        ) as contexts:
            memory = (await contexts.get("tagged-scope")).artifacts.memory
            original = await memory.remember(
                memory=None,
                entries=(
                    MemoryEntryInput(kind="fact", text="Alpha selected original evidence."),
                    MemoryEntryInput(kind="fact", text="Alpha untagged decoy evidence."),
                ),
                mode="append",
            )
            assert original is not None
            old_target = next(entry for entry in await memory.entries(original) if "selected" in entry.text)
            current = await memory.remember(
                memory=original,
                entries=(MemoryEntryInput(entry=old_target, kind="fact", text="Alpha selected revised evidence."),),
                mode="append",
            )
            assert current is not None
            target = next(entry for entry in await memory.entries(current) if entry.entry_id == old_target.entry_id)
            assert target.entry_version_id != old_target.entry_version_id
            tag_target = MemoryEntryTagTarget(artifact_id=current.artifact_id, entry_id=target.entry_id)
            previous_tags = await contexts.records.get_tags("tagged-scope", tag_target)
            await contexts.records.replace_tags(
                "tagged-scope", tag_target, ("selected",), expected_etag=previous_tags.etag
            )

            other_memory = (await contexts.get("other-scope")).artifacts.memory
            other = await other_memory.remember(
                memory=None,
                entries=(MemoryEntryInput(kind="fact", text="Alpha selected other scope evidence."),),
                mode="append",
            )
            assert other is not None
            (other_entry,) = await other_memory.entries(other)
            other_target = MemoryEntryTagTarget(artifact_id=other.artifact_id, entry_id=other_entry.entry_id)
            other_tags = await contexts.records.get_tags("other-scope", other_target)
            await contexts.records.replace_tags(
                "other-scope", other_target, ("selected",), expected_etag=other_tags.etag
            )

            statements: list[tuple[str, Any]] = []

            def record(_connection, _cursor, statement, parameters, _context, _executemany) -> None:
                statements.append((statement, parameters))

            event.listen(Engine, "before_cursor_execute", record)
            try:
                result = await memory.search(
                    "alpha", memories=(current,), mode="hybrid", limit=3, tag_filter=TagFilter(tags=("selected",))
                )
            finally:
                event.remove(Engine, "before_cursor_execute", record)

            candidates = [
                (statement, parameters)
                for statement, parameters in statements
                if "pc_memory_entry_heads" in statement
                and "pc_memory_vector_entries" in statement
                and "AGAINST" in statement.upper()
                and "l2_distance" in statement.lower()
            ]
            assert len(candidates) == 1, "tagged hybrid search must issue exactly one native candidate statement"
            statement, parameters = candidates[0]
            assert "pc_artifact_tags" in statement, "the captured candidate SQL must apply the tag predicate"
            assert len(result.hits) == 1, "the untagged and other-scope decoys must both be excluded"
            (hit,) = result.hits
            assert hit.memory_ref == current.as_ref()
            assert (hit.entry_id, hit.entry_version_id, hit.text) == (
                target.entry_id,
                target.entry_version_id,
                target.text,
            )
            assert hit.matched_by and set(hit.matched_by) <= {"fts", "vector"}
            exact = await memory.validate_citation(
                MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
            )
            assert exact == target

            # Replay the tagged SQL actually executed above, including its
            # original driver parameters; do not substitute an untagged plan.
            async with contexts.database.transaction() as connection:
                rows = (await connection.exec_driver_sql(f"EXPLAIN {statement}", parameters)).all()
            plan = "\n".join(str(row[0]) for row in rows)
            assert "TEXT RETRIEVAL SCAN" in plan, f"Tagged hybrid lost the native FTS index:\n{plan}"
            assert "VECTOR INDEX" in plan, f"Tagged hybrid lost the native vector index:\n{plan}"
