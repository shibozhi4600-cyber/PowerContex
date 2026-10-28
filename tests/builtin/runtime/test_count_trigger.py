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

"""Persistent journal eligibility behind automatic count admission."""

from __future__ import annotations

import asyncio

from powercontext.builtin.artifacts.topic_memory import TOPIC_MEMORY_SOURCE_WINDOW_BINDING
from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.sources import SourceRepository
from powercontext.builtin.persistence.sqlite import SQLiteConfig, SQLiteProfile
from powercontext.builtin.persistence.tables import SHARED_TABLES
from powercontext.builtin.runtime.count_trigger import count_eligible_scopes
from powercontext.builtin.sources import BUILTIN_SOURCE_REGISTRY, ContentSource, SourceCursor
from powercontext.builtin.triggers import SOURCE_WINDOW_TRIGGER_NAME
from powercontext.sources import SourceMaterialization


def test_count_uses_stable_journal_identities_and_family_cursor_across_restart(tmp_path) -> None:
    config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'count-journal.db'}")

    async def scenario() -> None:
        sources = SourceRepository(BUILTIN_SOURCE_REGISTRY)
        first = ContentSource(name="first", materialization=SourceMaterialization.CAPTURED, content="first source")
        second = ContentSource(name="second", materialization=SourceMaterialization.CAPTURED, content="second source")
        async with (
            SQLiteProfile.open(config, tables=SHARED_TABLES) as profile,
            profile.database.transaction() as connection,
        ):
            original, created = await sources.add_with_status(connection, "left", first)
            replay, replay_created = await sources.add_with_status(connection, "left", first)
            assert created and not replay_created
            assert original.journal_position == replay.journal_position
            await sources.add_with_status(connection, "left", second)
            await sources.add_with_status(connection, "right", first)
            assert await count_eligible_scopes(connection, ("left", "right"), SOURCE_WINDOW_TRIGGER_NAME, 2) == {"left"}
            await SourceCursorRepository().save(
                connection,
                "left",
                SOURCE_WINDOW_TRIGGER_NAME,
                SourceCursor(sequence=original.journal_position),
                expected_generation=None,
            )
            assert not await count_eligible_scopes(connection, ("left", "right"), SOURCE_WINDOW_TRIGGER_NAME, 2)
            assert await count_eligible_scopes(connection, ("left",), TOPIC_MEMORY_SOURCE_WINDOW_BINDING, 2) == {"left"}
        async with (
            SQLiteProfile.open(config, tables=SHARED_TABLES) as profile,
            profile.database.transaction() as connection,
        ):
            assert not await count_eligible_scopes(connection, ("left", "right"), SOURCE_WINDOW_TRIGGER_NAME, 2)
            assert await count_eligible_scopes(connection, ("left",), SOURCE_WINDOW_TRIGGER_NAME, 1) == {"left"}
            assert await count_eligible_scopes(connection, ("left",), TOPIC_MEMORY_SOURCE_WINDOW_BINDING, 2) == {"left"}
            assert not await count_eligible_scopes(connection, (), SOURCE_WINDOW_TRIGGER_NAME, 1)

    asyncio.run(scenario())
