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

"""Durable dependency graphs and Topic generation eligibility."""

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.sources import SourceRepository
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_GRAPH_TABLE, SOURCE_LIFECYCLE_TOPICS_TABLE
from powercontext.sources import SourceMaterialization, SourceRef


async def unavailable_topics(connection: AsyncConnection, scope_id: str) -> tuple[str, ...]:
    return tuple(
        str(item)
        for item in (
            await connection.scalars(
                select(SOURCE_LIFECYCLE_TOPICS_TABLE.c.artifact_id).where(
                    SOURCE_LIFECYCLE_TOPICS_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_TOPICS_TABLE.c.available.is_(False),
                )
            )
        ).all()
    )


class SourceLifecycleStore:
    """Store one bounded, single-writer graph per Scope without local state."""

    def __init__(self, database: AsyncDatabase, sources: SourceRepository) -> None:
        self.database = database
        self.sources = sources

    async def load(self, scope_id: str) -> bytes | None:
        async with self.database.transaction() as connection:
            value = await connection.scalar(
                select(SOURCE_LIFECYCLE_GRAPH_TABLE.c.payload).where(
                    SOURCE_LIFECYCLE_GRAPH_TABLE.c.scope_id == scope_id
                )
            )
            return None if value is None else bytes(value)

    async def save(self, scope_id: str, payload: bytes, topics: dict[str, bool]) -> None:
        async with self.database.transaction() as connection:
            exists = await connection.scalar(
                select(SOURCE_LIFECYCLE_GRAPH_TABLE.c.scope_id).where(
                    SOURCE_LIFECYCLE_GRAPH_TABLE.c.scope_id == scope_id
                )
            )
            if exists is None:
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_GRAPH_TABLE).values(scope_id=scope_id, payload=payload)
                )
            else:
                await connection.execute(
                    update(SOURCE_LIFECYCLE_GRAPH_TABLE)
                    .where(SOURCE_LIFECYCLE_GRAPH_TABLE.c.scope_id == scope_id)
                    .values(payload=payload)
                )
            # The contract bounds this projection to four Topics per graph.
            # Publish graph eligibility and its search projection atomically.
            await connection.execute(
                delete(SOURCE_LIFECYCLE_TOPICS_TABLE).where(SOURCE_LIFECYCLE_TOPICS_TABLE.c.scope_id == scope_id)
            )
            if topics:
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_TOPICS_TABLE),
                    [
                        {"scope_id": scope_id, "artifact_id": artifact_id, "available": available}
                        for artifact_id, available in topics.items()
                    ],
                )

    async def validate_source(self, scope_id: str, reference: SourceRef) -> None:
        async with self.database.transaction() as connection:
            stored = await self.sources.get(connection, scope_id, reference)
            if stored.value.materialization != SourceMaterialization.CAPTURED:
                raise ValueError("Lifecycle requires a captured Source")  # noqa: TRY003
