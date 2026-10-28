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

"""Count persisted Source identities after each Family's successful cursor."""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.tables import SOURCES_TABLE


async def count_eligible_scopes(
    connection: AsyncConnection,
    scope_ids: tuple[str, ...],
    binding_name: str,
    threshold: int,
) -> frozenset[str]:
    """Qualify a bounded metadata page without reading or filtering Source bodies."""

    if not scope_ids:
        return frozenset()
    cursors = await SourceCursorRepository().load_many(connection, scope_ids, binding_name)
    table = SOURCES_TABLE
    counts = await connection.scalars(
        select(table.c.scope_id)
        .where(
            or_(
                *(
                    (table.c.scope_id == scope_id)
                    & (table.c.journal_position > (cursors[scope_id].cursor.sequence if scope_id in cursors else 0))
                    for scope_id in scope_ids
                )
            )
        )
        .group_by(table.c.scope_id)
        .having(func.count() >= threshold)
    )
    return frozenset(counts)
