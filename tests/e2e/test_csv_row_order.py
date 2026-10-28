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

"""CSV column order and exact values survive canonical storage and restarts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import rfc8785

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    SourceDefinitionRegistry,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowSource,
    parse_csv_document,
)


def test_csv_row_views_keep_column_order_across_two_restarts_and_idempotent_replay(tmp_path: Path) -> None:
    rows = parse_csv_document(
        CSVDocumentInput(
            document_id="表/顺序",
            primary_key="id",
            content='z,id,a\r\n"中文\r\n含""引号""",001,\r\n尾部, 002 ,0007\r\n'.encode(),
        )
    )
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'csv-order.db'}"))

    async def scenario() -> None:
        sources = [await registry.resolve(row) for row in rows]
        observations = [project_source_for_transport(registry, source) for source in sources]
        canonical = {source.name: rfc8785.dumps(source.model_dump(mode="json")) for source in observations}
        expected_rows = {source.name: row for source, row in zip(sources, rows, strict=True)}
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="CSV order", summary="Exact captured rows", idempotency_key="csv-order")
            )
            await runtime.ingestion.register(manifest_for_definition(CSV_ROW_SOURCE_DEFINITION))
            receipts = [
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
                for observation in observations
            ]

        for _ in range(2):
            async with open_builtin_contexts(config) as contexts:
                entries = await (await contexts.get(scope.scope_id)).sources.journal.entries()
                assert len(entries) == len(rows)
                for entry in entries:
                    source = entry.source
                    assert isinstance(source, SourceObservation)
                    original = expected_rows[source.name]
                    assert source.definition_fingerprint == observations[0].definition_fingerprint
                    assert rfc8785.dumps(source.model_dump(mode="json")) == canonical[source.name]
                    assert await registry.read(CSVRowSource.model_validate(source.payload)) == original
                    raw = source.projection(TEXT_EVIDENCE_PROJECTION_KEY)
                    registered = registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
                    assert isinstance(raw, dict)
                    assert isinstance(registered, dict)
                    typed = TextEvidence.model_validate(raw)
                    expected_pairs = list(zip(original.columns, original.values, strict=True))
                    for metadata in (raw["metadata"], registered["metadata"], typed.metadata):
                        assert isinstance(metadata, dict)
                        record = metadata["row"]
                        assert isinstance(record, dict)
                        assert all(isinstance(value, str) for value in record.values())
                        assert list(record.items()) == expected_pairs
                    original_projection = registry.project(
                        CSVRowSource.model_validate(source.payload), TEXT_EVIDENCE_PROJECTION_KEY
                    )
                    assert isinstance(original_projection, dict)
                    original_content = original_projection["content"]
                    assert isinstance(original_content, str)
                    assert typed.content == original_content

            async with open_builtin_runtime(config) as runtime:
                for observation, receipt in zip(observations, receipts, strict=True):
                    assert (
                        await runtime.ingestion.submit(
                            SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                        )
                        == receipt
                    )

        async with open_builtin_contexts(config) as contexts:
            assert len(await (await contexts.get(scope.scope_id)).sources.journal.entries()) == len(rows)

    asyncio.run(scenario())
