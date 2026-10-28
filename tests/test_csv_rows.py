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

"""CSV evidence contracts, including exact snapshots across transport encoding."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceMaterialization,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _parse(content: bytes, *, document_id: str = "table", primary_key: str = "id") -> tuple[CSVRowInput, ...]:
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


def test_values_and_distinct_space_keys_survive_transport_roundtrip() -> None:
    rows = _parse(
        '\ufeff" id ",备注,empty,number,flag\r\n'
        ' key ,"第一行\r\n第二行,含""引号""",,0007,false\r\n'
        "key ,普通,,0,true\r\n".encode(),
        document_id="表:/%",
        primary_key=" id ",
    )
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    assert rows[0].values == (" key ", '第一行\r\n第二行,含"引号"', "", "0007", "false")
    assert rows[0].primary_key_value != rows[1].primary_key_value
    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
    assert manifest.name == "csv-row"
    assert manifest.version == "1"

    async def scenario() -> None:
        for row in rows:
            source = await registry.resolve(row)
            assert isinstance(source, CSVRowSource)
            assert source.materialization is SourceMaterialization.CAPTURED
            captured = CSVRowSource.model_validate_json(source.model_dump_json())
            assert await registry.read(captured) == row
            observation = project_source_for_transport(registry, captured)
            persisted = SourceObservation.model_validate_json(observation.model_dump_json())
            assert persisted.definition_fingerprint == manifest.fingerprint
            evidence = TextEvidence.model_validate(registry.project(persisted, TEXT_EVIDENCE_PROJECTION_KEY))
            original = dict(zip(row.columns, row.values, strict=True))
            assert evidence.source_id == source.name == source.source_id
            assert evidence.content == json.dumps(original, ensure_ascii=False, separators=(",", ":"))
            assert list(evidence.metadata["row"]) == list(row.columns)
            assert evidence.metadata == {
                "document_id": row.document_id,
                "primary_key": row.primary_key,
                "primary_key_value": row.primary_key_value,
                "columns": list(row.columns),
                "row": original,
            }

    asyncio.run(scenario())


@pytest.mark.parametrize("separator", [b"\n", b"\r\n", b"\r"])
def test_empty_physical_records_are_ignored_without_stripping_fields(separator: bytes) -> None:
    rows = _parse(separator.join((b"id,name", b"", b"001,  name  ", b"", b"002,")))
    assert tuple(row.values for row in rows) == (("001", "  name  "), ("002", ""))
    assert _parse(b"id,name") == ()


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\n",
        b"\xef\xbb\xbf",
        b"id,id\n1,2",
        b"id,\t\n1,2",
        b"id,\n1,2",
        b"id,name\n1,valid\n1,duplicate",
        b"id,name\n1,valid\n ,blank",
        b"id,name\n1,valid\n2,wide,extra",
        b"id,name\n1,valid\n2",
        b'id,name\n1,valid\n2,"unterminated',
        b'id,name\n1,valid\n2,"closed"trailing',
        b"id,name\n1,valid\n2,\xff",
        b"id,name\n1,valid\n2,\xed\xa0\x80",
        b'id,name\n1,valid\n"",blank',
    ],
)
def test_invalid_complete_document_is_rejected(content: bytes) -> None:
    with pytest.raises(ValueError):
        _parse(content)


def test_column_names_are_matched_exactly_and_bom_only_removed_at_start() -> None:
    row = _parse("id, id ,label\n1,2,\ufeffvalue".encode())[0]
    assert row.columns == ("id", " id ", "label")
    assert row.values[-1] == "\ufeffvalue"
    assert _parse(b" id ,id\n1,2", primary_key=" id ")[0].primary_key_value == "1"
    with pytest.raises(ValueError):
        _parse(b" id ,name\n1,one", primary_key="id")


def test_identity_length_is_measured_after_encoding_and_has_no_component_collisions() -> None:
    assert len(csv_row_source_id("d", "x" * 250)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "x" * 251)
    assert len(csv_row_source_id("d", "中" * 27 + "x" * 7)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "中" * 28)
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")
    assert csv_row_source_id("%3A", "/") == "csv:%253A:%2F"


@pytest.mark.parametrize("document_id", ["", " ", " table", "table\n"])
def test_invalid_document_identity_is_rejected_even_for_header_only_input(document_id: str) -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name", document_id=document_id)


@pytest.mark.parametrize("key", ["", " ", "\t"])
def test_blank_primary_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        csv_row_source_id("document", key)
    with pytest.raises(ValueError):
        _parse(b"id,name", primary_key=key)


def test_direct_row_input_rejects_inconsistent_record_and_source_identity() -> None:
    row = _parse(b"id,name\n001,name")[0]
    for changes in (
        {"values": ("different", "name")},
        {"columns": ("id", "id")},
        {"values": ("001",)},
        {"primary_key": "missing"},
    ):
        with pytest.raises(ValueError):
            CSVRowInput.model_validate(row.model_dump() | changes)
    with pytest.raises(ValueError):
        CSVRowSource(name="wrong", source_id="wrong", **row.model_dump())


def test_identity_stays_stable_after_non_key_edit_and_row_reorder() -> None:
    before = _parse(b"id,name\n001,one\n002,two")
    after = _parse(b"id,name\n002,changed\n001,one")
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))

    async def capture(rows: tuple[CSVRowInput, ...]) -> dict[str, str]:
        return {row.primary_key_value: (await registry.resolve(row)).name for row in rows}

    assert asyncio.run(capture(before)) == asyncio.run(capture(after))
