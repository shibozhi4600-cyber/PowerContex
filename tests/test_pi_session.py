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

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from powercontext.sources import ConnectorBinding, SourceDefinitionRegistry, SourceRef, project_source_for_transport
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventInput,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


def _archive(records: list[dict[str, JsonValue]], session_id: str = "session") -> bytes:
    header = {"type": "session", "version": 3, "id": session_id}
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in [header, *records]).encode()


def _message(event_id: str, parent: str | None, body: str) -> dict[str, JsonValue]:
    return {"type": "message", "id": event_id, "parentId": parent, "message": {"role": "user", "content": body}}


async def test_full_tools_and_control_ancestors_survive_transport_without_truncation() -> None:
    output = "完整工具结果\n" * 10_000
    records: list[dict[str, JsonValue]] = [
        _message("root", None, "  original request  "),
        {"type": "thinking_level_change", "id": "control", "parentId": "root"},
        {
            "type": "message",
            "id": "call",
            "parentId": "control",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "not visible"},
                    {"type": "toolCall", "name": "read", "id": "invoke", "arguments": {"z": 1, "a": "中文"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "result",
            "parentId": "call",
            "message": {"role": "toolResult", "toolName": "read", "toolCallId": "invoke", "content": output},
            "extra_metadata": {"retain": True},
        },
        {"type": "compaction", "id": "summary", "parentId": "result", "summary": "compressed view"},
    ]
    events = parse_pi_session(PiSessionInput(content=b"\xef\xbb\xbf" + _archive(records)))
    assert [event.event_id for event in events] == ["root", "call", "result", "summary"]
    assert events[0].content == "user:   original request  "
    assert events[1].content == 'assistant: [tool_call read invoke] {"a":"中文","z":1}'
    assert events[2].content == f"toolResult: [tool_result read invoke] {output}"
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    for event in events:
        source = await registry.resolve(event)
        assert len(source.name) <= 256
        assert await registry.read(source) == event
        observed = project_source_for_transport(registry, source)
        reloaded = type(observed).model_validate_json(observed.model_dump_json())
        evidence = TextEvidence.model_validate(reloaded.projection(TEXT_EVIDENCE_PROJECTION_KEY))
        assert evidence.content == event.content
        assert evidence.metadata["parent_id"] == event.parent_id
    assert events[2].record == records[3]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff",
        b"[]\n",
        b'{"type":"session","version":true,"id":"s"}',
        b'{"type":"session","version":3,"id":"s","id":"other"}',
        b'{"type":"session","version":3,"id":"s"}\nnull',
        b'{"type":"session","version":3,"id":"s"}\n{"id":"a","type":"control"}',
        b'{"type":"session","version":3,"id":"s"}\n{"id":"a","type":"control","parentId":"b"}',
        b'{"type":"session","version":3,"id":"s"}\n{"id":"a","type":"control","parentId":"a"}',
        b'{"type":"session","version":3,"id":"s"}\n{"id":"a","type":"control","parentId":null,"x":NaN}',
        b'{"type":"session","version":3,"id":"s"}\n{"id":"a","type":"control","parentId":null,"x":1e999}',
    ],
)
def test_invalid_archive_is_rejected_before_returning_events(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw))


def test_default_control_leaf_and_header_only_archive() -> None:
    root = _message("root", None, "before control")
    control: dict[str, JsonValue] = {"id": "control", "type": "model_change", "parentId": "root"}
    assert [event.event_id for event in parse_pi_session(PiSessionInput(content=_archive([root, control])))] == ["root"]
    assert parse_pi_session(PiSessionInput(content=_archive([]))) == ()
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=_archive([root]), leaf_id="missing"))


class _Checkpoints:
    def __init__(self) -> None:
        self.value: JsonValue | None = None
        self.writes = 0

    async def load(self, _binding: ConnectorBinding, /) -> JsonValue | None:
        return self.value

    async def save(self, _binding: ConnectorBinding, value: JsonValue | None, /, *, expected: JsonValue | None) -> None:
        assert self.value == expected
        self.value = json.loads(json.dumps(value))
        self.writes += 1


class _Sink:
    def __init__(self) -> None:
        self.captured: dict[str, PiEventInput] = {}
        self.attempts: list[str] = []
        self.fail_once: str | None = None

    async def submit(self, _binding: ConnectorBinding, item_id: str, definition: str, value: object, /) -> SourceRef:
        assert isinstance(value, PiEventInput)
        self.attempts.append(value.event_id)
        if value.event_id == self.fail_once:
            self.fail_once = None
            raise RuntimeError("temporary sink failure")  # noqa: TRY003
        if item_id in self.captured:
            assert self.captured[item_id] == value
        self.captured[item_id] = value
        return SourceRef(source_type=definition, source_id=item_id)


async def test_partial_failure_replays_safely_and_off_branch_mutation_cannot_advance_checkpoint(tmp_path: Path) -> None:
    records = [_message("root", None, "shared"), _message("left", "root", "left"), _message("right", "root", "right")]
    path = tmp_path / "session.jsonl"
    path.write_bytes(_archive(records))
    checkpoints, sink = _Checkpoints(), _Sink()
    lifecycle = ConnectorLifecycle(sink=sink, checkpoints=checkpoints)
    binding = ConnectorBinding(
        scope_id="scope", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )
    sink.fail_once = "left"
    with pytest.raises(RuntimeError):
        await lifecycle.run(PiSessionConnector(path, "left"), binding)
    assert checkpoints.value is None and len(sink.captured) == 1
    await lifecycle.run(PiSessionConnector(path, "left"), binding)
    assert len(sink.captured) == 2 and sink.attempts.count("root") == 2
    switched = await lifecycle.run(PiSessionConnector(path, "right"), binding)
    assert len(switched.items) == 1 and len(sink.captured) == 3
    stable = checkpoints.value
    attempts = list(sink.attempts)
    records[1] = _message("left", "root", "rewritten, even though right is selected")
    path.write_bytes(_archive(records))
    with pytest.raises(ValueError, match="changed or removed"):
        await lifecycle.run(PiSessionConnector(path, "right"), binding)
    assert checkpoints.value == stable and sink.attempts == attempts
