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

"""Capture the complete visible event history on one Pi v3 session branch."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    """A complete offline Pi archive and an optional selected branch leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One complete visible event, including its unabridged original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("session_id", "event_id", "event_type", "role")
    @classmethod
    def require_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pi event identity and role must not be blank")  # noqa: TRY003
        return value


class PiEventSource(Source, PiEventInput):
    """Immutable-by-identity captured evidence; never reads a live session file."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED


@dataclass(frozen=True)
class _Archive:
    session_id: str
    records: dict[str, dict[str, JsonValue]]
    events: tuple[PiEventInput, ...]


def _json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Pi JSON records must not contain duplicate fields")  # noqa: TRY003
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Pi JSON must not contain {value}")  # noqa: TRY003


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Pi JSON numbers must be finite")  # noqa: TRY003
    return result


def _record_digest(record: dict[str, JsonValue]) -> str:
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _event_identity(session_id: str, event_id: str) -> str:
    raw = json.dumps((session_id, event_id), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"pi:{hashlib.sha256(raw).hexdigest()}"


def _visible_blocks(value: JsonValue, *, tool_calls: bool = False) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Pi text blocks must contain string text")  # noqa: TRY003
            parts.append(text)
        elif tool_calls and block.get("type") == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if not isinstance(name, str) or not name.strip() or not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("Pi tool calls require a name and ID")  # noqa: TRY003
            if "arguments" not in block:
                raise ValueError("Pi tool calls require JSON arguments")  # noqa: TRY003
            arguments = json.dumps(block["arguments"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parts.append(f"[tool_call {name} {call_id}] {arguments}")
    return "\n".join(parts)


def _event_body(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    event_type = record.get("type")
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError("Pi summary events require string summary")  # noqa: TRY003
        return (event_type, summary) if summary.strip() else None
    if event_type == "custom_message":
        body = _visible_blocks(record.get("content"))
        return ("custom", body) if body.strip() else None
    if event_type != "message":
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        raise ValueError("Pi message records require a message object")  # noqa: TRY003, TRY004
    return _message_body(message)


def _message_body(message: dict[str, JsonValue]) -> tuple[str, str] | None:
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Pi messages require a role")  # noqa: TRY003
    if role == "bashExecution":
        command, output = message.get("command"), message.get("output")
        if not isinstance(command, str) or not isinstance(output, str):
            raise ValueError("Pi bash execution requires string command and output")  # noqa: TRY003
        return (role, f"$ {command}\n{output}") if command.strip() or output.strip() else None
    body = _visible_blocks(message.get("content"), tool_calls=True)
    if not body.strip():
        return None
    if role == "toolResult":
        name, call_id = message.get("toolName"), message.get("toolCallId")
        if not isinstance(name, str) or not name.strip() or not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Pi tool results require a tool name and call ID")  # noqa: TRY003
        body = f"[tool_result {name} {call_id}] {body}"
    return role, body


def _session_id(header: JsonValue) -> str:
    if not isinstance(header, dict):
        raise ValueError("Pi archives require a v3 session header")  # noqa: TRY003, TRY004
    session_id = header.get("id")
    if (
        header.get("type") != "session"
        or type(header.get("version")) is not int
        or header["version"] != 3
        or not isinstance(session_id, str)
        or not session_id.strip()
    ):
        raise ValueError("Pi archives require a v3 session header with a session ID")  # noqa: TRY003
    _record_digest(header)
    return session_id


def _parse_archive(value: PiSessionInput) -> _Archive:
    decoded = value.content.decode("utf-8-sig")
    entries = [
        json.loads(line, object_pairs_hook=_json_object, parse_constant=_invalid_constant, parse_float=_finite_float)
        for line in decoded.splitlines()
        if line.strip()
    ]
    session_id = _session_id(entries[0] if entries else None)
    records: dict[str, dict[str, JsonValue]] = {}
    for record in entries[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi records must be JSON objects")  # noqa: TRY003, TRY004
        event_id, parent_id, event_type = record.get("id"), record.get("parentId"), record.get("type")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in records:
            raise ValueError("Pi event IDs must be non-empty and unique")  # noqa: TRY003
        if not isinstance(event_type, str) or not event_type.strip() or event_type == "session":
            raise ValueError("Pi events require an event type")  # noqa: TRY003
        if "parentId" not in record or (
            parent_id is not None and (not isinstance(parent_id, str) or parent_id not in records)
        ):
            raise ValueError("Pi parentId must be null or reference an earlier node")  # noqa: TRY003
        _record_digest(record)
        # Validate visible formats even on unselected branches before any writes.
        _event_body(record)
        records[event_id] = record
    leaf_id = value.leaf_id if value.leaf_id is not None else next(reversed(records), None)
    if leaf_id is not None and leaf_id not in records:
        raise ValueError("Pi selected leaf does not exist")  # noqa: TRY003
    chain: list[dict[str, JsonValue]] = []
    while leaf_id is not None:
        record = records[leaf_id]
        chain.append(record)
        leaf_id = record["parentId"]
    events: list[PiEventInput] = []
    for record in reversed(chain):
        body = _event_body(record)
        if body is not None:
            role, text = body
            events.append(
                PiEventInput(
                    session_id=session_id,
                    event_id=record["id"],
                    parent_id=record["parentId"],
                    event_type=record["type"],
                    role=role,
                    content=f"{role}: {text}",
                    record=record,
                )
            )
    return _Archive(session_id, records, tuple(events))


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse a complete parent graph and retain every visible selected ancestor."""

    return _parse_archive(value).events


class _PiEventAdapter:
    input_class = PiEventInput
    name = "pi-event"
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(name=_event_identity(value.session_id, value.event_id), **value.model_dump())

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput.model_validate(source.model_dump())


class _PiTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    _PiEventAdapter(), projections=(_PiTextProjection(),)
)


class _PiCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["pi-session-history/1"] = "pi-session-history/1"
    imported: dict[str, dict[str, str]] = Field(default_factory=dict)


class PiSessionConnector:
    """Resume branch capture using only Lifecycle-owned durable checkpoints."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        archive = _parse_archive(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        previous = _PiCheckpoint() if session.checkpoint is None else _PiCheckpoint.model_validate(session.checkpoint)
        imported = previous.imported.get(archive.session_id, {})
        for event_id, digest in imported.items():
            if event_id not in archive.records or _record_digest(archive.records[event_id]) != digest:
                raise ValueError("A previously captured Pi event was changed or removed")  # noqa: TRY003
        next_imported = {key: dict(events) for key, events in previous.imported.items()}
        captured = dict(imported)
        for event in archive.events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(_event_identity(event.session_id, event.event_id), "pi-event", event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE, checkpoint=session.checkpoint)
            captured[event.event_id] = _record_digest(event.record)
        next_imported[archive.session_id] = captured
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_PiCheckpoint(imported=next_imported).model_dump(mode="json"),
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
