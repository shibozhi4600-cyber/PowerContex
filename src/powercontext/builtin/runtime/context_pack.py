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

"""Self-contained, verified transport for prepared context and prefix deltas.

Packs contain the actual UTF-8 bytes, never cache handles. A receiver can rebuild
the complete target using only a decoded base and a decoded delta.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ContextPackError(ValueError):
    """A context pack, chunk, or delta violates the transport contract."""


def _json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContextPackError("duplicate-json-field")
        result[key] = value
    return result


def _decode(data: bytes, fields: set[str], schema: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise ContextPackError("wire-data-must-be-bytes")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise ContextPackError("invalid-context-pack-json") from error
    if not isinstance(value, dict) or set(value) != fields or value.get("schema") != schema:
        raise ContextPackError("invalid-schema-or-fields")
    return value


def _items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
        raise ContextPackError("chunks-must-be-a-tuple-of-bytes")
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeError as error:
        raise ContextPackError("each-chunk-must-contain-valid-utf-8") from error


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decoded(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("encoded-chunks-must-be-an-array")
    result: list[bytes] = []
    try:
        for item in value:
            if not isinstance(item, str):
                raise ContextPackError("encoded-chunks-must-be-strings")
            decoded = base64.b64decode(item.encode("ascii"), validate=True)
            if base64.b64encode(decoded).decode("ascii") != item:
                raise ContextPackError("noncanonical-base64")
            result.append(decoded)
    except (UnicodeError, binascii.Error, ValueError) as error:
        raise ContextPackError("invalid-base64-chunk") from error
    chunks = tuple(result)
    _items(chunks)
    return chunks


def _digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": _encoded(items)}
    return "sha256:" + hashlib.sha256(_json(unsigned)).hexdigest()


def _check_digest(value: object) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContextPackError("invalid-sha-256-digest")


def _pack_wire(items: tuple[bytes, ...], digest: str, byte_count: int) -> bytes:
    return _json({"schema": _PACK_SCHEMA, "items": _encoded(items), "digest": digest, "byte_count": byte_count})


@dataclass(frozen=True)
class ContextPack:
    """An immutable complete context with independently verifiable identity."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    def __post_init__(self) -> None:
        _items(self.items)
        _check_digest(self.digest)
        if self.digest != _digest(self.items):
            raise ContextPackError("pack-digest-mismatch")
        if type(self.byte_count) is not int or self.byte_count != len(self.to_bytes()):
            raise ContextPackError("pack-byte-count-mismatch")

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        _items(items)
        digest = _digest(items)
        count = 0
        while True:
            measured = len(_pack_wire(items, digest, count))
            if measured == count:
                return cls(items, digest, count)
            count = measured

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content-must-be-text")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk-limit-must-be-a-positive-integer")
        try:
            encoded = content.encode("utf-8")
        except UnicodeError as error:
            raise ContextPackError("content-must-contain-valid-unicode") from error
        chunks: list[bytes] = []
        start = 0
        while start < len(encoded):
            end = min(start + chunk_bytes, len(encoded))
            # A continuation byte cannot begin the next independently decoded chunk.
            while end < len(encoded) and end > start and encoded[end] & 0xC0 == 0x80:
                end -= 1
            if end == start:
                raise ContextPackError("chunk-limit-cannot-fit-the-next-utf-8-character")
            chunks.append(encoded[start:end])
            start = end
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _pack_wire(self.items, self.digest, self.byte_count)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        value = _decode(data, {"schema", "items", "digest", "byte_count"}, _PACK_SCHEMA)
        return cls(_decoded(value["items"]), value["digest"], value["byte_count"])


@dataclass(frozen=True)
class ContextPackDelta:
    """Reuse a verified base prefix, followed by explicitly transmitted chunks."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _check_digest(self.base_digest)
        _check_digest(self.target_digest)
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix-must-be-a-nonnegative-integer")
        _items(self.appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta-requires-complete-base-and-target-packs")
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        return _json({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        value = _decode(
            data,
            {"schema", "base_digest", "target_digest", "prefix_items", "appended"},
            _DELTA_SCHEMA,
        )
        return cls(value["base_digest"], value["target_digest"], value["prefix_items"], _decoded(value["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Reconstruct and verify the entire target without sender state or I/O."""
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("expected-a-complete-pack-and-delta")
    if base.digest != delta.base_digest:
        raise ContextPackError("delta-base-digest-mismatch")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta-prefix-exceeds-the-base")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta-target-digest-mismatch")
    return target
