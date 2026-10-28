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

"""Transport invariants are checked independently of the sender's objects."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import cast

import pytest

from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)


def _wire(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5, 7, 511, 512])
def test_multibyte_chunks_keep_every_original_byte(limit: int) -> None:
    content = "a海洋😀e\u0301\r\nβ" * 70
    if limit < 4:
        with pytest.raises(ContextPackError):
            ContextPack.from_content(content, chunk_bytes=limit)
        return
    pack = ContextPack.from_content(content, chunk_bytes=limit)
    assert all(len(chunk) <= limit and chunk.decode("utf-8") for chunk in pack.items)
    assert b"".join(pack.items) == content.encode()
    assert ContextPack.from_content(content, chunk_bytes=limit) == pack
    assert ContextPack.from_bytes(pack.to_bytes()) == pack
    payload = json.loads(pack.to_bytes())
    unsigned = {"schema": payload["schema"], "items": payload["items"]}
    assert payload["digest"] == "sha256:" + hashlib.sha256(_wire(unsigned)).hexdigest()
    assert payload["byte_count"] == len(pack.to_bytes())


@pytest.mark.parametrize("limit", [True, False, 1.0, -1, 0, None, "512"])
def test_chunk_limit_is_a_strict_positive_integer(limit: object) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("", chunk_bytes=cast(int, limit))


@pytest.mark.parametrize("length", [0, 1, 4, 50, 500, 5000, 50000])
def test_self_inclusive_byte_count_is_exact_across_decimal_boundaries(length: int) -> None:
    pack = ContextPack.from_content("x" * length)
    assert pack.byte_count == len(pack.to_bytes())
    assert ContextPack.from_bytes(pack.to_bytes()) == pack


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"schema": "other"},
        {"byte_count": True},
        {"byte_count": 1.5},
        {"byte_count": 0},
        {"items": ["@@=="]},
        {"items": ["YR=="]},
        {"items": ["/w=="]},
        {"items": [1]},
        {"items": "YQ=="},
        {"digest": "sha256:" + "F" * 64},
        {"digest": "sha256:" + "0" * 64},
    ],
)
def test_pack_rejects_invalid_fields_encodings_and_integrity(change: dict[str, object]) -> None:
    payload = json.loads(ContextPack.build((b"a",)).to_bytes())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire({**payload, **change}))


def test_duplicate_json_keys_are_not_silently_accepted() -> None:
    wire = ContextPack.build(()).to_bytes()
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(wire[:-1] + b',"items":[]}')
    delta = ContextPackDelta.between(ContextPack.build(()), ContextPack.build(())).to_bytes()
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(delta[:-1] + b',"prefix_items":0}')


def test_receiver_accepts_every_valid_prefix_without_shared_objects() -> None:
    base = ContextPack.build((b"first", "第二".encode(), b"old"))
    target = ContextPack.build((b"first", "第二".encode(), b"new"))
    template = json.loads(ContextPackDelta.between(base, target).to_bytes())
    for prefix in (0, 1, 2):
        payload = {
            **template,
            "prefix_items": prefix,
            "appended": [base64.b64encode(item).decode() for item in target.items[prefix:]],
        }
        rebuilt = apply_context_pack_delta(
            ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(_wire(payload))
        )
        assert rebuilt.to_bytes() == target.to_bytes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefix_items", True),
        ("prefix_items", 1.0),
        ("prefix_items", -1),
        ("appended", ["/w=="]),
        ("base_digest", 1),
        ("target_digest", None),
        ("extra", "unexpected"),
    ],
)
def test_delta_decoder_rejects_invalid_wire(field: str, value: object) -> None:
    base = ContextPack.build((b"base",))
    payload = json.loads(ContextPackDelta.between(base, ContextPack.build((b"target",))).to_bytes())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(_wire({**payload, field: value}))


def test_stale_base_out_of_bounds_and_corrupt_target_fail_closed() -> None:
    base = ContextPack.build((b"same",))
    target = ContextPack.build((b"same", b"more"))
    delta = ContextPackDelta.between(base, target)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(ContextPack.build((b"different revision",)), delta)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, target.digest, 2, ()))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, target.digest, 1, (b"tampered",)))


def test_empty_and_unchanged_targets_are_independently_reconstructible() -> None:
    base = ContextPack.from_content("A body that the receiver already has.")
    for target in (base, ContextPack.from_content("")):
        delta = ContextPackDelta.from_bytes(ContextPackDelta.between(base, target).to_bytes())
        rebuilt = apply_context_pack_delta(ContextPack.from_bytes(base.to_bytes()), delta)
        assert rebuilt == target
        assert delta.byte_count == len(delta.to_bytes())
