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

"""Literal evidence selection protects addresses, offsets and full-envelope limits."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import ValidationError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.runtime import ContextAssembly, PrepareContextRequest
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.prepared_context import PreparedContextBuilder, PreparedMemoryCandidates
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY


def _group(
    text: str, *, scope: str = "project:one", revision: int = 4, entry: str = "entry"
) -> PreparedMemoryCandidates:
    ref = ArtifactRef(family="memory", artifact_id="memory", revision=revision)
    return PreparedMemoryCandidates(
        scope,
        ref,
        (
            MemoryHit(
                memory_ref=ref,
                entry_id=entry,
                entry_version_id=entry + "-v3",
                text=text,
                score=1.0,
                matched_by=("fts",),
            ),
        ),
    )


def _build(query: str, groups: tuple[PreparedMemoryCandidates, ...], *, budget: int = 1400):
    return PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query=query, max_bytes=budget, evidence_selection=True),
        current_scope_id="project:one",
        memory_candidates=groups,
    )


@pytest.mark.parametrize("value", [1, "true", None, 0.0])
def test_evidence_selection_requires_a_real_boolean(value: object) -> None:
    with pytest.raises(ValidationError):
        PrepareContextRequest(query="text", evidence_selection=cast(bool, value))


def test_complementary_memory_records_can_both_supply_query_evidence() -> None:
    groups = (
        _group("The cobalt sensor pressure is 29 bars.", entry="pressure"),
        _group("The amber valve temperature is 53 degrees.", scope="project:other", entry="temperature"),
    )
    build = _build("cobalt pressure amber temperature", groups, budget=1400)
    assert {fragment.address.entry_id for fragment in build.fragments} == {"pressure", "temperature"}
    assert {fragment.address.memory.scope_id for fragment in build.fragments} == {"project:one", "project:other"}
    assert build.context.content_bytes <= 1400


def test_original_unicode_escaping_and_exact_identity_survive_the_envelope() -> None:
    text = '  无关背景。\r\nSensor “海洋😀” reports "37°C" at station\\north.\nEnding.'
    groups = (_group(text, scope="project:foreign", revision=9),)
    assert groups[0].memory_ref is not None
    build = _build("海洋 station sensor", groups)
    assert build.fragments
    assert build.context.content.startswith(TRUST_POLICY)
    assert build.context.content.endswith("END_POWERCONTEXT_PREPARED_CONTEXT_V1")
    envelope = json.loads(build.context.content.splitlines()[-2])
    assert envelope["trust"] == "untrusted_history"
    assert len(build.fragments) == len(build.origins) == len(envelope["items"])
    for fragment, origin, item in zip(build.fragments, build.origins, envelope["items"], strict=True):
        assert fragment.address == origin
        assert fragment.address.memory.scope_id == "project:foreign"
        assert fragment.address.memory.artifact.revision == 9
        assert fragment.address.entry_id == "entry"
        assert fragment.address.entry_version_id == "entry-v3"
        assert type(fragment.start) is int and type(fragment.end) is int
        assert text[fragment.start : fragment.end] == fragment.text == item["content"]
        assert item["start"] == fragment.start and item["end"] == fragment.end
        assert item["citation"]["memory_ref"] == groups[0].memory_ref.model_dump(mode="json")
    assert build.context.content_bytes == len(build.context.content.encode()) <= 1400


def test_query_selects_late_evidence_inside_an_unpunctuated_entry() -> None:
    text = "unrelated background " * 200 + "zirconium sensor calibrated at 53 degrees " + "other material " * 100
    build = _build("zirconium sensor calibrated", (_group(text),), budget=1200)
    assert any("zirconium sensor calibrated at 53" in fragment.text for fragment in build.fragments)
    assert all(fragment.text == text[fragment.start : fragment.end] for fragment in build.fragments)


def test_changing_query_changes_selection_without_rewriting_prose() -> None:
    prefix = "Routine background information. " * 100
    groups = (
        _group(prefix + "Violet turbine output is 71 megawatts.", entry="turbine"),
        _group(prefix + "Silver reactor pressure is 29 bars.", entry="reactor"),
    )
    first = _build("violet turbine output", groups, budget=1000)
    second = _build("silver reactor pressure", groups, budget=1000)
    assert any("71 megawatts" in fragment.text for fragment in first.fragments)
    assert any("29 bars" in fragment.text for fragment in second.fragments)
    assert first.fragments != second.fragments


@pytest.mark.parametrize("budget", [512, 600, 800, 1400, 1800, 32768])
def test_full_envelope_budget_and_probe_agree_even_when_no_fragment_fits(budget: int) -> None:
    groups = (_group('海洋 "quoted" \\ exact words. ' * 45),)
    request = PrepareContextRequest(query="海洋 quoted", max_bytes=budget, evidence_selection=True)
    builder = PreparedContextBuilder()
    build = builder.build_scopes_result(request=request, current_scope_id="project:one", memory_candidates=groups)
    measured = len((build.context.content or "").encode())
    assert build.context.content_bytes == measured <= budget
    assert bool(build.fragments) == bool(build.context.content)
    probe = builder.probe_budget(request=request, current_scope_id="project:one", memory_candidates=groups)
    assert probe.delivered_items == len(build.fragments)
    assert probe.unused_bytes == budget - measured


def test_exact_versions_are_deduplicated_but_cross_scope_identities_are_not_confused() -> None:
    first = _group("The cobalt device is stable.")
    same_id_foreign = _group("The amber device is unstable.", scope="project:foreign")
    build = _build("amber unstable", (first, first, same_id_foreign))
    assert build.fragments[0].address.memory.scope_id == "project:foreign"
    identities = [(fragment.address.memory.scope_id, fragment.start, fragment.end) for fragment in build.fragments]
    assert len(identities) == len(set(identities))


def test_selection_preserves_existing_candidate_reference_invariant() -> None:
    group = _group("Some literal evidence.")
    mismatch = ArtifactRef(family="memory", artifact_id="other", revision=1)
    with pytest.raises(PreparedContextInvariantError, match="memory-ref-mismatch"):
        _build("evidence", (PreparedMemoryCandidates(group.scope_id, mismatch, group.hits),))


def test_default_prepare_and_explicit_disabled_mode_are_byte_identical() -> None:
    group = _group("A complete ordinary record.")
    builder = PreparedContextBuilder()
    for assembly in (None, ContextAssembly()):
        request = PrepareContextRequest(query="ordinary")
        if assembly is not None:
            request = request.model_copy(update={"assembly": assembly})
        default = builder.build_scopes_result(
            request=request,
            current_scope_id=group.scope_id,
            memory_candidates=(group,),
        )
        disabled = builder.build_scopes_result(
            request=request.model_copy(update={"evidence_selection": False}),
            current_scope_id=group.scope_id,
            memory_candidates=(group,),
        )
        assert default == disabled
        assert not default.fragments
