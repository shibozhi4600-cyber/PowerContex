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

"""Document routing preserves complete companies and unambiguous fiscal periods."""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from collections.abc import Sequence

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import (
    BuiltinConfig,
    BuiltinRuntime,
    GetMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.pdf_pages import PDF_PAGE_SOURCE_DEFINITION, PDFPageInput, search_pdf_pages
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


class _EvidencePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(
                kind="source-evidence",
                text=TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content,
                sources=(source,),
            )
            for source in request.sources
            if isinstance(source, SourceObservation)
        )


async def _capture_pages(
    runtime: BuiltinRuntime, name: str, documents: Sequence[tuple[str, int, str]]
) -> tuple[str, dict[str, PDFPageInput]]:
    assert runtime.scopes is not None
    scope = await runtime.scopes.create(ScopeDraft(title=name, summary="Exact document periods", idempotency_key=name))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
    pages = {}
    for document_id, count, body in documents:
        for number in range(1, count + 1):
            page = PDFPageInput(
                document_id=document_id,
                document_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
                page_number=number,
                text=body + f"\nUnique record {number}.",
            )
            source = await registry.resolve(page)
            pages[source.name] = page
            await runtime.ingestion.submit(
                SubmitSourceObservation(
                    scope_id=scope.scope_id,
                    observation=project_source_for_transport(registry, source),
                )
            )
    memory = runtime.memory.for_scope(scope.scope_id)
    while (await memory.flush()).processed:
        pass
    assert len((await memory.list()).entries) == len(pages)
    return scope.scope_id, pages


async def _documents(
    runtime: BuiltinRuntime, scope_id: str, pages: dict[str, PDFPageInput], question: str, limit: int = 200
) -> list[str]:
    memory = runtime.memory.for_scope(scope_id)
    citations = await search_pdf_pages(memory, question, limit=limit)
    locations = []
    sources = set()
    for citation in citations:
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        assert len(entry.sources) == 1
        reference = entry.sources[0]
        assert reference.source_id not in sources
        sources.add(reference.source_id)
        page = pages[reference.source_id]
        assert entry.text == f"{page.document_id}\n{page.text}".strip()
        locations.append(page.document_id)
    assert len(citations) <= limit
    return locations


def test_all_explicit_whole_company_names_survive_unequal_word_counts() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_EvidencePipeline()) as runtime:
            scope_id, pages = await _capture_pages(
                runtime,
                "Multiple whole identities",
                (
                    ("ACME_2022_10K", 24, "Consolidated Statement of Cash Flows\nOperating cash flows 300."),
                    ("CEDAR_HARBOR_2022_10K", 24, "Consolidated Statement of Cash Flows\nOperating cash flows 300."),
                ),
            )
            question = "Compare Acme and Cedar Harbor operating cash flows in 2022."
            assert Counter(await _documents(runtime, scope_id, pages, question)) == {
                "ACME_2022_10K": 24,
                "CEDAR_HARBOR_2022_10K": 24,
            }
            for company, document in (("Acme", "ACME_2022_10K"), ("Cedar Harbor", "CEDAR_HARBOR_2022_10K")):
                returned = await _documents(
                    runtime, scope_id, pages, f"What are {company} operating cash flows in 2022?"
                )
                assert len(returned) == 24 and set(returned) == {document}

    asyncio.run(scenario())


def test_short_fiscal_years_route_like_observed_four_digit_years_without_amount_confusion() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_EvidencePipeline()) as runtime:
            scope_id, pages = await _capture_pages(
                runtime,
                "Short fiscal years",
                (
                    (
                        "ORION_2021_10K",
                        30,
                        "Consolidated Statement of Cash Flows\nCash flows amount 22. Cash paid 22. Operating cash 22.",
                    ),
                    (
                        "ORION_2022_10K",
                        30,
                        "Consolidated Statement of Cash Flows\nCash flows amount 21. Cash paid 21. Operating cash 21.",
                    ),
                ),
            )
            for year, expected in (
                ("FY22", "ORION_2022_10K"),
                ("FY 22", "ORION_2022_10K"),
                ("FY'21", "ORION_2021_10K"),
                ("FY21", "ORION_2021_10K"),
                ("2022", "ORION_2022_10K"),
                ("2021", "ORION_2021_10K"),
            ):
                question = f"What are Orion operating cash flows in {year}?"
                top = await _documents(runtime, scope_id, pages, question, limit=20)
                assert len(top) == 20 and set(top) == {expected}
                all_candidates = await _documents(runtime, scope_id, pages, question)
                assert len(all_candidates) == 30 and set(all_candidates) == {expected}
            both = await _documents(runtime, scope_id, pages, "Compare Orion operating cash flows in 2022 and 2021.")
            assert Counter(both) == {"ORION_2021_10K": 30, "ORION_2022_10K": 30}

    asyncio.run(scenario())


def test_partial_filing_year_coverage_does_not_hide_a_report_with_the_missing_period() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_EvidencePipeline()) as runtime:
            scope_id, pages = await _capture_pages(
                runtime,
                "Filing versus fiscal periods",
                (
                    (
                        "CEDAR_2021_10K",
                        12,
                        "Consolidated Statement of Cash Flows\nFiscal year 2021 operating cash flows 100.",
                    ),
                    (
                        "CEDAR_2023_10K",
                        12,
                        "Consolidated Statement of Cash Flows\nFiscal years 2023 2022 2021\nOperating cash flows 300 200 100.",
                    ),
                ),
            )
            for question in (
                "What were Cedar operating cash flows in 2022 and 2021?",
                "What were Cedar operating cash flows in FY22?",
                "Compare Cedar operating cash flows in 2021 and FY99.",
            ):
                assert Counter(await _documents(runtime, scope_id, pages, question)) == {
                    "CEDAR_2021_10K": 12,
                    "CEDAR_2023_10K": 12,
                }

    asyncio.run(scenario())


def test_two_digit_year_does_not_guess_a_century_when_candidate_years_are_ambiguous() -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_EvidencePipeline()) as runtime:
            scope_id, pages = await _capture_pages(
                runtime,
                "Ambiguous century",
                (
                    ("ORION_1922_10K", 12, "Operating cash flows were 30."),
                    ("ORION_2022_10K", 12, "Operating cash flows were 40."),
                ),
            )
            assert Counter(
                await _documents(runtime, scope_id, pages, "What are Orion operating cash flows in FY22?")
            ) == {
                "ORION_1922_10K": 12,
                "ORION_2022_10K": 12,
            }
            full = await _documents(runtime, scope_id, pages, "What are Orion operating cash flows in FY2022?")
            assert len(full) == 12 and set(full) == {"ORION_2022_10K"}

    asyncio.run(scenario())
