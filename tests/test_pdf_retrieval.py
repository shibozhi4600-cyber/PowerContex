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

"""Financial evidence diversity preserves document intent and exact stored pages."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
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
from powercontext.sources._pdf_retrieval import rank_pdf_candidates
from powercontext.sources.pdf_pages import PDF_PAGE_SOURCE_DEFINITION, PDFPageInput, search_pdf_pages
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _two_company_texts() -> tuple[str, ...]:
    return tuple(
        f"{company}_2024_10K\n{company} operating cash flows, page {page}. "
        + ("NOVA CEO CFO governance." if company == "KAPPA" else "ORBIT chief executive officer governance.")
        for company in ("KAPPA", "LAMBDA")
        for page in range(1, 6)
    )


@pytest.mark.parametrize(
    "question",
    [
        "Summarize CEO responsibilities.",
        "Explain the audit methodology.",
        "What does NOVA mean in this report?",
    ],
)
def test_nonfinancial_questions_keep_the_original_pool_and_order(question: str) -> None:
    texts = _two_company_texts()
    baseline = (4, 8, 0, 6, 2, 9, 1, 5, 3, 7)
    assert rank_pdf_candidates(texts, baseline, question, 20, allow_acronym_routing=True) == baseline


@pytest.mark.parametrize(
    "question",
    [
        "Compare KAPPA and LAMBDA operating cash flows.",
        "Compare NOVA and ORBIT gross margins.",
        "How does the CEO explain operating cash flows?",
        "How does the CFO explain gross margin?",
    ],
)
def test_multiple_entities_and_officer_titles_cannot_force_one_company(question: str) -> None:
    texts = _two_company_texts()
    baseline = tuple(range(len(texts)))
    ranked = rank_pdf_candidates(texts, baseline, question, 20, allow_acronym_routing=True)
    assert ranked == baseline


def test_explicit_identity_controls_routing_and_unresolved_financial_alias_has_support() -> None:
    texts = _two_company_texts()
    baseline = tuple(range(len(texts)))
    question = "Does NOVA have an improving gross margin?"
    assert rank_pdf_candidates(texts, baseline, question, 20, allow_acronym_routing=False) == baseline
    routed = rank_pdf_candidates(texts, baseline, question, 20, allow_acronym_routing=True)
    assert set(routed) == set(range(5))
    assert all(texts[index].startswith("KAPPA_2024_10K\n") for index in routed)


@pytest.mark.parametrize(
    "question, expected_titles",
    [
        ("How do we calculate the quick ratio?", ("Consolidated Balance Sheets",)),
        ("Compute gross margin from reported revenue.", ("Consolidated Statements of Income",)),
        (
            "Compute EBITDA using operating income and cash flows.",
            (
                "Consolidated Statements of Income",
                "Consolidated Statements of Cash Flows",
            ),
        ),
    ],
)
def test_financial_formulas_add_complementary_structured_evidence(
    question: str, expected_titles: tuple[str, ...]
) -> None:
    titles = (
        "Consolidated Balance Sheets",
        "Consolidated Statements of Income",
        "Consolidated Statements of Cash Flows",
    )
    texts = tuple(f"REPORT\nOperating commentary {index}." for index in range(24)) + tuple(
        f"REPORT\n{title}\nYears ended 2024 and 2023\nCurrent assets 400 300\nRevenue 200 180" for title in titles
    )
    baseline = tuple(range(len(texts)))
    ranked = rank_pdf_candidates(texts, baseline, question, 20)
    selected = [texts[index] for index in ranked[:20]]
    assert all(any(title in text for text in selected) for title in expected_titles)
    # Diversification adds at most two pages without displacing strong lexical evidence.
    assert ranked[:18] == baseline[:18]
    assert len(ranked) == len(baseline) and set(ranked) == set(baseline)


def test_contents_rows_comprehensive_income_and_prose_do_not_replace_an_income_statement() -> None:
    commentary = tuple(f"REPORT\nOrdinary operating commentary {index}." for index in range(24))
    misleading = (
        "REPORT\nTable of Contents\nConsolidated Statement of Income\n57\nAudit report\n56",
        "REPORT\nConsolidated Statement of Comprehensive Income\nOther comprehensive income 400",
        "REPORT\nNote 3\nThese consolidated statements of income include segment information.",
    )
    actual = "REPORT\nTable of Contents\nConsolidated Statement of Income\nYears ended 2024 and 2023\nRevenue 400 300"
    texts = commentary + misleading + (actual,)
    baseline = tuple(range(len(texts)))
    ranked = rank_pdf_candidates(texts, baseline, "Calculate gross margin using the income statement.", 20)
    assert len(texts) - 1 in ranked[:20]
    assert not set(range(24, 27)) & set(ranked[:20])


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 20])
def test_one_page_supporting_two_formulas_keeps_unique_original_indexes(limit: int) -> None:
    texts = (
        *(f"REPORT\nOperating commentary {index}." for index in range(24)),
        "REPORT\nConsolidated Statements of Income\nRevenue 400\nConsolidated Statements of Cash Flows\nCash 200",
    )
    baseline = tuple(range(len(texts)))
    ranked = rank_pdf_candidates(texts, baseline, "Compute EBITDA from income and cash flows.", limit)
    assert len(texts) - 1 in ranked[:limit]
    assert len(ranked) == len(set(ranked)) == len(texts)
    assert set(ranked) == set(baseline)


class _EvidencePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(
                kind=kind,
                text=TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content,
                sources=(source,),
            )
            for source in request.sources
            if isinstance(source, SourceObservation)
            for kind in ("source-evidence", "audit-copy")
        )


def test_runtime_preserves_entities_distinct_sources_limits_and_exact_bodies_after_restart(tmp_path: Path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pdf-retrieval.db'}"))

    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        observations = []
        async with open_builtin_runtime(config, candidate_pipeline=_EvidencePipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Financial pages", summary="Independent entities", idempotency_key="pdf-helper")
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            for company in ("KAPPA", "LAMBDA"):
                document_id = company + "_2024_10K"
                for number in range(1, 6):
                    source = await registry.resolve(
                        PDFPageInput(
                            document_id=document_id,
                            document_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
                            page_number=number,
                            text=f"{company} operating cash flows on page {number}. "
                            + (
                                "NOVA CEO CFO governance."
                                if company == "KAPPA"
                                else "ORBIT chief executive officer governance."
                            ),
                        )
                    )
                    observation = project_source_for_transport(registry, source)
                    observations.append(observation)
                    await runtime.ingestion.submit(
                        SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                    )
            memory = runtime.memory.for_scope(scope.scope_id)
            while (await memory.flush()).processed:
                pass
            # The extraction pipeline emits two eligible entries for each Source;
            # result limits must still count distinct pages rather than entries.
            assert len((await memory.list()).entries) > len(observations)

        async with open_builtin_contexts(config) as contexts:
            journal = await (await contexts.get(scope.scope_id)).sources.journal.entries()
            stored = {entry.source_ref.source_id: entry.source for entry in journal}
            assert len(stored) == len(observations)
        async with open_builtin_runtime(config) as runtime:
            memory = runtime.memory.for_scope(scope.scope_id)
            for question in (
                "Compare KAPPA and LAMBDA operating cash flows.",
                "Compare NOVA and ORBIT gross margins.",
                "Summarize CEO responsibilities.",
                "How does the CFO explain operating cash flows?",
            ):
                citations = await search_pdf_pages(memory, question, limit=20)
                refs = []
                companies = set()
                for citation in citations:
                    entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
                    assert len(entry.sources) == 1
                    source = stored[entry.sources[0].source_id]
                    assert isinstance(source, SourceObservation)
                    evidence = TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                    assert entry.text == evidence.content.strip()
                    companies.add(evidence.metadata["document_id"])
                    refs.append(entry.sources[0].source_id)
                assert len(refs) == len(set(refs)) == 10
                assert companies == {"KAPPA_2024_10K", "LAMBDA_2024_10K"}
            for limit in (1, 2, 3):
                citations = await search_pdf_pages(memory, "Compute EBITDA from cash flows.", limit=limit)
                assert len(citations) == limit
                page_refs = []
                for citation in citations:
                    entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
                    page_refs.append(entry.sources[0].source_id)
                    source = stored[entry.sources[0].source_id]
                    assert isinstance(source, SourceObservation)
                    assert (
                        entry.text
                        == TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content.strip()
                    )
                assert len(page_refs) == len(set(page_refs))

    asyncio.run(scenario())
