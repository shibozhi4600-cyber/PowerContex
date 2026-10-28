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

"""Financial retrieval keeps metric abbreviations distinct from company names."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import (
    BuiltinConfig,
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
from powercontext.sources._pdf_retrieval import rank_pdf_candidates
from powercontext.sources.pdf_pages import PDF_PAGE_SOURCE_DEFINITION, PDFPageInput, search_pdf_pages
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

_BALANCE = (
    "REPORT\nConsolidated Balance Sheets\nCurrent assets 400\nCurrent liabilities 150\n"
    "Total assets 900\nTotal liabilities 350\nStockholders equity 550"
)
_INCOME = (
    "REPORT\nConsolidated Statements of Income\nNet sales 900\nCost of sales 400\nOperating income 300\nNet income 240"
)
_CASH = (
    "REPORT\nConsolidated Statements of Cash Flows\nOperating activities 300\n"
    "Investing activities 100\nFinancing activities 80"
)


def _commentary(count: int = 24) -> tuple[str, ...]:
    return tuple(f"REPORT\nOperating discussion, observation {index}." for index in range(count))


def test_capital_intensity_retrieves_three_independent_financial_statements() -> None:
    texts = (*_commentary(), _BALANCE, _INCOME, _CASH)
    order = tuple(range(len(texts)))
    ranked = rank_pdf_candidates(texts, order, "Is this a capital-intensive business?", 20)
    assert {24, 25, 26} <= set(ranked[:20])
    assert ranked[:17] == order[:17]
    assert len(ranked) == len(set(ranked)) == len(texts)


@pytest.mark.parametrize(
    "question",
    [
        "Calculate inventory turnover for this business.",
        "What is the return on assets?",
        "How do we calculate ROA?",
    ],
)
def test_formulas_needing_assets_and_earnings_retrieve_two_independent_bodies(question: str) -> None:
    texts = (*_commentary(), _BALANCE, _INCOME)
    ranked = rank_pdf_candidates(texts, tuple(range(len(texts))), question, 20)
    assert {24, 25} <= set(ranked[:20])
    assert len(ranked) == len(set(ranked)) == len(texts)


def test_primary_statement_rows_outrank_a_note_column_and_parent_company_schedule() -> None:
    note_column = (
        "REPORT\nNote 7 Financial Instruments\nGains recognized in\nIncome Statement\n"
        "Net sales gains 10\nNet income effects 4"
    )
    parent_schedule = (
        "REPORT\nSCHEDULE I\nFinancial Information of Parent Company\nConsolidated Statements of Income\n"
        "Net sales 500\nCost of sales 300\nNet income 100"
    )
    texts = (*_commentary(), note_column, parent_schedule, _INCOME)
    ranked = rank_pdf_candidates(texts, tuple(range(len(texts))), "Calculate gross margin.", 20)
    assert 26 in ranked[:20]
    assert 24 not in ranked[:20] and 25 not in ranked[:20]


def test_fragmented_contents_and_comprehensive_income_are_not_primary_profit_statements() -> None:
    false_titles = (
        "REPORT\nTable\nOf\nContents\nConsolidated Statements of Income\n57\nAudit opinion\n56",
        "REPORT\nConsolidated Statements of Comprehensive Income\nOther comprehensive income 300",
    )
    texts = (*_commentary(), *false_titles)
    order = tuple(range(len(texts)))
    assert rank_pdf_candidates(texts, order, "Calculate gross margin.", 20) == order


def test_complementary_period_table_is_added_when_one_income_statement_is_already_present() -> None:
    texts = list(_commentary())
    texts[0] = _INCOME.replace("REPORT", "REPORT_2024")
    texts.append(_INCOME.replace("REPORT", "REPORT_2023"))
    ranked = rank_pdf_candidates(texts, tuple(range(len(texts))), "Compare gross margin across the two periods.", 20)
    assert 0 in ranked[:20] and 24 in ranked[:20]
    assert len(ranked) == len(set(ranked)) == len(texts)


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 20])
def test_formula_diversification_preserves_unique_original_candidates_at_small_limits(limit: int) -> None:
    texts = (*_commentary(), _BALANCE, _INCOME, _CASH)
    order = tuple(range(len(texts)))
    ranked = rank_pdf_candidates(texts, order, "Assess capital intensity.", limit)
    assert len(ranked) == len(set(ranked)) == len(order)
    assert set(ranked) == set(order)
    assert len(ranked[:limit]) == limit


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


def test_compound_metrics_do_not_hard_route_real_memory_but_independent_company_alias_still_does() -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_EvidencePipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(
                    title="Compound accounting metrics",
                    summary="Independent companies",
                    idempotency_key="compound-metrics",
                )
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            originals = {}
            for company, count in (("IRON", 24), ("COPPER", 2)):
                document_id = f"{company}_2024_10K"
                for number in range(1, count + 1):
                    text = f"{company} SG&A PP&E R&D M&A P&L accounting results, record {number}. " + (
                        "NOVA operations and assets." if company == "IRON" else "Independent operating results."
                    )
                    source = await registry.resolve(
                        PDFPageInput(
                            document_id=document_id,
                            document_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
                            page_number=number,
                            text=text,
                        )
                    )
                    observation = project_source_for_transport(registry, source)
                    originals[source.name] = TextEvidence.model_validate(
                        observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)
                    )
                    await runtime.ingestion.submit(
                        SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                    )
            memory = runtime.memory.for_scope(scope.scope_id)
            while (await memory.flush()).processed:
                pass
            assert len((await memory.list()).entries) == 26

            async def documents(question: str) -> list[str]:
                citations = await search_pdf_pages(memory, question, limit=100)
                returned = []
                sources = set()
                for citation in citations:
                    entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
                    assert len(entry.sources) == 1
                    source_id = entry.sources[0].source_id
                    assert source_id not in sources
                    sources.add(source_id)
                    evidence = originals[source_id]
                    assert entry.text == evidence.content.strip()
                    document = evidence.metadata["document_id"]
                    assert isinstance(document, str)
                    returned.append(document)
                return returned

            for metric in ("SG&A", "PP&E", "R&D", "M&A", "P&L"):
                found = await documents(f"How did {metric} affect the change in net sales?")
                assert len(found) == 26
                assert set(found) == {"IRON_2024_10K", "COPPER_2024_10K"}
            named = await documents("Does NOVA report SG&A in its balance sheet?")
            assert len(named) == 24 and set(named) == {"IRON_2024_10K"}

    asyncio.run(scenario())
