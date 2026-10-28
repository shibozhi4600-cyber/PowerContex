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

"""Complete document identities can be spaced differently from natural queries."""

from __future__ import annotations

import asyncio
import hashlib

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


def test_spaced_compact_and_internal_stop_word_identities_match_whole_words() -> None:
    async def scenario() -> None:
        labels = (
            "CEDARHARBOR_2024_10K",
            "BANKOFMOON_2024_10K",
            "STAR_OF_SEA_2024_10K",
            "BASALT_2024_10K",
            "BANK_OF_MOON_RIVER_2024_10K",
        )
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_EvidencePipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(
                    title="Document identities", summary="Exact phrase matches", idempotency_key="pdf-identities"
                )
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            for label in labels:
                source = await registry.resolve(
                    PDFPageInput(
                        document_id=label,
                        document_sha256=hashlib.sha256(label.encode()).hexdigest(),
                        page_number=1,
                        text="Operating cash flows were 100 in 2024.",
                    )
                )
                await runtime.ingestion.submit(
                    SubmitSourceObservation(
                        scope_id=scope.scope_id,
                        observation=project_source_for_transport(registry, source),
                    )
                )
            memory = runtime.memory.for_scope(scope.scope_id)
            while (await memory.flush()).processed:
                pass

            async def found(question: str) -> set[str]:
                citations = await search_pdf_pages(memory, question, limit=20)
                return {
                    (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.text.partition("\n")[0]
                    for citation in citations
                }

            for apostrophe in ("'", chr(0x2019)):
                assert await found(f"What are Cedar Harbor{apostrophe}s operating cash flows in 2024?") == {labels[0]}
                assert await found(f"What are Bank of Moon{apostrophe}s operating cash flows in 2024?") == {labels[1]}
            assert await found("What are StarofSea's operating cash flows in 2024?") == {labels[2]}
            assert await found("What are Basalt's operating cash flows in 2024?") == {labels[3]}
            for question in (
                "What are operating cash flows in fiscal 2024?",
                "What are our operating cash flows in the annual PDF report for 2024?",
                "2024",
                "What were Cedar observed Harbor operating cash flows?",
                "What were Cedar 10-K Harbor operating cash flows?",
                "What were Cedar of Harbor operating cash flows?",
                "What were Cedar Harborfield operating cash flows?",
                "What were CedarHarboring operating cash flows?",
            ):
                assert await found(question) == set(labels)

    asyncio.run(scenario())
