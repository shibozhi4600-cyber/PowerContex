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

import hashlib
import io
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.models import MemoryEntriesPage, MemorySearchPage
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _pdf(texts: tuple[str, ...], *, password: str | None = None) -> bytes:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({
                    NameObject("/F1"): DictionaryObject({
                        NameObject("/Type"): NameObject("/Font"),
                        NameObject("/Subtype"): NameObject("/Type1"),
                        NameObject("/BaseFont"): NameObject("/Helvetica"),
                    })
                })
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("ascii")
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET")
            page[NameObject("/Contents")] = stream
    if password is not None:
        writer.encrypt(user_password=password, owner_password="owner-only")  # noqa: S106 - Generated encryption fixture.
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


async def test_every_page_including_last_and_blank_is_captured_with_exact_revision() -> None:
    raw = _pdf(("First page revenue 001", "", "Third page liabilities 900", "", "Last page appendix 123"))
    pages = parse_pdf_document(PDFDocumentInput(document_id="Example / report", content=raw))
    assert [page.page_number for page in pages] == [1, 2, 3, 4, 5]
    assert [page.text for page in pages] == [
        "First page revenue 001",
        "",
        "Third page liabilities 900",
        "",
        "Last page appendix 123",
    ]
    assert all(page.document_sha256 == hashlib.sha256(raw).hexdigest() for page in pages)
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    names = set()
    for page in pages:
        source = await registry.resolve(page)
        names.add(source.name)
        assert len(source.name) <= 256
        assert await registry.read(source) == page
        projection = TextEvidence.model_validate(
            project_source_for_transport(registry, source).projection(TEXT_EVIDENCE_PROJECTION_KEY)
        )
        assert projection.content == f"{page.document_id}\n{page.text}"
        assert projection.metadata == {
            "document_id": page.document_id,
            "document_sha256": page.document_sha256,
            "page_number": page.page_number,
        }
    assert len(names) == len(pages)
    other = pages[0].model_copy(update={"document_id": "Another document"})
    assert (await registry.resolve(other)).name not in names


def test_empty_password_is_valid_and_real_password_is_not_silently_ignored() -> None:
    assert parse_pdf_document(PDFDocumentInput(document_id="open", content=_pdf(("",), password="")))[0].text == ""
    with pytest.raises(ValueError):
        parse_pdf_document(
            PDFDocumentInput(document_id="closed", content=_pdf(("secret",), password="required"))  # noqa: S106
        )


@pytest.mark.parametrize("page", [0, -1, True, 1.0])
def test_physical_page_number_must_be_a_positive_integer(page) -> None:
    with pytest.raises(ValueError):
        PDFPageInput(document_id="report", document_sha256="f" * 64, page_number=page, text="")


@pytest.mark.parametrize("digest", ["", "f" * 63, "F" * 64, "x" * 64])
def test_page_revision_digest_is_exact_lowercase_sha256(digest: str) -> None:
    with pytest.raises(ValueError):
        PDFPageInput(document_id="report", document_sha256=digest, page_number=1, text="")


class _PagePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        entries = []
        for source in request.sources:
            assert isinstance(source, SourceObservation)
            projection = source.projection(TEXT_EVIDENCE_PROJECTION_KEY)
            assert isinstance(projection, dict)
            text = projection["content"]
            assert isinstance(text, str)
            # Both entries cite the same physical page; retrieval must dedupe it.
            for kind in ("page", "page-copy"):
                entries.append(MemoryEntryInput(kind=kind, text=text, sources=(source,)))
        return tuple(entries)


async def test_retrieval_uses_current_scoped_records_distinct_pages_and_gettable_citations(tmp_path: Path) -> None:
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pdf-search.db'}"))
    sources = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    async with open_builtin_runtime(config) as runtime:
        assert runtime.scopes is not None
        scopes = [
            await runtime.scopes.create(ScopeDraft(title=name, summary=name, idempotency_key=name))
            for name in ("reports", "other", "empty")
        ]
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        observations = []
        for document, page_number, text in (
            ("ALPHA_2024_10K", 1, "Legal proceedings describe litigation and contingencies."),
            ("ALPHA_2024_10K", 2, "Regional operations are in countries across several markets."),
            ("BETA_2024_10K", 1, "Legal litigation legal proceedings litigation legal."),
        ):
            source = await sources.resolve(
                PDFPageInput(document_id=document, document_sha256="1" * 64, page_number=page_number, text=text)
            )
            observation = project_source_for_transport(sources, source)
            await runtime.ingestion.submit(
                SubmitSourceObservation(scope_id=scopes[0].scope_id, observation=observation)
            )
            observations.append(observation)
        private_source = observations[0]
        await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scopes[1].scope_id, observation=private_source))
    async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
        for scope in scopes[:2]:
            while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
                pass
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scopes[0].scope_id)
        citations = await search_pdf_pages(memory, "Does Alpha report legal battles in 2024?", limit=2)
        assert 1 <= len(citations) <= 2
        returned = [(await memory.get(GetMemoryEntryRequest(citation=citation))).entry for citation in citations]
        assert returned[0].text.startswith("ALPHA_2024_10K\nLegal proceedings")
        assert all(entry.text.startswith("ALPHA_2024_10K\n") for entry in returned)
        assert len({entry.sources[0].source_id for entry in returned}) == len(citations)
        assert await search_pdf_pages(runtime.memory.for_scope(scopes[2].scope_id), "legal", limit=2) == ()


class _UnusedMemory:
    async def search(self, request: SearchMemoryRequest, /) -> MemorySearchPage:
        pytest.fail("invalid arguments must fail before Memory search")

    async def list(self) -> MemoryEntriesPage:
        pytest.fail("invalid arguments must fail before Memory listing")


async def test_document_identity_preserves_short_names_without_selecting_year_or_file_noise() -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    target = "Net cash provided by operating activities was 100 million in 2024."
    pages = [(f"{company}_2024_10K", 1, target) for company in ("GE", "3M", "ORION")]
    pages.extend((
        ("ACME_2024_10K", 1, "Operating cash flows from operating activities were 900 million in 2024."),
        (
            "ACME_2024_10K",
            2,
            "Consolidated statements of cash flows. Operating cash flows and operating activities for 2024.",
        ),
        ("THE_2024_ANNUAL_REPORT.PDF", 1, "Other cash flow information for 2024."),
        ("IN_2024_ANNUAL_REPORT.PDF", 1, "Other operating activity information for 2024."),
        ("WE_2024_ANNUAL_REPORT.PDF", 1, "Other cash flow information for 2024."),
    ))
    async with open_builtin_runtime(
        BuiltinConfig(database=SQLiteConfig()), candidate_pipeline=_PagePipeline()
    ) as runtime:
        assert runtime.scopes is not None
        scope = await runtime.scopes.create(
            ScopeDraft(title="Identity labels", summary="Synthetic report pages", idempotency_key="identity-labels")
        )
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        for document_id, page_number, text in pages:
            source = await registry.resolve(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
                    page_number=page_number,
                    text=text,
                )
            )
            await runtime.ingestion.submit(
                SubmitSourceObservation(
                    scope_id=scope.scope_id, observation=project_source_for_transport(registry, source)
                )
            )
        memory = runtime.memory.for_scope(scope.scope_id)
        while (await memory.flush()).processed:
            pass

        async def documents(question: str, limit: int) -> list[str]:
            citations = await search_pdf_pages(memory, question, limit=limit)
            return [
                (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.text.partition("\n")[0]
                for citation in citations
            ]

        for company in ("GE", "3M", "ORION"):
            for apostrophe in ("'", chr(0x2019)):
                question = f"What are {company}{apostrophe}s operating cash flows in fiscal 2024?"
                for limit in (1, 20):
                    assert await documents(question, limit) == [f"{company}_2024_10K"]
        expected = {document for document, _, _ in pages}
        for question in (
            "What are operating cash flows in fiscal 2024?",
            "What are the operating cash flows in the annual PDF report for fiscal 2024?",
            "How much operating cash flow did we have in fiscal 2024?",
            "2024",
        ):
            returned = await documents(question, 20)
            assert len(returned) == len(pages) and set(returned) == expected


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_invalid_search_limit_fails_before_any_memory_access(limit) -> None:
    with pytest.raises(ValueError):
        await search_pdf_pages(_UnusedMemory(), "revenue", limit=limit)
