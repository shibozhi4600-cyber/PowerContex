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

"""PDF Unicode text survives the unchanged whole-page evidence pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import io
import unicodedata
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from competition.selftest_support.source_acceptance import persist
from powercontext.builtin.runtime import GetMemoryEntryRequest, open_builtin_runtime
from powercontext.sources import SourceDefinitionRegistry, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    PDFPageSource,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _pdf_with_decomposed_unicode() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    unicode_map = DecodedStreamObject()
    unicode_map.set_data(
        b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /UnicodeEvidence def\n/CMapType 2 def\n"
        b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        b"1 beginbfchar\n<41> <00650301>\nendbfchar\n"
        b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
    )
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/ToUnicode"): unicode_map,
            })
        })
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Revenue evidence glyph A; 2026 total 12345.) Tj ET")
    page[NameObject("/Contents")] = stream
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_unicode_pdf_page_source_and_memory_preserve_identical_citable_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        document_id = "UNICODE_REPORT_2026_10K"
        raw = _pdf_with_decomposed_unicode()
        extracted = PdfReader(io.BytesIO(raw), strict=True).pages[0].extract_text()
        assert extracted == "Revenue evidence glyph e\u0301; 2026 total 12345."
        assert unicodedata.normalize("NFC", extracted) != extracted

        pages = parse_pdf_document(PDFDocumentInput(document_id=document_id, content=raw))
        assert [(page.page_number, page.document_id) for page in pages] == [(1, document_id), (2, document_id)]
        assert pages[1].text == ""
        digest = hashlib.sha256(raw).hexdigest()
        assert all(page.document_sha256 == digest for page in pages)
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        observations = []
        for page in pages:
            source = await registry.resolve(page)
            assert isinstance(source, PDFPageSource)
            assert await registry.read(source) == page
            assert source.text == page.text
            assert len(source.name) <= 256
            assert (await registry.resolve(page)).name == source.name
            observation = project_source_for_transport(registry, source)
            projection = observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)
            assert isinstance(projection, dict)
            assert projection["content"] == f"{document_id}\n{page.text}"
            assert projection["metadata"] == {
                "document_id": document_id,
                "document_sha256": digest,
                "page_number": page.page_number,
            }
            observations.append(observation)

        # The unchanged official fixture uses complete projection content as
        # each entry body, persists every Source, retries one observation, and
        # closes the Runtime before we reopen it for exact citation checks.
        config, scope_id, stored = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, observations)
        async with open_builtin_runtime(config) as runtime:
            memory = runtime.memory.for_scope(scope_id)
            citations = await search_pdf_pages(memory, "What revenue evidence was reported in 2026?", limit=1)
            assert len(citations) == 1
            entry = (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry
            assert len(entry.sources) == 1
            source = stored[entry.sources[0].source_id]
            projection = source.projection(TEXT_EVIDENCE_PROJECTION_KEY)
            assert isinstance(projection, dict)
            assert entry.text == projection["content"].strip()
            assert source.payload["text"] == pages[0].text == unicodedata.normalize("NFC", extracted)
            assert projection["metadata"] == {
                "document_id": document_id,
                "document_sha256": digest,
                "page_number": 1,
            }

    asyncio.run(scenario())


def test_direct_unicode_page_input_round_trips_through_real_source_and_memory(tmp_path: Path) -> None:
    async def scenario() -> None:
        document_id = "DIRECT_UNICODE_REPORT_2026_10K"
        raw = _pdf_with_decomposed_unicode()
        extracted = PdfReader(io.BytesIO(raw), strict=True).pages[0].extract_text()
        assert unicodedata.normalize("NFC", extracted) != extracted
        digest = hashlib.sha256(raw).hexdigest()
        page = PDFPageInput(document_id=document_id, document_sha256=digest, page_number=1, text=extracted)
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        source = await registry.resolve(page)
        assert isinstance(source, PDFPageSource)
        assert await registry.read(source) == page
        assert source.text == page.text
        assert len(source.name) <= 256
        assert (await registry.resolve(page)).name == source.name
        observation = project_source_for_transport(registry, source)
        projection = observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)
        assert isinstance(projection, dict)
        assert projection["content"] == f"{document_id}\n{page.text}"
        assert projection["metadata"] == {
            "document_id": document_id,
            "document_sha256": digest,
            "page_number": 1,
        }

        config, scope_id, stored = await persist(tmp_path, PDF_PAGE_SOURCE_DEFINITION, (observation,))
        async with open_builtin_runtime(config) as runtime:
            memory = runtime.memory.for_scope(scope_id)
            citations = await search_pdf_pages(memory, "What revenue evidence was reported in 2026?", limit=1)
            assert len(citations) == 1
            entry = (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry
            assert len(entry.sources) == 1
            captured = stored[entry.sources[0].source_id]
            stored_projection = captured.projection(TEXT_EVIDENCE_PROJECTION_KEY)
            assert isinstance(stored_projection, dict)
            assert stored_projection == projection
            assert entry.text == stored_projection["content"].strip()
            assert captured.payload["text"] == page.text == unicodedata.normalize("NFC", extracted)
            assert page.document_id == document_id
            assert page.document_sha256 == digest
            assert page.page_number == 1

    asyncio.run(scenario())
