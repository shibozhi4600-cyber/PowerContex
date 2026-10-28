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

"""Complete PDF page captures and offline retrieval of exact Memory citations.

Install ``powercontext[pdf]`` for the pinned parser used by page capture.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from itertools import pairwise
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import MemoryEntriesPage, MemorySearchPage, SearchMemoryRequest
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PDFDocumentInput(BaseModel):
    """The complete original bytes of one offline PDF document."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def require_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PDF document_id must not be blank")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical page retaining its complete extractable text and revision."""

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1, strict=True)
    text: str

    @field_validator("document_id")
    @classmethod
    def require_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PDF document_id must not be blank")  # noqa: TRY003
        return value

    @field_validator("document_sha256")
    @classmethod
    def require_sha256(cls, value: str) -> str:
        if re.fullmatch("[0-9a-f]{64}", value) is None:
            raise ValueError("PDF document_sha256 must be lowercase SHA-256 hexadecimal")  # noqa: TRY003
        return value


class PDFPageSource(Source, PDFPageInput):
    """A captured page whose identity includes the entire PDF revision digest."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Read every physical page, including empty pages, before returning a result."""

    from pypdf import PdfReader

    if not value.content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a user password")  # noqa: TRY003, TRY301
        # Extraction is eager: a broken later page never yields partial results.
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=index,
                text=page.extract_text() or "",
            )
            for index, page in enumerate(reader.pages, start=1)
        )
        if not pages:
            raise ValueError("PDF must contain at least one physical page")  # noqa: TRY003, TRY301
    except Exception as error:
        raise ValueError("PDF could not be completely parsed") from error  # noqa: TRY003
    return pages


class _PDFPageAdapter:
    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        document_key = hashlib.sha256(value.document_id.encode("utf-8")).hexdigest()
        identity = f"pdf:{document_key}:{value.document_sha256}:{value.page_number}"
        return PDFPageSource(name=identity, **value.model_dump())

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput.model_validate(source.model_dump())


class _PDFTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    _PDFPageAdapter(), projections=(_PDFTextProjection(),)
)


class _ScopedMemory(Protocol):
    async def search(self, request: SearchMemoryRequest, /) -> MemorySearchPage: ...

    async def list(self) -> MemoryEntriesPage: ...


_WORDS = re.compile(r"[a-z]+|\d+")
_IDENTITY_WORDS = re.compile(r"[^\W_]+")
_FILING_MARKERS = re.compile(r"(?<![a-z0-9])(?:10[-_ ]?[kq]|20[-_ ]?f|40[-_ ]?f|[68][-_ ]?k)(?![a-z0-9])")
_IDENTITY_NOISE = frozenset([
    "pdf",
    "txt",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "csv",
    "json",
    "html",
    "htm",
    "md",
    "quarterly",
    "quarter",
    "reports",
    "annualreport",
    "annualreports",
    "filing",
    "filings",
    "financial",
    "financials",
    "we",
    "us",
    "me",
    "my",
    "mine",
    "ourselves",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "whom",
    "whose",
    "any",
    "all",
    "some",
    "every",
    "much",
    "many",
    "such",
    "whether",
    "also",
])
_STOP_WORDS = frozenset([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "only",
    "or",
    "our",
    "per",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "fy",
    "fiscal",
    "year",
    "years",
    "annual",
    "report",
    "form",
    "question",
    "answer",
    "respond",
    "give",
    "calculate",
    "assuming",
    "assume",
    "perspective",
    "investment",
    "analyst",
    "details",
    "shown",
    "relying",
    "use",
    "using",
    "directly",
    "outlined",
    "explicitly",
    "state",
    "quantity",
    "primarily",
    "important",
    "ongoing",
])
_FINANCIAL_GROUPS = (
    ("revenue", "revenues", "sales", "turnover"),
    ("profit", "profits", "earnings", "income"),
    ("legal", "litigation", "lawsuits", "proceedings", "contingencies"),
    ("geography", "geographies", "geographic", "geographical", "regions", "countries", "markets"),
    ("restructuring", "reorganization", "reorganisation"),
    ("debt", "borrowings", "borrowing", "indebtedness", "maturities"),
    ("equity", "shareholders", "stockholders"),
    ("capex", "expenditures", "purchases", "property", "equipment"),
    ("dividend", "dividends", "distributions"),
)


def _stem(word: str) -> str:
    if word in {"geography", "geographies", "geographic", "geographical"}:
        return "geograph"
    if word in {"operate", "operates", "operating", "operations", "operational"}:
        return "operat"
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    return word


def _terms(text: str) -> list[str]:
    return [_stem(word) for word in _WORDS.findall(text.lower()) if word not in _STOP_WORDS]


def _identity_terms(text: str) -> set[str]:
    # Identity words keep digits attached to letters and do not undergo stemming.
    # Filing/date metadata and possessive suffixes are not company identities.
    normalized = re.sub(r"['’]s(?![a-z0-9])", "", text.casefold())  # noqa: RUF001 - Also handle typographic possessives.
    words = _IDENTITY_WORDS.findall(_FILING_MARKERS.sub(" ", normalized))
    return {
        word
        for word in words
        if any(character.isalpha() for character in word)
        and word not in _STOP_WORDS
        and word not in _IDENTITY_NOISE
        and re.fullmatch(r"(?:(?:fy|fiscal)\d{2,4}|q[1-4]|(?:19|20)\d{2}q[1-4])", word) is None
    }


def _query_weights(question: str) -> dict[str, float]:
    words = set(_WORDS.findall(question.lower()))
    weights = dict.fromkeys(_terms(question), 1.0)
    for group in _FINANCIAL_GROUPS:
        if words.intersection(group):
            for word in group:
                weights.setdefault(_stem(word), 0.35)
    if "ebitda" in words:
        for word in ("operating", "income", "depreciation", "amortization"):
            weights.setdefault(_stem(word), 0.6)
    return weights


async def search_pdf_pages(memory: _ScopedMemory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search scoped Memory and rerank its current captured pages without models.

    Whole-scope reranking reads the public current-entry list once per query;
    its time and memory cost scale with the scope's page count. No corpus or
    results survive the call, and citations come directly from stored entries.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("PDF question must be a non-blank string")  # noqa: TRY003
    if type(limit) is not int or limit < 1:
        raise ValueError("PDF result limit must be a positive integer")  # noqa: TRY003
    weights = _query_weights(question)
    query = " ".join(dict.fromkeys(_WORDS.findall(question.lower())))
    coarse = await memory.search(SearchMemoryRequest(query=query or question, mode="fts", limit=100))
    coarse_ranks = {hit.entry_version_id: rank for rank, hit in enumerate(coarse.hits, start=1)}
    current = await memory.list()
    records = [
        record
        for record in current.entries
        if len(record.entry.sources) == 1 and record.entry.sources[0].source_type == "pdf-page"
    ]
    if not records:
        return ()

    # Document labels are part of the stored projection, not an external corpus.
    # Restrict only when the question explicitly matches a document label word;
    # a year alone never selects an unrelated company.
    question_terms = set(_terms(question))
    identity_terms = _identity_terms(question)
    labels = [_identity_terms(record.entry.text.partition("\n")[0]) for record in records]
    label_matches = [len(label & identity_terms) for label in labels]
    best_label = max(label_matches)
    if best_label:
        selected = [record for record, count in zip(records, label_matches, strict=True) if count == best_label]
        years = {word for word in question_terms if re.fullmatch(r"(?:19|20)\d{2}", word)}
        year_matched = [record for record in selected if years & set(_terms(record.entry.text.partition("\n")[0]))]
        records = year_matched or selected

    frequencies: list[Counter[str]] = []
    lengths: list[int] = []
    document_frequency: Counter[str] = Counter()
    for record in records:
        terms = _terms(record.entry.text)
        counts = Counter(term for term in terms if term in weights)
        frequencies.append(counts)
        lengths.append(len(terms))
        document_frequency.update(counts.keys())
    average_length = max(sum(lengths) / len(lengths), 1)
    inverse_frequency = {
        term: math.log1p((len(records) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in weights
    }
    question_pairs = set(pairwise(_terms(question)))
    scored: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        counts = frequencies[index]
        normalization = 1.2 * (0.25 + 0.75 * lengths[index] / average_length)
        score = sum(
            weights[term] * inverse_frequency[term] * count * 2.2 / (count + normalization)
            for term, count in counts.items()
        )
        heading_terms = _terms(record.entry.text[:700])
        heading_pairs = set(pairwise(heading_terms))
        score += sum(
            inverse_frequency.get(left, 0) + inverse_frequency.get(right, 0)
            for left, right in question_pairs & heading_pairs
        )
        score += 0.1 / (60 + coarse_ranks.get(record.entry.entry_version_id, 1000))
        scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    citations: list[MemoryCitation] = []
    pages: set[str] = set()
    for _, index in scored:
        record = records[index]
        page_id = record.entry.sources[0].source_id
        if page_id not in pages:
            citations.append(record.citation)
            pages.add(page_id)
            if len(citations) == limit:
                break
    return tuple(citations)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]
