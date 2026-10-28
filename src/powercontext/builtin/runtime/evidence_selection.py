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

"""Offline query-guided extraction with literal ranges and a complete wire budget.

Sentence windows compete on query coverage and local lexical relevance. The
original text is never normalized or rewritten: normalization is only an index
used for scoring. All final fitting measures the complete rendered envelope.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from powercontext.artifacts import ArtifactAddress
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

if TYPE_CHECKING:
    from powercontext.builtin.runtime.models import PrepareContextRequest
    from powercontext.builtin.runtime.prepared_context import (
        MemoryEntryAddress,
        PreparedContextBuild,
        PreparedContextFragment,
        PreparedMemoryCandidates,
    )

_BEGIN = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_BREAKS = re.compile(r"[.!?。！？](?:[\"'”’\)\]])?(?=\s|$)|\n+")  # noqa: RUF001 -- literal multilingual boundaries
_STOP = frozenset([
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "to",
    "with",
    "without",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "their",
    "there",
    "they",
    "them",
    "we",
    "our",
    "you",
    "your",
    "can",
    "could",
    "may",
    "might",
    "will",
    "would",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "than",
    "then",
    "which",
    "who",
    "what",
    "when",
    "where",
    "how",
    "also",
    "both",
    "through",
    "during",
    "using",
])
_ABBREVIATIONS = frozenset({"al", "dr", "eg", "fig", "ie", "mr", "mrs", "no", "prof", "vs"})
_RESULT_WORDS = frozenset({
    "found",
    "observed",
    "result",
    "results",
    "revealed",
    "demonstrated",
    "increased",
    "reduced",
    "associated",
    "conclude",
    "conclusion",
    "conclusions",
    "suggest",
    "showed",
    "significantly",
})


@dataclass(frozen=True)
class _Sentence:
    start: int
    end: int
    terms: Counter[str]
    raw_terms: frozenset[str]


@dataclass(frozen=True)
class _Document:
    address: MemoryEntryAddress
    text: str
    sentences: tuple[_Sentence, ...]
    rank: int


@dataclass(frozen=True)
class _Window:
    document: int
    first: int
    last: int
    fragment: PreparedContextFragment
    utility: float


def _stem(word: str) -> str:
    # Conservative morphological matching preserves short biomedical identifiers.
    if not word.isascii() or len(word) < 5:
        return word
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ingly", "edly", "ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
                word = word[:-1]
            break
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def _tokens(text: str) -> list[str]:
    return [_stem(word) for word in _WORDS.findall(unicodedata.normalize("NFKC", text).casefold()) if word not in _STOP]


def _sentences(text: str) -> tuple[_Sentence, ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BREAKS.finditer(text):
        if match.group().startswith("."):
            previous = re.search(r"([A-Za-z]+)\.$", text[start : match.start() + 1])
            if previous is not None and (previous[1].casefold() in _ABBREVIATIONS or len(previous[1]) == 1):
                continue
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    result = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            raw = frozenset(_WORDS.findall(text[start:end].casefold()))
            result.append(_Sentence(start, end, Counter(_tokens(text[start:end])), raw))
    return tuple(result)


def _render(fragments: Sequence[PreparedContextFragment]) -> str:
    items = []
    for fragment in fragments:
        address = fragment.address
        items.append({
            "citation": {
                "scope_id": address.memory.scope_id,
                "memory_ref": address.memory.artifact.model_dump(mode="json"),
                "entry_id": address.entry_id,
                "entry_version_id": address.entry_version_id,
            },
            "content": fragment.text,
            "start": fragment.start,
            "end": fragment.end,
        })
    body = json.dumps({"trust": "untrusted_history", "items": items}, ensure_ascii=False, separators=(",", ":"))
    return f"{TRUST_POLICY}\n\n{_BEGIN}\n{body}\n{_END}"


def _size(fragments: Sequence[PreparedContextFragment]) -> int:
    return len(_render(fragments).encode("utf-8"))


def _documents(candidates: Sequence[PreparedMemoryCandidates], current_scope_id: str | None) -> tuple[_Document, ...]:
    from powercontext.builtin.runtime.prepared_context import MemoryEntryAddress

    result: list[_Document] = []
    seen: set[tuple[str, str, str, int, str, str]] = set()
    for group in candidates:
        if group.hits and group.memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        for hit in group.hits:
            if hit.memory_ref != group.memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            scope_id = group.scope_id or current_scope_id
            if not scope_id:
                raise PreparedContextInvariantError("memory-scope-missing")
            address = MemoryEntryAddress(
                ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref), hit.entry_id, hit.entry_version_id
            )
            identity = (
                scope_id,
                hit.memory_ref.family,
                hit.memory_ref.artifact_id,
                hit.memory_ref.revision,
                hit.entry_id,
                hit.entry_version_id,
            )
            if identity not in seen:
                seen.add(identity)
                result.append(_Document(address, hit.text, _sentences(hit.text), len(result)))
    return tuple(result)


def _scores(documents: tuple[_Document, ...], query: str) -> list[list[float]]:
    query_terms = set(_tokens(query))
    if not query_terms:
        query_terms = set(_WORDS.findall(query.casefold()))
    sentences = [sentence for document in documents for sentence in document.sentences]
    frequencies = Counter(term for sentence in sentences for term in query_terms if term in sentence.terms)
    weights = {term: 1.0 + math.log((len(sentences) + 1) / (frequencies[term] + 1)) for term in query_terms}
    total = sum(weights.values()) or 1.0
    average = sum(sum(sentence.terms.values()) for sentence in sentences) / max(len(sentences), 1)
    result: list[list[float]] = []
    for document in documents:
        document_terms = set().union(*(sentence.terms for sentence in document.sentences))
        document_coverage = sum(weights[term] for term in query_terms & document_terms) / total
        scores = []
        for sentence in document.sentences:
            length = sum(sentence.terms.values())
            score = 0.0
            coverage = 0.0
            for term in query_terms:
                count = sentence.terms[term]
                # Shared long prefixes match inflectional variants; exact identifiers
                # and numbers are never expanded this way.
                if not count and len(term) >= 6 and term.isascii() and term.isalpha():
                    count = 0.6 * max(
                        (value for token, value in sentence.terms.items() if len(token) >= 6 and token[:6] == term[:6]),
                        default=0,
                    )
                if count:
                    coverage += weights[term] * min(1.0, count)
                    score += weights[term] * (2.2 * count) / (count + 1.2 * (0.35 + 0.65 * length / max(average, 1)))
            if score:
                score = (score / total + 0.6 * coverage / total) * (0.5 + document_coverage)
                score *= 1.10 if sentence.raw_terms & _RESULT_WORDS else 1.0
            scores.append(score)
        result.append(scores)
    return result


def _utility(scores: Sequence[float], length: int) -> float:
    ranked = sorted(scores, reverse=True)
    return sum(
        score * weight for score, weight in zip(ranked, (1.0, 0.65, 0.45, 0.3, 0.2), strict=False)
    ) + 0.008 * math.log1p(length)


def _windows(
    document: _Document, index: int, scores: list[float], max_bytes: int, query_terms: set[str]
) -> list[_Window]:
    from powercontext.builtin.runtime.prepared_context import PreparedContextFragment

    windows: list[_Window] = []
    for first, sentence in enumerate(document.sentences):
        for last in range(first, len(document.sentences)):
            end = document.sentences[last].end
            fragment = PreparedContextFragment(
                document.address, sentence.start, end, document.text[sentence.start : end]
            )
            if _size((fragment,)) > max_bytes:
                break
            if max(scores[first : last + 1], default=0) > 0:
                utility = _utility(scores[first : last + 1], len(fragment.text))
                windows.append(_Window(index, first, last, fragment, utility))
    windows.extend(_clipped_windows(document, index, scores, max_bytes, query_terms, bool(windows)))
    if not windows:
        # A candidate with no lexical overlap is still an honest fallback, rather
        # than fabricated evidence. Its score remains below matching passages.
        for position, sentence in enumerate(document.sentences):
            fragment = PreparedContextFragment(
                document.address, sentence.start, sentence.end, document.text[sentence.start : sentence.end]
            )
            if _size((fragment,)) <= max_bytes:
                windows.append(_Window(index, position, position, fragment, _utility([], len(fragment.text))))
    return _rank_windows(windows)


def _clipped_windows(
    document: _Document,
    index: int,
    scores: list[float],
    max_bytes: int,
    query_terms: set[str],
    has_matching_window: bool,
) -> list[_Window]:
    """Fit query-centred ranges when a complete sentence exceeds the wire budget."""
    from powercontext.builtin.runtime.prepared_context import PreparedContextFragment

    windows = []
    for position, sentence in enumerate(document.sentences):
        whole = PreparedContextFragment(
            document.address, sentence.start, sentence.end, document.text[sentence.start : sentence.end]
        )
        if _size((whole,)) <= max_bytes or (scores[position] == 0 and has_matching_window):
            continue
        matches = [
            match
            for match in _WORDS.finditer(document.text, sentence.start, sentence.end)
            if _stem(unicodedata.normalize("NFKC", match.group()).casefold()) in query_terms
        ]
        anchors = [match.start() for match in matches] or [sentence.start]
        for anchor in anchors[:32]:
            empty = PreparedContextFragment(document.address, anchor, anchor + 1, document.text[anchor : anchor + 1])
            capacity = max(0, max_bytes - _size((empty,)))
            start = max(sentence.start, anchor - capacity // 4)
            # Prefer a complete word at the leading boundary while retaining the match.
            while start < anchor and start > sentence.start and not document.text[start - 1].isspace():
                start += 1
            lower, upper = anchor + 1, sentence.end
            best = None
            while lower <= upper:
                end = (lower + upper) // 2
                fragment = PreparedContextFragment(document.address, start, end, document.text[start:end])
                if _size((fragment,)) <= max_bytes:
                    best = fragment
                    lower = end + 1
                else:
                    upper = end - 1
            if best is not None:
                coverage = len(set(_tokens(best.text)) & query_terms) / max(len(query_terms), 1)
                utility = _utility([scores[position] * coverage], len(best.text))
                windows.append(_Window(index, position, position, best, utility))
    return windows


def _rank_windows(windows: list[_Window]) -> list[_Window]:
    # Keep different starting positions represented, including later evidence.
    best_by_start: dict[int, _Window] = {}
    for window in windows:
        previous = best_by_start.get(window.first)
        if previous is None or window.utility > previous.utility:
            best_by_start[window.first] = window
    ranked = sorted(windows, key=lambda window: (-window.utility, window.first, window.last))[:32]
    unique = {(window.fragment.start, window.fragment.end): window for window in ranked}
    for window in best_by_start.values():
        unique.setdefault((window.fragment.start, window.fragment.end), window)
    return list(unique.values())


def _choose_windows(windows: list[list[_Window]], scores: list[list[float]], max_bytes: int) -> tuple[_Window, ...]:
    options: list[tuple[_Window, ...]] = [(window,) for group in windows for window in group]
    for group in windows:
        for left, right in combinations(group[:48], 2):
            left, right = sorted((left, right), key=lambda window: window.first)
            if right.first > left.last + 1:
                options.append((left, right))
    # Complementary sources compete against the existing single-document options
    # under exactly the same complete-envelope budget and scoring objective.
    cross = sorted((window for group in windows for window in group[:24]), key=lambda window: -window.utility)[:96]
    options.extend((left, right) for left, right in combinations(cross, 2) if left.document != right.document)
    best: tuple[_Window, ...] = ()
    best_utility = -1.0
    for option in options:
        if _size(tuple(window.fragment for window in option)) > max_bytes:
            continue
        utility = option[0].utility
        if len(option) > 1:
            selected_scores = [
                score for window in option for score in scores[window.document][window.first : window.last + 1]
            ]
            utility = _utility(selected_scores, sum(len(window.fragment.text) for window in option))
        if utility > best_utility:
            best, best_utility = option, utility
    return best


def select_memory_evidence(
    request: PrepareContextRequest,
    candidates: Sequence[PreparedMemoryCandidates],
    *,
    current_scope_id: str | None,
) -> PreparedContextBuild:
    """Select exact literal evidence and fit its entire trust/citation envelope."""
    from powercontext.builtin.runtime.models import PreparedContext
    from powercontext.builtin.runtime.prepared_context import PreparedContextBuild, PreparedContextOmissions

    documents = _documents(candidates, current_scope_id)
    scores = _scores(documents, request.query)
    query_terms = set(_tokens(request.query))
    windows = [
        _windows(document, index, scores[index], request.max_bytes, query_terms)
        for index, document in enumerate(documents)
    ]
    best = _choose_windows(windows, scores, request.max_bytes)
    fragments = tuple(window.fragment for window in best)
    if not fragments:
        return PreparedContextBuild(context=PreparedContext(status="empty", content=None, content_bytes=0), origins=())
    content = _render(fragments)
    selected = {window.document for window in best}
    omissions = PreparedContextOmissions(
        truncated_items=sum(
            fragment.start != 0 or fragment.end != len(documents[window.document].text)
            for fragment, window in zip(fragments, best, strict=True)
        ),
        dropped_items=len(documents) - len(selected),
        dropped_no_fitting_truncation=len(documents) - len(selected),
    )
    return PreparedContextBuild(
        context=PreparedContext(status="ready", content=content, content_bytes=len(content.encode("utf-8"))),
        origins=tuple(fragment.address for fragment in fragments),
        omissions=omissions,
        fragments=fragments,
    )
