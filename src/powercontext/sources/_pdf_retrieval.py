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

"""Text-only financial evidence diversification over already retrieved PDF pages."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

_PRIORITY = ("balance", "income", "cash", "performance", "segment", "acquisition", "business", "customer")
_FINANCIAL_KINDS = frozenset({"balance", "income", "cash", "performance", "segment", "acquisition"})
_NON_COMPANY_ACRONYMS = frozenset({
    "GAAP",
    "IFRS",
    "EBITDA",
    "EBIT",
    "EPS",
    "ROE",
    "ROA",
    "ROI",
    "SEC",
    "USD",
    "FY",
    "PDF",
    "CEO",
    "CFO",
    "COO",
    "CTO",
    "CIO",
    "CMO",
    "CRO",
    "CHRO",
    "HR",
    "SVP",
    "EVP",
})
_STATEMENT_PREFIX = r"(?:(?:consolidated|combined|condensed|audited|unaudited)\s+)*"
_PAGE_REFERENCE = re.compile(r"(?:[a-z][-–]?)?\d{1,3}|[ivxlcdm]{1,7}")  # noqa: RUF001 - Printed page references use both dashes.
_TITLE_RULES = (
    ("income", _STATEMENT_PREFIX + r"(?:statements?\s+of\s+(?:income|operations|earnings)|income\s+statements?)", 3),
    ("cash", _STATEMENT_PREFIX + r"(?:statements?\s+of\s+cash\s+flows?|cash\s+flow\s+statements?)", 3),
    ("balance", _STATEMENT_PREFIX + r"(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)", 3),
    (
        "performance",
        r"(?:summary\s+of\s+)?(?:financial\s+(?:performance|highlights|summary)|selected\s+financial\s+data)",
        3,
    ),
    ("performance", r"(?:item\s*7[.\s]*)?management'?s?\s+discussion\s+and\s+analysis\b.*(?<!\d)", 2),
    ("performance", r"results of operations", 1),
    (
        "segment",
        r"(?:summary\s+of\s+)?(?:business\s+)?segment(?:s|\s+(?:data|information|reporting|results))(?:\s+and\s+related\s+information)?",
        3,
    ),
    (
        "acquisition",
        r"(?:business\s+combinations?|acquisitions?(?:\s+(?:and|&)\s+divestitures?)?|acquisitions?,?\s+divestitures?\s+and\s+other\s+transactions)",
        2,
    ),
    ("business", r"(?:part\s+i\s+)?item\s*1[.\s]+business", 3),
    ("business", r"business overview|business|company overview|overview of business", 2),
    ("customer", r"business environment|customer relationships|customer metrics|customer retention", 3),
)
_TITLE_PATTERNS = tuple((kind, re.compile(pattern), confidence) for kind, pattern, confidence in _TITLE_RULES)
_INTENT_PATTERNS = (
    (
        "balance",
        re.compile(
            r"\b(?:quick\s+ratio|current\s+ratio|liquidity|debt.to.equity|return\s+on\s+(?:assets|equity))\b|balance\s+sheet"
        ),
    ),
    (
        "income",
        re.compile(
            r"\b(?:gross|operating|net|profit)\s+(?:profit\s+)?margin|\bebitda\b|\bincome\s+statements?\b|\bp\s*&\s*l\b|effective\s+tax\s+rate"
        ),
    ),
    ("cash", re.compile(r"\bcash\s*flows?\b|\bebitda\b")),
    (
        "performance",
        re.compile(
            r"(?:margin|revenue|sales|tax\s+rate).*?(?:chang|improv|declin)|(?:drove|drives|changed|change).*?(?:margin|revenue|sales)|effective\s+tax\s+rate"
        ),
    ),
    ("segment", re.compile(r"\bsegments?\b")),
    ("acquisition", re.compile(r"\bacquisitions?\b|\bmergers?\b|business\s+combinations?")),
    ("business", re.compile(r"\bproducts?\b.*\bservices?\b|\bindustry\b|\bgeograph")),
    ("customer", re.compile(r"\bretain|\bretention|card\s*members?|\bcustomers?\b|\bclients?\b")),
)


def _intents(question: str) -> set[str]:
    normalized = " ".join(question.casefold().split())
    result = {kind for kind, pattern in _INTENT_PATTERNS if pattern.search(normalized)}
    if re.search(r"categor(?:y|ies)|products?|services?", normalized) and re.search(
        r"revenues?|sales|%|percent", normalized
    ):
        result.add("segment")
    return result


def _statement_features(text: str) -> dict[str, int]:
    lines = [" ".join(line.split()).casefold().strip(" .:") for line in text.partition("\n")[2][:2000].splitlines()]
    lines = [line for line in lines if line]
    contents = "table of contents" in lines
    features: dict[str, int] = {}
    for index, original in enumerate(lines):
        title = original.replace("’", "'")  # noqa: RUF001 - Normalize typographic possessives.
        if len(title) > 190 or "...." in title:
            continue
        # A running "Table of Contents" link also appears on real statement
        # pages. Only a following standalone page reference makes a TOC row.
        if contents and index + 1 < len(lines) and _PAGE_REFERENCE.fullmatch(lines[index + 1]):
            continue
        title = re.sub(r"^table\s+\d+\s*[:.-]?\s*", "", title)
        note = bool(re.match(r"^note\s+\d+\b", title))
        title = re.sub(r"^note\s+\d+\s*[-.:]?\s*", "", title)
        title = re.sub(r"\s*\((?:unaudited|audited|continued)\)\s*$", "", title)
        for kind, pattern, confidence in _TITLE_PATTERNS:
            if pattern.fullmatch(title):
                value = 3 if kind == "acquisition" and note else confidence
                features[kind] = max(features.get(kind, 0), value)
    return features


def _route_acronym(order: list[int], texts: Sequence[str], question: str) -> list[int]:
    labels = {texts[index].partition("\n")[0] for index in order}
    if len(labels) < 2:
        return order
    acronyms = {
        term
        for term in re.findall(r"(?<!\w)[A-Z][A-Z0-9]{1,7}(?!\w)", question)
        if term not in _NON_COMPANY_ACRONYMS and not re.fullmatch(r"FY\d+|\d+[KQF]", term)
    }
    # Ambiguous or multi-entity questions retain their complete candidate pool.
    if len(acronyms) != 1:
        return order
    pattern = re.compile(r"(?<!\w)" + re.escape(next(iter(acronyms))) + r"(?!\w)", re.I)
    support: Counter[str] = Counter()
    sizes = Counter(texts[index].partition("\n")[0] for index in order)
    for index in order:
        label, _, body = texts[index].partition("\n")
        if pattern.search(body):
            support[label] += 1
    if not support:
        return order
    ordered = support.most_common()
    label, count = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    if count < max(3, sizes[label] * 0.02) or count < 5 * max(1, runner_up):
        return order
    return [index for index in order if texts[index].partition("\n")[0] == label]


def rank_pdf_candidates(
    texts: Sequence[str],
    baseline_order: Sequence[int],
    question: str,
    limit: int,
    *,
    allow_acronym_routing: bool = False,
) -> tuple[int, ...]:
    """Reserve at most two slots for independently named financial evidence.

    Positions identify the caller's original records; this function never
    creates text or citations. Explicit document matching remains authoritative.
    """

    order = list(baseline_order)
    kinds = _intents(question)
    if allow_acronym_routing and kinds & _FINANCIAL_KINDS:
        order = _route_acronym(order, texts, question)
    if not kinds:
        return tuple(order)
    positions = {index: position for position, index in enumerate(order)}
    features = {index: _statement_features(texts[index]) for index in order}
    anchors: list[int] = []
    for kind in _PRIORITY:
        if kind not in kinds:
            continue
        found = [index for index in order if kind in features[index]]
        found.sort(key=lambda index: (-features[index][kind], positions[index]))
        if found and found[0] not in anchors:
            anchors.append(found[0])
    quota = min(2, max(1, limit // 4))
    prefix = order[: max(0, limit - quota)]
    promoted = [index for index in anchors if index not in prefix][:quota]
    return tuple(prefix + promoted + [index for index in order[len(prefix) :] if index not in promoted])
