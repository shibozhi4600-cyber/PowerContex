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

_PRIORITY = ("balance", "income", "cash", "dividend", "performance", "segment", "acquisition", "business", "customer")
_FINANCIAL_KINDS = frozenset({"balance", "income", "cash", "dividend", "performance", "segment", "acquisition"})
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
_FORMULA_INTENTS = (
    (re.compile(r"\bcapital[-\s]+intens(?:ive|ity)\b"), frozenset({"balance", "income", "cash"})),
    (
        re.compile(r"\binventor(?:y|ies)\b.*\bturnover\b|\bturnover\b.*\binventor(?:y|ies)\b"),
        frozenset({"balance", "income"}),
    ),
    (
        re.compile(r"\breturn\s+on\s+(?:assets|equity|(?:invested\s+)?capital)\b|\b(?:roa|roe|roic|roce)\b"),
        frozenset({"balance", "income"}),
    ),
    (re.compile(r"\b(?:assets?|receivables?|payables?)\s+turnover\b"), frozenset({"balance", "income"})),
    (re.compile(r"\bdebt\s+(?:service\s+)?coverage\b"), frozenset({"balance", "income", "cash"})),
)
_STATEMENT_ROWS = {
    "balance": (
        r"total\s+assets",
        r"total\s+liabilities",
        r"current\s+assets",
        r"current\s+liabilities",
        r"(?:share|stock)holders.?\s*equity",
    ),
    "income": (
        r"(?:net\s+sales|(?:total\s+)?revenues?)",
        r"cost\s+of\s+(?:sales|revenue)",
        r"(?:operating\s+income|income\s+before|net\s+income|net\s+(?:loss|earnings))",
    ),
    "cash": (r"operating\s+activities", r"investing\s+activities", r"financing\s+activities"),
}


def _intents(question: str) -> set[str]:
    normalized = " ".join(question.casefold().split())
    result = {kind for kind, pattern in _INTENT_PATTERNS if pattern.search(normalized)}
    if re.search(r"categor(?:y|ies)|products?|services?", normalized) and re.search(
        r"revenues?|sales|%|percent", normalized
    ):
        result.add("segment")
    for pattern, kinds in _FORMULA_INTENTS:
        if pattern.search(normalized):
            result.update(kinds)
    if re.search(r"\b(?:statements?\s+of\s+(?:income|operations|earnings)|net\s+income)\b", normalized):
        result.add("income")
    if re.search(r"\b(?:statements?\s+of\s+financial\s+position|assets?|liabilities)\b", normalized) and re.search(
        r"\b(?:ratio|return|statements?)\b", normalized
    ):
        result.add("balance")
    if re.search(r"\bdividends?\b|\bdividend\s+distribution\b", normalized):
        result.add("dividend")
    if re.search(r"(?<!\w)m\s*&\s*a(?!\w)|\borganic\b|\bunderlying\b|local[-\s]+currency", normalized):
        result.add("acquisition")
    return result


def _statement_features(text: str) -> dict[str, int]:
    body = text.partition("\n")[2]
    lines = [" ".join(line.split()).casefold().strip(" .:") for line in body[:2500].splitlines()]
    lines = [line for line in lines if line]
    contents = "tableofcontents" in re.sub(r"\s+", "", body[:150].casefold())
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
    _primary_statement_features(body, lines, features)
    normalized = " ".join(body.casefold().split())
    if re.search(r"\bdividends?\b", normalized):
        if re.search(r"\bconsecutive\s+years?\b|\byears?\s+of\s+(?:dividend\s+)?increases\b", normalized):
            features["dividend"] = 6
        elif re.search(r"\bdividends?\s+per\s+(?:common\s+)?share\b|\bdeclared\b.*\bdividends?\b", normalized):
            features["dividend"] = 3
        elif any(
            re.fullmatch(r"(?:cash\s+)?dividends?(?:\s+(?:and\s+)?(?:share|stock)\s+repurchases)?", line)
            for line in lines
        ):
            features["dividend"] = 4
    return features


def _primary_statement_features(body: str, lines: list[str], features: dict[str, int]) -> None:
    normalized = " ".join(body.casefold().split())
    for index, original in enumerate(lines):
        # Primary statements start with short headers and tabular rows. A title
        # appearing after narrative prose is often a column caption or reference.
        if index > 10 or len(original) > 130 or len(original.split()) > 20:
            break
        title = original.replace("’", "'")  # noqa: RUF001 - Normalize typographic possessives.
        if "...." in title:
            continue
        if index + 1 < len(lines) and _PAGE_REFERENCE.fullmatch(lines[index + 1]):
            continue
        title = re.sub(r"\s*\((?:unaudited|audited|continued)\)\s*$", "", title)
        for kind, pattern, _ in _TITLE_PATTERNS:
            if kind not in _STATEMENT_ROWS or not pattern.fullmatch(title):
                continue
            structure = sum(bool(re.search(row, normalized)) for row in _STATEMENT_ROWS[kind])
            full = title.startswith(("consolidated ", "combined ", "condensed ", "audited ", "unaudited "))
            confidence = 6 if full and structure >= 2 else 5 if structure >= 2 else 4 if full else 3
            header = " ".join(lines[:index])
            if re.search(
                r"schedule\s+i\b|financial\s+information\s+of\s+parent|parent[-\s]+company|"
                r"notes?\s+to\s+(?:the\s+)?consolidated\s+financial\s+statements",
                header,
            ):
                confidence -= 2
            features[kind] = max(features.get(kind, 0), confidence)


def _route_acronym(order: list[int], texts: Sequence[str], question: str) -> list[int]:
    labels = {texts[index].partition("\n")[0] for index in order}
    if len(labels) < 2:
        return order
    # A component of SG&A, PP&E or another ampersand compound is not an
    # independent company alias. Standalone unknown acronyms remain eligible.
    alias_query = re.sub(r"(?<!\w)[A-Z][A-Z0-9]{0,7}(?:\s*&\s*[A-Z][A-Z0-9]{0,7})+(?!\w)", " ", question)
    acronyms = {
        term
        for term in re.findall(r"(?<!\w)[A-Z][A-Z0-9]{1,7}(?!\w)", alias_query)
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
    """Reserve bounded slots for independently named financial evidence.

    Positions identify the caller's original records; this function never
    creates text or citations. Explicit document matching remains authoritative.
    Three-statement formulas may reserve three slots; other intents reserve two.
    """

    order = list(baseline_order)
    kinds = _intents(question)
    if allow_acronym_routing and kinds & _FINANCIAL_KINDS:
        order = _route_acronym(order, texts, question)
    if not kinds:
        return tuple(order)
    positions = {index: position for position, index in enumerate(order)}
    features = {index: _statement_features(texts[index]) for index in order}
    max_slots = 3 if {"balance", "income", "cash"} <= kinds else 2
    quota = min(max_slots, max(1, limit // 4))
    prefix = order[: max(0, limit - quota)]
    anchors: list[int] = []
    for kind in _PRIORITY:
        if kind not in kinds:
            continue
        found = [index for index in order if kind in features[index]]
        found.sort(key=lambda index: (-features[index][kind], positions[index]))
        # A covered first statement must not hide a useful second statement
        # of the same kind, such as a different period or a continuation page.
        found = [index for index in found if index not in prefix and index not in anchors]
        if found:
            anchors.append(found[0])
    promoted = [index for index in anchors if index not in prefix][:quota]
    return tuple(prefix + promoted + [index for index in order[len(prefix) :] if index not in promoted])
