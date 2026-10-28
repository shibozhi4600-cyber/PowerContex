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

"""HTTP scoring with bounded batches and identity-preserving global selection."""

from __future__ import annotations

import json
import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Rank Memory candidates using scores from a caller-owned HTTP client.

    Batches retain the original one-based ranks, including when their responses
    arrive in a different order. The caller retains ownership of ``client``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a nonempty string")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a nonempty string")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select by score, then original rank."""

        if not isinstance(query, str):
            raise ValueError("query must be a string")  # noqa: TRY003, TRY004 - Reranking inputs fail with ValueError.
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("rerank limit must be an integer within the candidate count")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        requests = 0
        for offset in range(0, len(candidates), self._batch_size):
            batch = candidates[offset : offset + self._batch_size]
            ranks = range(offset + 1, offset + len(batch) + 1)
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                },
                # A redirect is a non-2xx failure, not another scoring request.
                follow_redirects=False,
            )
            requests += 1
            response.raise_for_status()
            scores.update(_batch_scores(response, set(ranks)))

        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(
            selected_ranks=tuple(selected),
            usage=InferenceUsage(requests=requests),
        )


def _batch_scores(response: httpx.Response, expected_ranks: set[int]) -> dict[int, int | float]:
    payload = json.loads(response.content, object_pairs_hook=_unique_json_object)
    if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
        raise ValueError("reranker response must contain only a scores array")  # noqa: TRY003

    scores: dict[int, int | float] = {}
    for item in payload["scores"]:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each score must contain only rank and score")  # noqa: TRY003
        rank, score = item["rank"], item["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in scores:
            raise ValueError("scores must cover each rank of this batch exactly once")  # noqa: TRY003
        if type(score) not in (int, float) or (isinstance(score, float) and not math.isfinite(score)):
            raise ValueError("score must be a finite number")  # noqa: TRY003
        scores[rank] = score
    if scores.keys() != expected_ranks:
        raise ValueError("scores must cover each rank of this batch exactly once")  # noqa: TRY003
    return scores


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("reranker response contains a duplicate JSON field")  # noqa: TRY003
        result[key] = value
    return result


__all__ = ["HTTPMemoryReranker"]
