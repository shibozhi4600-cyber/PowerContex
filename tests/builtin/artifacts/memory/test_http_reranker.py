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

"""HTTP protocol tests; native Runtime acceptance is tested separately."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hit(rank: int, *, text: str | None = None) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id=f"memory-{rank}", revision=rank),
        entry_id=f"entry-{rank}",
        entry_version_id=f"version-{rank}",
        text=text if text is not None else f"Candidate {rank}",
        score=1 / rank,
        matched_by=("fts",),
    )


def test_globally_sorts_reversed_responses_and_cross_batch_ties_without_changing_hits() -> None:
    candidates = tuple(_hit(rank, text="Same evidence" if rank in (1, 4) else None) for rank in range(1, 6))
    original = tuple(hit.model_dump() for hit in candidates)
    scores = {1: 0.9, 2: 0.8, 3: -1, 4: 0.9, 5: 1}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert set(payload) == {"query", "candidates"}
        assert payload["query"] == "原始 query"
        assert 1 <= len(payload["candidates"]) <= 2
        for item in payload["candidates"]:
            assert set(item) == {"rank", "text"}
            assert item["text"] == candidates[item["rank"] - 1].text
        seen.extend(item["rank"] for item in payload["candidates"])
        return httpx.Response(
            200,
            json={
                "scores": [
                    {"rank": item["rank"], "score": scores[item["rank"]]} for item in reversed(payload["candidates"])
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(
                "https://reranker.invalid/score", client=client, batch_size=2, policy_id="custom"
            )
            decision = await reranker.rerank("原始 query", candidates, 4)
            assert not client.is_closed
        assert reranker.policy_id == "custom"
        assert decision.selected_ranks == (5, 1, 4, 2)
        assert decision.usage.requests == 3
        assert decision.usage.input_tokens is None
        assert decision.usage.output_tokens is None
        assert decision.discarded_rank_count == 0
        assert decision.used_fallback is False

    asyncio.run(scenario())
    assert sorted(seen) == list(range(1, 6))
    assert tuple(hit.model_dump() for hit in candidates) == original


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "2", None])
def test_rejects_invalid_batch_size(batch_size) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=batch_size)

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [0, -1, 2, True, False, 1.0, "1", None])
def test_rejects_invalid_limits_without_sending_requests(limit) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid input must fail before making an HTTP request")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), limit)

    asyncio.run(scenario())


def test_rejects_empty_candidates_without_sending_requests() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("empty candidates must fail before making an HTTP request")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"scores": [], "extra": True},
        {"scores": {}},
        {"scores": None},
        {"scores": []},
        {"scores": [{"rank": 1, "score": 0.5}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.6}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 3, "score": 0.6}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 2}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 2, "score": 0.6, "extra": True}]},
        {"scores": [{"rank": 1, "score": 0.5}, None]},
    ],
)
def test_rejects_schema_errors_and_incomplete_duplicate_or_cross_batch_ranks(payload) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=2)
            with pytest.raises(ValueError):
                await reranker.rerank("q", (_hit(1), _hit(2), _hit(3)), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("rank", [True, False, 1.0, "1", None])
def test_rejects_noninteger_response_ranks(rank) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"scores": [{"rank": rank, "score": 0.5}]})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("score", [True, False, "0.5", None, float("nan"), float("inf"), -float("inf")])
def test_rejects_nonnumeric_or_nonfinite_scores(score) -> None:
    async def scenario() -> None:
        payload = {"scores": [{"rank": 1, "score": score}]}
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "content",
    [
        b"not JSON",
        b'{"scores":[}',
        b'{"scores":[],"scores":[{"rank":1,"score":1}]}',
        b'{"scores":[{"rank":1,"rank":1,"score":1}]}',
        b'{"scores":[{"rank":1,"score":0,"score":1}]}',
    ],
)
def test_rejects_invalid_json_and_duplicate_fields(content: bytes) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=content))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [302, 307, 400, 401, 429, 500])
def test_preserves_http_status_errors_without_following_redirects(status: int) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, headers={"location": "https://another.invalid/score"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            with pytest.raises(httpx.HTTPStatusError) as error:
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), 1)
            assert error.value.response.status_code == status

    asyncio.run(scenario())
    assert len(seen) == 1


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_preserves_transport_errors(error_type) -> None:
    raised = []

    def handler(request: httpx.Request) -> httpx.Response:
        error = error_type("offline", request=request)
        raised.append(error)
        raise error

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(error_type) as error:
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), 1)
            assert error.value is raised[0]

    asyncio.run(scenario())


def test_concurrent_calls_keep_their_scores_and_request_usage_separate() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": item["rank"]} for item in payload["candidates"]]},
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=2)
            first, second = await asyncio.gather(
                reranker.rerank("first", tuple(_hit(rank) for rank in range(1, 4)), 1),
                reranker.rerank("second", tuple(_hit(rank) for rank in range(1, 6)), 2),
            )
            assert first.selected_ranks == (3,)
            assert first.usage.requests == 2
            assert second.selected_ranks == (5, 4)
            assert second.usage.requests == 3

    asyncio.run(scenario())
