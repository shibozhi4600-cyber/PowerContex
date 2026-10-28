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

"""Embedding coalescing regressions across independent caller lifecycles."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel


class _ControlledModel:
    profile = EmbeddingProfile(profile_id="controlled", model="offline", dimension=1)

    def __init__(self) -> None:
        self.started: asyncio.Queue[tuple[str, ...]] = asyncio.Queue()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        self.calls.append(texts)
        self.started.put_nowait(texts)
        await self.release.wait()
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=2, input_tokens=7, output_tokens=0),
        )


def test_late_waiter_shares_running_work_with_cache_disabled_and_retains_batch_usage() -> None:
    async def scenario() -> None:
        base = _ControlledModel()
        model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=0, cache_size=0)
        first = asyncio.create_task(model.embed(("shared", "first", "shared")))
        assert await base.started.get() == ("shared", "first")
        late = asyncio.create_task(model.embed(("second", "shared", "second")))
        assert await base.started.get() == ("second",)
        base.release.set()
        first_result, late_result = await asyncio.gather(first, late)
        assert first_result.vectors == ((6.0,), (5.0,), (6.0,))
        assert late_result.vectors == ((6.0,), (6.0,), (6.0,))
        assert first_result.usage == InferenceUsage(requests=2, input_tokens=7, output_tokens=0)
        assert late_result.usage == InferenceUsage(requests=4, input_tokens=14, output_tokens=0)
        await model.embed(("shared",))
        assert base.calls == [("shared", "first"), ("second",), ("shared",)]

    asyncio.run(scenario())


def test_failed_batch_does_not_block_or_evict_independent_successful_work() -> None:
    class IndependentModel:
        profile = _ControlledModel.profile

        def __init__(self) -> None:
            self.calls = []

        async def embed(self, texts):
            self.calls.append(texts)
            if "bad" in texts:
                raise RuntimeError("controlled provider failure")  # noqa: TRY003
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts), usage=InferenceUsage(requests=3))

    async def scenario() -> None:
        base = IndependentModel()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        failed, successful = await asyncio.gather(model.embed(("bad",)), model.embed(("good",)), return_exceptions=True)
        assert isinstance(failed, RuntimeError)
        assert isinstance(successful, EmbeddingResult)
        assert successful.vectors == ((1.0,),)
        assert successful.usage.requests == 3
        assert (await model.embed(("good",))).usage.requests == 0
        with pytest.raises(RuntimeError, match="controlled provider failure"):
            await model.embed(("bad",))
        assert base.calls.count(("bad",)) == 2
        assert base.calls.count(("good",)) == 1

    asyncio.run(scenario())


def test_cancelling_all_original_callers_still_allows_a_late_survivor() -> None:
    async def scenario() -> None:
        base = _ControlledModel()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        first = asyncio.create_task(model.embed(("shared",)))
        await base.started.get()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        survivor = asyncio.create_task(model.embed(("shared", "shared")))
        base.release.set()
        result = await survivor
        assert result.vectors == ((6.0,), (6.0,))
        assert result.usage.requests == 2
        assert base.calls == [("shared",)]

    asyncio.run(scenario())


def test_invalid_cardinality_is_not_partially_cached_and_can_be_retried() -> None:
    class IncorrectOnce:
        profile = _ControlledModel.profile

        def __init__(self) -> None:
            self.calls = []

        async def embed(self, texts):
            self.calls.append(texts)
            vectors = ((1.0,),) if len(self.calls) == 1 else tuple((2.0,) for _ in texts)
            return EmbeddingResult(vectors=vectors, usage=InferenceUsage(requests=1))

    async def scenario() -> None:
        base = IncorrectOnce()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        with pytest.raises(ValueError):
            await model.embed(("one", "two"))
        result = await model.embed(("one", "two"))
        assert result.vectors == ((2.0,), (2.0,))
        assert result.usage.requests == 1
        assert base.calls == [("one", "two"), ("one", "two")]

    asyncio.run(scenario())


def test_asymmetric_query_embeddings_keep_their_operation_and_cache_identity() -> None:
    class AsymmetricModel:
        profile = _ControlledModel.profile

        def __init__(self) -> None:
            self.calls = []

        async def embed(self, texts):
            self.calls.append(("document", texts))
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts), usage=InferenceUsage(requests=1))

        async def embed_query(self, texts):
            self.calls.append(("query", texts))
            return EmbeddingResult(vectors=tuple((2.0,) for _ in texts), usage=InferenceUsage(requests=1))

    async def scenario() -> None:
        base = AsymmetricModel()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        document, query = await asyncio.gather(model.embed(("same",)), model.embed_query(("same", "same")))
        assert document.vectors == ((1.0,),)
        assert query.vectors == ((2.0,), (2.0,))
        assert (await model.embed(("same",))).usage.requests == 0
        assert (await model.embed_query(("same",))).usage.requests == 0
        assert base.calls == [("document", ("same",)), ("query", ("same",))]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "settings",
    [
        {"max_batch_size": True},
        {"max_batch_size": 0},
        {"max_batch_size": 1.0},
        {"cache_size": False},
        {"cache_size": -1},
        {"cache_size": "2"},
        {"coalesce_ms": True},
        {"coalesce_ms": -0.1},
        {"coalesce_ms": float("nan")},
        {"coalesce_ms": float("inf")},
    ],
)
def test_invalid_window_and_capacity_configuration_fails_before_scheduling(settings) -> None:
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(_ControlledModel(), **settings)
