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

"""Event-loop-local embedding request coalescing and bounded successful-vector reuse."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel, QueryEmbeddingModel, embed_query


@dataclass(frozen=True, slots=True)
class _EmbeddedText:
    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share pending text computations without sharing caller cancellation.

    A wrapper is shared within one event loop. Provider work is owned by the
    wrapper, while each call independently attributes the batches it awaited.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")  # noqa: TRY003
        if type(coalesce_ms) not in (int, float) or not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a finite nonnegative number")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[bool, str], EmbeddingVector] = OrderedDict()
        self._inflight: dict[tuple[bool, str], asyncio.Future[_EmbeddedText]] = {}
        self._pending: dict[tuple[bool, str], asyncio.Future[_EmbeddedText]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()
        self._batch_id = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Restore input order and bill each depended-on provider batch once."""

        return await self._embed(texts, query=False)

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Keep asymmetric query embeddings distinct; symmetric models reuse the same cache."""

        if not isinstance(self._model, QueryEmbeddingModel):
            return await self.embed(texts)
        return await self._embed(texts, query=True)

    async def _embed(self, texts: tuple[str, ...], *, query: bool) -> EmbeddingResult:
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("embedding inputs must be strings")  # noqa: TRY003
        vectors: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_EmbeddedText]] = {}
        loop = asyncio.get_running_loop()
        # Registration does not yield: all positions join their shared work
        # before the coalescing window or another caller can dispatch it.
        for text in texts:
            key = (query, text)
            if key in self._cache:
                vectors[text] = self._cache[key]
                self._cache.move_to_end(key)
            else:
                future = self._inflight.get(key)
                if future is None:
                    future = loop.create_future()
                    future.add_done_callback(_observe_completion)
                    self._inflight[key] = future
                    self._pending[key] = future
                waiting[text] = future
        if self._pending and (self._dispatcher is None or self._dispatcher.done()):
            self._dispatcher = asyncio.create_task(self._dispatch())

        usage: dict[int, InferenceUsage] = {}
        for text, future in waiting.items():
            result = await asyncio.shield(future)
            vectors[text] = result.vector
            usage[result.batch_id] = result.usage
        return EmbeddingResult(vectors=tuple(vectors[text] for text in texts), usage=_sum_usage(tuple(usage.values())))

    async def _dispatch(self) -> None:
        await asyncio.sleep(self._coalesce_seconds)
        pending = tuple(self._pending.items())
        self._pending.clear()
        self._dispatcher = None
        for query in (False, True):
            # Query-specific providers may return different vectors for the
            # same text. Never mix the two operations in one provider request.
            operation = tuple((key, future) for key, future in pending if key[0] == query)
            for start in range(0, len(operation), self._max_batch_size):
                batch = operation[start : start + self._max_batch_size]
                self._batch_id += 1
                task = asyncio.create_task(self._execute(batch, self._batch_id))
                self._batches.add(task)
                task.add_done_callback(self._batches.discard)

    async def _execute(
        self, batch: tuple[tuple[tuple[bool, str], asyncio.Future[_EmbeddedText]], ...], batch_id: int
    ) -> None:
        try:
            texts = tuple(key[1] for key, _ in batch)
            result = await embed_query(self._model, texts) if batch[0][0][0] else await self._model.embed(texts)
            if len(result.vectors) != len(batch):
                # Schema failures must fan out to exactly this batch's waiters.
                raise ValueError("provider returned a different number of vectors than input texts")  # noqa: TRY003, TRY301
        except (Exception, asyncio.CancelledError) as error:
            for key, future in batch:
                self._inflight.pop(key, None)
                future.set_exception(error)
        else:
            for (key, future), vector in zip(batch, result.vectors, strict=True):
                self._inflight.pop(key, None)
                if self._cache_size:
                    self._cache[key] = vector
                    self._cache.move_to_end(key)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                future.set_result(_EmbeddedText(vector=vector, batch_id=batch_id, usage=result.usage))


def _observe_completion(future: asyncio.Future[_EmbeddedText]) -> None:
    # A caller may have cancelled while the wrapper-owned request completed.
    # Observing here prevents an orphaned exception warning without hiding it
    # from any surviving caller that awaits the same Future.
    if not future.cancelled():
        future.exception()


def _sum_usage(values: tuple[InferenceUsage, ...]) -> InferenceUsage:
    return InferenceUsage(
        requests=sum(item.requests for item in values),
        input_tokens=None
        if all(item.input_tokens is None for item in values)
        else sum(item.input_tokens or 0 for item in values),
        output_tokens=None
        if all(item.output_tokens is None for item in values)
        else sum(item.output_tokens or 0 for item in values),
    )


__all__ = ["CoalescingEmbeddingModel"]
