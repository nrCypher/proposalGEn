"""Event bus abstraction.

Production: Kafka / MSK (blueprint §4.5). Dev & tests: in-memory. Both expose the
same tiny interface so services never import a broker client directly.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel

Handler = Callable[[str, BaseModel], Awaitable[None]]

TOPIC_ZONE_EVENTS = "zone_events"
TOPIC_POS_EVENTS = "pos_events"
TOPIC_BOOKING_EVENTS = "booking_events"
TOPIC_WAIT_ESTIMATES = "wait_estimates"
TOPIC_VLM_TRIGGERS = "vlm_triggers"
TOPIC_VLM_ASSESSMENTS = "vlm_assessments"


class EventBus(Protocol):
    async def publish(self, topic: str, event: BaseModel) -> None: ...
    def subscribe(self, topic: str, handler: Handler) -> None: ...


class InMemoryBus:
    """Synchronous fan-out in-process bus. Deterministic — ideal for tests."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self.published: list[tuple[str, BaseModel]] = []

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    async def publish(self, topic: str, event: BaseModel) -> None:
        self.published.append((topic, event))
        for h in list(self._handlers.get(topic, [])):
            await h(topic, event)


class KafkaBus:
    """Thin aiokafka wrapper. Import is lazy so the package works without Kafka."""

    def __init__(self, bootstrap_servers: str, group_id: str = "wte") -> None:
        self._bootstrap = bootstrap_servers
        self._group = group_id
        self._producer: Any = None
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer  # type: ignore

        self._producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap)
        await self._producer.start()
        for topic, handlers in self._handlers.items():
            consumer = AIOKafkaConsumer(topic, bootstrap_servers=self._bootstrap,
                                        group_id=self._group)
            await consumer.start()
            self._tasks.append(asyncio.create_task(self._consume(consumer, topic, handlers)))

    async def _consume(self, consumer: Any, topic: str, handlers: list[Handler]) -> None:
        from wte.core.registry import model_for_topic

        async for msg in consumer:
            payload = json.loads(msg.value)
            event = model_for_topic(topic).model_validate(payload)
            for h in handlers:
                await h(topic, event)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    async def publish(self, topic: str, event: BaseModel) -> None:
        await self._producer.send_and_wait(topic, event.model_dump_json().encode())

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._producer:
            await self._producer.stop()
