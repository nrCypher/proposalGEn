"""Topic → model mapping used by broker adapters to deserialise messages."""

from __future__ import annotations

from pydantic import BaseModel

from wte.core import bus
from wte.core.schemas import (
    BookingEvent,
    PosEvent,
    VlmAssessment,
    VlmTrigger,
    WaitEstimate,
    ZoneEvent,
)

_TOPIC_MODELS: dict[str, type[BaseModel]] = {
    bus.TOPIC_ZONE_EVENTS: ZoneEvent,
    bus.TOPIC_POS_EVENTS: PosEvent,
    bus.TOPIC_BOOKING_EVENTS: BookingEvent,
    bus.TOPIC_WAIT_ESTIMATES: WaitEstimate,
    bus.TOPIC_VLM_TRIGGERS: VlmTrigger,
    bus.TOPIC_VLM_ASSESSMENTS: VlmAssessment,
}


def model_for_topic(topic: str) -> type[BaseModel]:
    return _TOPIC_MODELS[topic]
