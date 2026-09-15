"""Structured VLM client (blueprint §3.3–3.4).

Talks to any OpenAI-compatible endpoint: self-hosted vLLM (EU privacy tier,
``guided_json``), OpenAI, or Gemini's OpenAI-compatible surface. The keyframe
sent MUST already be face-blurred — the edge guarantees that; we re-assert it.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from wte.core.schemas import VlmAssessment

ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queue_state": {"type": "string", "enum": ["short", "normal", "long", "gridlocked"]},
        "estimated_wait_minutes": {"type": "number"},
        "bottleneck": {
            "type": "object",
            "properties": {
                "location": {"type": "string",
                             "enum": ["counter", "kitchen", "fulfillment", "none"]},
                "evidence": {"type": "string"},
            },
            "required": ["location", "evidence"],
        },
        "operational_recommendation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["queue_state", "estimated_wait_minutes", "bottleneck", "confidence"],
}

SYSTEM_PROMPT = (
    "You are a restaurant operations analyst. The image shows a queue area in a "
    "restaurant. Faces have been blurred for privacy. Use posture, group size, distance "
    "to counter, and fulfillment activity to assess flow. Be concise. Never speculate "
    "about identity, age, gender or any personal characteristic. Output strictly the "
    "JSON schema."
)


def encode_jpeg(frame_bgr: Any, quality: int = 80) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@dataclass
class VlmClient:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 400
    _client: Any = None

    def _openai(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore

            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    async def assess(self, keyframe_data_url: str, metadata_summary: str,
                     faces_blurred: bool = True) -> VlmAssessment:
        if not faces_blurred:
            raise ValueError("Refusing to send an un-blurred keyframe to a VLM (§5.2)")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": keyframe_data_url}},
                {"type": "text", "text": metadata_summary},
            ]},
        ]
        extra: dict[str, Any] = {}
        if "openai.com" not in self.base_url:
            extra["extra_body"] = {"guided_json": ASSESSMENT_SCHEMA}  # vLLM
        else:
            extra["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "assessment", "schema": ASSESSMENT_SCHEMA, "strict": True}}
        resp = await self._openai().chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens, **extra)
        return parse_assessment(resp.choices[0].message.content or "{}")


def parse_assessment(text: str) -> VlmAssessment:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    return VlmAssessment.model_validate(json.loads(text))
