"""asyncpg repository with per-transaction tenant context (blueprint §7.4).

Every call runs inside a transaction that first executes
``SELECT set_config('app.current_tenant_id', $1, true)`` — the ``true`` makes it
transaction-local (``SET LOCAL``), so a connection returned to the pool can
never leak another tenant's context.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID

from wte.core.schemas import Forecast, PosEvent, ZoneEvent

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Repository:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None

    async def start(self) -> None:
        import asyncpg  # type: ignore

        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()

    async def migrate(self) -> None:
        """Run with the migrator DSN (BYPASSRLS). Idempotent."""
        async with self._pool.acquire() as conn:
            for sql in sorted(MIGRATIONS_DIR.glob("*.sql")):
                await conn.execute(sql.read_text())

    @asynccontextmanager
    async def tenant(self, tenant_id: UUID) -> AsyncIterator[Any]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, true)",
                                   str(tenant_id))
                yield conn

    # ---- writes ----------------------------------------------------------

    async def insert_zone_events(self, events: list[ZoneEvent]) -> None:
        if not events:
            return
        tenant_id = events[0].tenant_id
        async with self.tenant(tenant_id) as conn:
            await conn.executemany(
                "INSERT INTO zone_events (ts, tenant_id, location_id, camera_id, zone, "
                "event_type, tracker_id, dwell_seconds) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                [(e.ts, e.tenant_id, e.location_id, e.camera_id, e.zone.value,
                  e.event_type.value, e.tracker_id, e.dwell_seconds) for e in events])

    async def insert_pos_event(self, e: PosEvent) -> None:
        import json

        async with self.tenant(e.tenant_id) as conn:
            await conn.execute(
                "INSERT INTO pos_events (received_at, occurred_at, tenant_id, location_id, "
                "provider, event_id, event_type, order_id, party_size, channel, raw) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb) ON CONFLICT DO NOTHING",
                e.received_at, e.occurred_at, e.tenant_id, e.location_id, e.provider,
                e.event_id, e.event_type, e.order_id, e.party_size, e.channel, json.dumps(e.raw))

    async def insert_forecast(self, tenant_id: UUID, fc: Forecast) -> None:
        async with self.tenant(tenant_id) as conn:
            await conn.executemany(
                "INSERT INTO forecasts (generated_at, tenant_id, location_id, ts, model, tier, "
                "q10, q50, q90) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
                [(fc.generated_at, tenant_id, fc.location_id, p.ts, fc.model, fc.tier,
                  p.q10, p.q50, p.q90) for p in fc.points])

    # ---- reads -----------------------------------------------------------

    async def wait_series(self, tenant_id: UUID, location_id: str,
                          since: datetime) -> list[tuple[datetime, float]]:
        async with self.tenant(tenant_id) as conn:
            rows = await conn.fetch(
                "SELECT bucket, p50_minutes FROM wait_time_15m "
                "WHERE tenant_id=$1 AND location_id=$2 AND bucket >= $3 ORDER BY bucket",
                tenant_id, location_id, since)
        return [(r["bucket"], float(r["p50_minutes"])) for r in rows if r["p50_minutes"] is not None]

    async def export_tenant(self, tenant_id: UUID) -> dict[str, list[dict]]:
        """Art. 20 GDPR portability: aggregated data only (no per-track events)."""
        async with self.tenant(tenant_id) as conn:
            rows = await conn.fetch("SELECT * FROM wait_time_buckets WHERE tenant_id=$1", tenant_id)
        return {"wait_time_buckets": [dict(r) for r in rows]}
