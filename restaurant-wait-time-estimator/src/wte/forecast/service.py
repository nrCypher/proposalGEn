"""Forecast service: tier selection + 7-day, 15-min horizon (blueprint §4.1).

Nightly (or on demand) per location: pick the tier from data age, fit, predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from wte.core.schemas import Forecast
from wte.forecast.features import STEP, bucket
from wte.forecast.models import Forecaster, ProphetForecaster, SeasonalNaiveForecaster, Series

Tier = Literal["cold", "warm", "mature"]


def tier_for(series: Series, n_locations_in_tenant: int = 1) -> Tier:
    if not series:
        return "cold"
    age_days = (series[-1][0] - series[0][0]).days
    if age_days < 14:
        return "cold"
    if age_days < 60 or n_locations_in_tenant <= 5:
        return "warm"
    return "mature"


def make_forecaster(tier: Tier, country: str = "PT") -> Forecaster:
    if tier == "cold":
        return SeasonalNaiveForecaster()
    try:
        return ProphetForecaster(country=country)
    except ImportError:  # prophet not installed → still serve something
        return SeasonalNaiveForecaster()


def aggregate_to_buckets(observations: list[tuple[datetime, float]]) -> Series:
    """Collapse raw (ts, wait_minutes) observations into 15-min bucket medians."""
    import numpy as np

    by_bucket: dict[datetime, list[float]] = {}
    for ts, v in observations:
        by_bucket.setdefault(bucket(ts), []).append(v)
    return sorted((b, float(np.median(v))) for b, v in by_bucket.items())


@dataclass
class ForecastService:
    horizon_days: int = 7
    step: timedelta = STEP
    country: str = "PT"
    _cache: dict[str, Forecast] = field(default_factory=dict)

    def run(self, location_id: str, series: Series, now: datetime,
            n_locations_in_tenant: int = 1) -> Forecast:
        tier = tier_for(series, n_locations_in_tenant)
        model = make_forecaster(tier, self.country)
        model.fit(series)
        start = bucket(now) + self.step
        steps = int(self.horizon_days * 24 * 60 / (self.step.total_seconds() / 60))
        points = model.predict(start, steps, self.step)
        fc = Forecast(location_id=location_id, generated_at=now, model=model.name, tier=tier,
                      step_minutes=int(self.step.total_seconds() // 60), points=points)
        self._cache[location_id] = fc
        return fc

    def latest(self, location_id: str) -> Forecast | None:
        return self._cache.get(location_id)
