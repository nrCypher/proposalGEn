"""Forecasters behind one interface (blueprint §4.1).

Tiering by data maturity is done in ``service.py``:
- cold  (<14 d)  → ``SeasonalNaiveForecaster`` (dependency-free stand-in for Chronos-2
                   zero-shot; swap in ``ChronosForecaster`` when the GPU pool exists)
- warm  (14–60 d) → ``ProphetForecaster``
- mature (>60 d & >5 sites) → TFT (out of MVP scope; interface kept)

All return quantiles (q10/q50/q90) — customers get bands, never points (§4.3).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from wte.core.schemas import ForecastPoint

Series = list[tuple[datetime, float]]  # (bucket_ts, wait_minutes)


class Forecaster(Protocol):
    name: str

    def fit(self, series: Series) -> None: ...
    def predict(self, start: datetime, steps: int, step: timedelta) -> list[ForecastPoint]: ...


@dataclass
class SeasonalNaiveForecaster:
    """Weekly-seasonal median with empirical quantiles per (weekday, minute-of-day) slot.

    Falls back to (minute-of-day) then to the global distribution when a slot has
    too few observations — so it gives *something* from day 1 (pitfall #10).
    """

    name: str = "seasonal_naive"
    min_slot_obs: int = 3
    _weekly: dict[tuple[int, int], list[float]] = None  # type: ignore[assignment]
    _daily: dict[int, list[float]] = None  # type: ignore[assignment]
    _all: list[float] = None  # type: ignore[assignment]

    def fit(self, series: Series) -> None:
        self._weekly, self._daily, self._all = defaultdict(list), defaultdict(list), []
        for ts, v in series:
            mod = ts.hour * 60 + ts.minute
            self._weekly[(ts.weekday(), mod)].append(v)
            self._daily[mod].append(v)
            self._all.append(v)

    def predict(self, start: datetime, steps: int, step: timedelta) -> list[ForecastPoint]:
        out: list[ForecastPoint] = []
        for i in range(steps):
            ts = start + i * step
            mod = ts.hour * 60 + ts.minute
            obs = self._weekly.get((ts.weekday(), mod), [])
            if len(obs) < self.min_slot_obs:
                obs = self._daily.get(mod, [])
            if len(obs) < self.min_slot_obs:
                obs = self._all or [0.0]
            arr = np.array(obs, dtype=float)
            q10, q50, q90 = (float(np.percentile(arr, q)) for q in (10, 50, 90))
            out.append(ForecastPoint(ts=ts, q10=q10, q50=q50, q90=max(q90, q50)))
        return out


class ProphetForecaster:
    """Facebook Prophet with holiday regressors and an 80 % interval (§4.3)."""

    name = "prophet"

    def __init__(self, country: str = "PT", interval_width: float = 0.8) -> None:
        self._country, self._iw = country, interval_width
        self._m = None

    def fit(self, series: Series) -> None:
        import pandas as pd  # type: ignore
        from prophet import Prophet  # type: ignore

        df = pd.DataFrame({"ds": [t.replace(tzinfo=None) for t, _ in series],
                           "y": [v for _, v in series]})
        m = Prophet(interval_width=self._iw, daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=len(series) > 4 * 96 * 365 // 4)
        try:
            m.add_country_holidays(country_name=self._country)
        except Exception:
            pass
        m.fit(df)
        self._m = m

    def predict(self, start: datetime, steps: int, step: timedelta) -> list[ForecastPoint]:
        import pandas as pd  # type: ignore

        future = pd.DataFrame({"ds": [(start + i * step).replace(tzinfo=None) for i in range(steps)]})
        fc = self._m.predict(future)
        return [
            ForecastPoint(ts=start + i * step, q10=max(0.0, float(r.yhat_lower)),
                          q50=max(0.0, float(r.yhat)), q90=max(0.0, float(r.yhat_upper)))
            for i, r in enumerate(fc.itertuples())
        ]


class ChronosForecaster:
    """Chronos-2 zero-shot (amazon-science/chronos-forecasting). GPU recommended."""

    name = "chronos2"

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu") -> None:
        from chronos import BaseChronosPipeline  # type: ignore

        self._p = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
        self._ctx: np.ndarray | None = None

    def fit(self, series: Series) -> None:
        self._ctx = np.array([v for _, v in series], dtype=np.float32)

    def predict(self, start: datetime, steps: int, step: timedelta) -> list[ForecastPoint]:
        import torch  # type: ignore

        q, _ = self._p.predict_quantiles(context=torch.tensor(self._ctx),
                                         prediction_length=steps, quantile_levels=[0.1, 0.5, 0.9])
        q = q[0].numpy()
        return [ForecastPoint(ts=start + i * step, q10=float(q[i, 0]), q50=float(q[i, 1]),
                              q90=float(q[i, 2])) for i in range(steps)]
