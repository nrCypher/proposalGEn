"""Feature engineering for the 15-minute forecast buckets (blueprint §4.2).

Calendar features are dependency-free. Weather / events / POS features are
plugged in via callables so the service can run without external APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

STEP = timedelta(minutes=15)


def bucket(ts: datetime, step: timedelta = STEP) -> datetime:
    secs = int(step.total_seconds())
    floored = (int(ts.timestamp()) // secs) * secs
    return datetime.fromtimestamp(floored, tz=ts.tzinfo)


def calendar_features(ts: datetime, country: str = "PT") -> dict[str, float]:
    f: dict[str, float] = {
        "minute_of_day": ts.hour * 60 + ts.minute,
        "day_of_week": ts.weekday(),
        "week_of_year": ts.isocalendar().week,
        "is_weekend": float(ts.weekday() >= 5),
        "is_month_end": float((ts + timedelta(days=1)).month != ts.month),
        "is_lunch": float(11 * 60 + 30 <= ts.hour * 60 + ts.minute < 14 * 60),
        "is_dinner": float(18 * 60 <= ts.hour * 60 + ts.minute < 22 * 60),
    }
    f["is_holiday"] = float(_is_holiday(ts, country))
    return f


def _is_holiday(ts: datetime, country: str) -> bool:
    try:
        import holidays  # type: ignore

        return ts.date() in holidays.country_holidays(country, years=ts.year)
    except Exception:  # library missing → conservative default
        return False


WeatherFn = Callable[[datetime], dict[str, float]]
EventsFn = Callable[[datetime], dict[str, float]]


@dataclass
class FeatureBuilder:
    country: str = "PT"
    weather: WeatherFn | None = None
    events: EventsFn | None = None
    extra: list[Callable[[datetime], dict[str, float]]] = field(default_factory=list)

    def build(self, ts: datetime) -> dict[str, float]:
        f = calendar_features(ts, self.country)
        if self.weather:
            f.update({f"wx_{k}": v for k, v in self.weather(ts).items()})
        if self.events:
            f.update({f"ev_{k}": v for k, v in self.events(ts).items()})
        for fn in self.extra:
            f.update(fn(ts))
        return f
