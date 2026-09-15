from datetime import timedelta

from wte.forecast.features import bucket, calendar_features
from wte.forecast.models import SeasonalNaiveForecaster
from wte.forecast.service import ForecastService, aggregate_to_buckets, tier_for

from tests.conftest import LOCATION, T0, at


def _weekly_pattern(days: int):
    """Synthetic wait series: lunch/dinner peaks, weekends busier; 15-min buckets."""
    out = []
    t = T0 - timedelta(days=days)
    while t < T0:
        m = t.hour * 60 + t.minute
        base = 2.0
        if 690 <= m < 840:      # 11:30–14:00
            base = 12.0
        elif 1080 <= m < 1320:  # 18:00–22:00
            base = 18.0
        if t.weekday() >= 5:
            base *= 1.5
        out.append((t, base))
        t += timedelta(minutes=15)
    return out


def test_bucket_floors_to_15_minutes():
    assert bucket(at(7 * 60 + 12)) == T0  # 12:07:12 → 12:00
    assert bucket(at(15 * 60)) == T0 + timedelta(minutes=15)


def test_calendar_features():
    f = calendar_features(T0)  # Tue 2026-09-15 12:00
    assert f["is_lunch"] == 1.0 and f["is_dinner"] == 0.0
    assert f["is_weekend"] == 0.0 and f["day_of_week"] == 1


def test_tier_selection():
    assert tier_for([]) == "cold"
    assert tier_for(_weekly_pattern(7)) == "cold"
    assert tier_for(_weekly_pattern(30)) == "warm"
    assert tier_for(_weekly_pattern(90), n_locations_in_tenant=3) == "warm"
    assert tier_for(_weekly_pattern(90), n_locations_in_tenant=10) == "mature"


def test_seasonal_naive_learns_weekly_shape():
    m = SeasonalNaiveForecaster()
    m.fit(_weekly_pattern(28))
    pts = m.predict(T0, steps=96, step=timedelta(minutes=15))  # next 24 h from Tue 12:00
    by_minute = {p.ts.hour * 60 + p.ts.minute: p for p in pts}
    assert by_minute[13 * 60].q50 == 12.0   # lunch
    assert by_minute[19 * 60].q50 == 18.0   # dinner
    assert by_minute[3 * 60].q50 == 2.0     # night
    for p in pts:
        assert p.q10 <= p.q50 <= p.q90


def test_aggregate_to_buckets_median():
    obs = [(at(10), 4.0), (at(20), 8.0), (at(30), 6.0), (at(20 * 60), 1.0)]
    series = aggregate_to_buckets(obs)
    assert series[0] == (T0, 6.0)
    assert series[1][1] == 1.0


def test_service_cold_start_produces_7_day_horizon_and_best_windows():
    svc = ForecastService()
    fc = svc.run(LOCATION, _weekly_pattern(5), now=T0)
    assert fc.tier == "cold" and fc.model == "seasonal_naive"
    assert len(fc.points) == 7 * 96
    assert fc.points[0].ts == T0 + timedelta(minutes=15)
    best = fc.best_windows(n=3)
    assert len(best) == 3
    assert all(b.q50 <= 3.0 for b in best)  # off-peak
    assert svc.latest(LOCATION) is fc
