import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def sample_readings():
    """Provide shared hourly readings spanning two dates."""
    return [
        HourlyReading(
            timestamp="2026-04-08T00:00",
            temperature_c=10.0,
            precipitation_mm=1.0,
        ),
        HourlyReading(
            timestamp="2026-04-08T01:00",
            temperature_c=15.0,
            precipitation_mm=2.0,
        ),
        HourlyReading(
            timestamp="2026-04-08T02:00",
            temperature_c=12.0,
            precipitation_mm=0.5,
        ),
        HourlyReading(
            timestamp="2026-04-09T00:00",
            temperature_c=8.0,
            precipitation_mm=3.0,
        ),
        HourlyReading(
            timestamp="2026-04-09T01:00",
            temperature_c=20.0,
            precipitation_mm=1.0,
        ),
    ]


def test_grouping_produces_two_summaries(sample_readings):
    """Readings from two dates should produce two daily summaries."""
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temperature_max_and_min(sample_readings):
    """Daily maximum and minimum temperatures should be correct."""
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_max == 15.0
    assert summaries[0].temp_min == 10.0
    assert summaries[1].temp_max == 20.0
    assert summaries[1].temp_min == 8.0


def test_precipitation_sum(sample_readings):
    """Daily precipitation should be added correctly."""
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(3.5)
    assert summaries[1].precipitation_sum == pytest.approx(4.0)


def test_incomplete_day_is_dropped(sample_readings):
    """A day below min_hours should be excluded and reported as incomplete."""
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-04-08"
    assert "2026-04-09" in incomplete


@pytest.mark.parametrize("min_hours", [1, 2])
def test_lower_min_hours_keeps_same_day(sample_readings, min_hours):
    """Lowering min_hours should allow the shorter day to be included."""
    aggregator = DailyAggregator(min_hours=min_hours)

    summaries = aggregator.summarize(sample_readings)

    dates = [summary.date for summary in summaries]

    assert "2026-04-09" in dates