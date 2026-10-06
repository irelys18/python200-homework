import json
from pathlib import Path

from weatherkit import WeatherResponse
from weatherkit.records import HourlyReading, to_readings


def load_weather_response():
    """Load and validate the real weather JSON file."""
    path = Path(__file__).parent.parent / "weather_raw.json"

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return WeatherResponse.model_validate(data)


def test_to_readings_returns_one_reading_per_hour():
    """to_readings should preserve the number and order of observations."""
    response = load_weather_response()

    readings = to_readings(response)

    assert len(readings) == 168
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


def test_reading_values_match_input_indexes():
    """Each reading should contain values from the same source index."""
    response = load_weather_response()

    readings = to_readings(response)

    for i, reading in enumerate(readings):
        assert reading.timestamp == response.hourly.time[i]
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]


def test_hourly_readings_compare_equal():
    """HourlyReading objects with identical fields should compare equal."""
    reading_a = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    reading_b = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    assert reading_a == reading_b