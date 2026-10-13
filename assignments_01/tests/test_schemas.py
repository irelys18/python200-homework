import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def test_valid_response():
    """The real weather response should validate successfully."""
    # A plain relative path depends on the directory where pytest is started,
    # while this path is based on the test file's location and works from any
    # working directory.
    path = Path(__file__).parent.parent / "weather_raw.json"

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude():
    """A latitude outside the valid range should raise ValidationError."""
    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [15.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_hourly_lengths():
    """Mismatched hourly columns should raise ValidationError."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [15.0, 14.5],
            "precipitation": [0.0, 0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature():
    """A null temperature should fail validation."""
    data = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
            ],
            "temperature_2m": [15.0, None],
            "precipitation": [0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)