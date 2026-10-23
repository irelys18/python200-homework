
import json
from pathlib import Path


# Question 1
class ModelInfo:
    def __init__(self, path: str):
        """Load model information from a JSON file."""
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Return the number of features in the model."""
        return len(self.data["features"])


# Create a small JSON file to use for testing.
metadata_path = Path("model_info.json")

sample_data = {
    "features": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(sample_data, file, indent=4)

# Create the object and display the feature count.
model_info = ModelInfo(metadata_path)
print("Number of features:", model_info.feature_count())

# Loading the file once in __init__ avoids reading it from disk
# every time feature_count() is called. This is more efficient.


# Question 2
from dataclasses import dataclass

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Return a readable summary of the day's forecast."""
        if self.will_rain:
            rain_message = "rain expected"
        else:
            rain_message = "no rain expected"

        return f"{self.date}: high {self.high_c}C, {rain_message}"


# Create two DayForecast objects.
forecast1 = DayForecast("2023-06-15", 24.1, True)
forecast2 = DayForecast("2023-06-16", 27.5, False)

# Print each forecast summary.
print(forecast1.summary())
print(forecast2.summary())

# The @dataclass decorator automatically generated the __init__()
# method, so I did not have to write it by hand.


# Question 3
import pytest


def parse_temperature(text: str) -> float:
    """Convert a temperature string to a float."""
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature: {text}")


def test_parse_temperature_valid():
    """Test that a valid temperature string converts correctly."""
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    """Test that invalid temperature text raises ValueError."""
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, pytest.raises(ValueError) would not catch an error
# message that has the wrong content, as long as its type is ValueError.
