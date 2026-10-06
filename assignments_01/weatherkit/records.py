from dataclasses import dataclass
from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """Represents one hourly weather reading.

    Attributes:
        timestamp: Observation timestamp as a string.
        temperature_c: Temperature in degrees Celsius.
        precipitation_mm: Precipitation in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated weather response into hourly reading records.

    Args:
        response: A validated WeatherResponse containing columnar hourly data.

    Returns:
        A list of HourlyReading objects in the same order as the source data.
    """
    return [
        HourlyReading(
            timestamp=response.hourly.time[i],
            temperature_c=response.hourly.temperature_2m[i],
            precipitation_mm=response.hourly.precipitation[i],
        )
        for i in range(len(response.hourly.time))
    ]

# WeatherResponse is a Pydantic model because it sits at the API boundary
# and needs validation of untrusted external data. HourlyReading is a
# dataclass because it is an internal application record created after the
# data has already passed validation at the boundary.

# The deliberate i + 1 change in to_readings() was caught by
# test_reading_values_match_input_indexes.


