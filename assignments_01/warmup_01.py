from dataclasses import dataclass, FrozenInstanceError, field
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest

# --- Classes ---
# Q1
class Thermometer:
    
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None

        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None

        return max(self.readings)

    def __repr__(self):
        return (
            f"Thermometer(location='{self.location}', "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


thermometer = Thermometer("San Juan")

thermometer.add(27)
thermometer.add(29)
thermometer.add(30)
thermometer.add(28)

print("Average temperature:", thermometer.average())
print("Hottest temperature:", thermometer.hottest())

# Comment:
# average() needs to handle the empty case because dividing by the
# number of readings when there are zero readings would cause a
# ZeroDivisionError.

print(thermometer)

thermometer2 = Thermometer("Charlotte")

thermometer2.add(17)
thermometer2.add(18)
thermometer2.add(19)
thermometer2.add(20)

print([thermometer, thermometer2])

# Comment:
# Without __repr__, Python displays a default representation such as
# <__main__.Thermometer object at 0x...>. This is unhelpful for debugging
# because it shows the class and memory address, but not the object's data.

class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [reading for reading in thermometer.readings
                if reading > self.threshold]

alert1 = TemperatureAlert(28.0)
alert2 = TemperatureAlert(29.0)

print("Readings above 28.0:", alert1.breaches(thermometer))
print("Readings above 29.0:", alert2.breaches(thermometer))

# Comment:
# The threshold is stored on TemperatureAlert so the same alert rule
# can be reused for many thermometers without passing the threshold
# every time breaches() is called. This is especially useful when
# checking twenty thermometers because each alert object remembers
# its own threshold.

@dataclass(frozen=True)
class Station:
    """Represents a weather station and its geographic information."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station("S001", "San Juan Station", 18.4655, -66.1057, 10.0)
station_b = Station("S001", "San Juan Station", 18.4655, -66.1057, 10.0)

print(station_a == station_b)

# Comment:
# Dataclassess automatically generate an __eq__ method that compares
# the field values of two objects, so identical Station objects are equal.
# With the original hand-written class, Python would compare objects identity
# by default, so station_a == station_b would be False.

try:
    station_a.name = "New Name"
except FrozenInstanceError as error:
    print("FrozenInstanceError:", error)

station_c = Station("S002", "Charlotte Station", 35.2271, -80.8431, 229.0)

stations = {station_a, station_b, station_c}

print("Number of stations in set:", len(stations))

# Comment:
# frozen=True makes the dataclass immutable and also makes it hashable
# when its fields are hashable. This allows Station objects to be sused
# in sets, where identical stations are treated as the same value.

@dataclass
class StationBatch:
    """Represents a group of the weather stations in a region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)

# First attempt produced:
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

# Python refuses [] as a dataclass default because lists are mutable.
# Using default_factory=list creates a new empty list for each StationBatch,
# so different batches do not accidentally share the same list.

# Pydantic Q1
class Reading(BaseModel):
    """Represents a validated weather reading."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)


# Field constraints validate individual fields, but this rule depends on
# the relationship between humidity and temperature_c, so a model_validator
# is needed to check both values together.

    @model_validator(mode="after")
    def check_sensor_failure(self):
        """Reject a combination that indicates a failed sensor."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Humidity 0.0 with temperature below -40 indicates a failed sensor.")
        return self

reading = Reading(
    station_id="S001",
    timestamp="2026-10-05T15:00:00",
    temperature_c=28.5,
    humidity=75.0,
)

print(reading)

# Q2
try:
    Reading(
        station_id="S001",
        temperature_c=25.0,
        humidity=60.0,
    )
except ValidationError as error:
    print("Missing field error:")
    print(error)

try:
    Reading(
        station_id="S001",
        timestamp="2026-10-05T15:00:00",
        temperature_c=150.0,
        humidity=60.0,
    )
except ValidationError as error:
    print("Temperature error:")
    print(error)

try:
    Reading(
        station_id="S001",
        timestamp="2026-10-05T15:00:00",
        temperature_c=25.0,
        humidity="very humid",
    )
except ValidationError as error:
    print("Humidity error:")
    print(error)


converted_reading = Reading(
    station_id="S001",
    timestamp="2026-10-05T15:00:00",
    temperature_c="21.5",
    humidity=40,
)

print(converted_reading)
print(type(converted_reading.temperature_c))
print(type(converted_reading.humidity))

# Pydantic accepts values that can be safely converted to the declared type.
# "21.5" can be converted to a float, but "very humid" cannot, so Pydantic
# rejects the value that cannot be converted to the required type.

try:
    Reading(
        station_id="S1",
        temperature_c="not a number",
        humidity=50.0,
    )
except ValidationError as error:
    for item in error.errors():
        print("Location:", item["loc"])
        print("Message:", item["msg"])


# Three errors were reported. Reporting all errors at once is more useful
# because it lets us fix multiple problems in the data at the same time
# instead of fixing one error, running the program again, and discovering
# another error.


valid_reading = Reading(
    station_id="S001",
    timestamp="2026-10-05T15:00:00",
    temperature_c=-35.0,
    humidity=0.0,
)

print("Valid reading:", valid_reading)

try:
    Reading(
        station_id="S001",
        timestamp="2026-10-05T15:00:00",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as error:
    print("Sensor failure error:")
    print(error)

# Pytest Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212

    # Floating-point calculations can produce tiny rounding differences,
    # so pytest.approx checks whether the result is close enough to 98.6.
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a list of values."""
    if not values:
        raise ValueError("values cannot be empty")
    return sum(values) / len(values)

def test_mean_of_empty_raises():
    # pytest.raises(ValueError) alone checks only the exception type.
    # match= also checks that the error message contains the expected word.
    with pytest.raises(ValueError, match="empty"):
        mean([])


#Q3
@pytest.mark.parametrize(
    "values, expected",
    [
        ([10], 10),
        ([10, 20], 15),
        ([1, 2, 3, 4], 2.5),
        ([-10, -20, -30], -20),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected

# 5 passed in 0.XXs

# One parametrized test is better than four nearly identical test functions
# because it avoids duplicated test code while still testing multiple cases.
# It also makes it easier to add or change test cases in one place.


#Q4
# Deliberataly broken test output:
# E       assert 257.0 == 212
# Pytest showed the actual value produced by the function and the expected
# value, such as 257.0 versus 212. This is more useful than "assertion failed"
# because it shows exactly what the function returned and what the test expected,
# which helps identify the problem in the calculation.



