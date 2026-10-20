import json

from weatherkit import DailyAggregator, WeatherResponse
from weatherkit.records import to_readings


def main() -> None:
    """Load, validate, convert, summarize, and report weather data."""
    with open("weather_raw.json", "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # Validate the raw API data before converting it into application records.
    weather = WeatherResponse.model_validate(raw_data)

    readings = to_readings(weather)

    aggregator = DailyAggregator()

    summaries = aggregator.summarize(readings)
    incomplete_days = aggregator.incomplete_days(readings)

    print("Date        High   Low   Precipitation   Range")
    print("-" * 50)

    for summary in summaries:
        print(
            f"{summary.date}  "
            f"{summary.temp_max:5.1f}  "
            f"{summary.temp_min:5.1f}  "
            f"{summary.precipitation_sum:13.1f}  "
            f"{summary.temp_range():5.1f}"
        )

    if incomplete_days:
        print(f"Warning: incomplete days dropped: {', '.join(incomplete_days)}")
    else:
        print("Warning: incomplete days dropped: None")


if __name__ == "__main__":
    # Without this guard, main() would run automatically whenever report.py
    # was imported, causing the report to print even if someone only wanted
    # to reuse a helper function from this module.
    main()


# Reflection
#
# 1. Rejecting the whole file when one temperature is null can be useful when
#    data quality is critical and we need to know that the source data is
#    complete and trustworthy. For example, I would want this behavior for
#    a report where missing temperatures could make the final results
#    misleading. However, I would rather tolerate a missing temperature if
#    one sensor failed for a short period but the rest of the day's weather
#    data is still useful. To tolerate the gap, I could allow None values in
#    temperature_2m by changing its type from list[float] to list[float | None].
#    The pipeline would then need to handle those None values when calculating
#    daily minimums, maximums, and other summaries.
#
# 2. If a pipeline runs at noon, the current day may only have about 12 hourly
#    observations instead of the expected 24. With min_hours set to 24,
#    DailyAggregator will drop that day from the summaries. incomplete_days()
#    helps by identifying the date that was dropped because it did not have
#    enough observations. This lets the pipeline report the incomplete day
#    instead of silently leaving it out.
#
# 3. Using weatherkit as a package makes it easier for a future pipeline to
#    import specific functionality from separate modules. For example, in
#    Week 10 a pipeline could import DailyAggregator from weatherkit and use
#    it to summarize new weather data without copying the aggregation code
#    into another script.