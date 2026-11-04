
from pathlib import Path

from weather_model import WeatherClassifier


def main() -> None:
    """Predict running conditions for four hypothetical days."""

    # Locate the saved model relative to this script.
    model_path = Path(__file__).parent / "models" / "weather_classifier.pkl"
    classifier = WeatherClassifier(model_path)

    # Four hypothetical days: good, bad, and borderline conditions.
    days = [
        {
            "date": "Sunny and mild",
            "temperature_2m_max": 24.0,
            "temperature_2m_min": 15.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        },
        {
            "date": "Cold, rainy, and windy",
            "temperature_2m_max": 2.0,
            "temperature_2m_min": -5.0,
            "precipitation_sum": 12.0,
            "wind_speed_10m_max": 35.0,
        },
        {
            "date": "Warm with some rain",
            "temperature_2m_max": 27.0,
            "temperature_2m_min": 18.0,
            "precipitation_sum": 2.0,
            "wind_speed_10m_max": 20.0,
        },
        {
            "date": "Cool and breezy",
            "temperature_2m_max": 14.0,
            "temperature_2m_min": 5.0,
            "precipitation_sum": 1.0,
            "wind_speed_10m_max": 28.0,
        },
    ]

    # Keep the descriptions separate from the features expected by the model.
    descriptions = [day["date"] for day in days]
    weather_inputs = [
        {
            key: value
            for key, value in day.items()
            if key != "date"
        }
        for day in days
    ]

    predictions = classifier.predict(weather_inputs)

    print("Weather Running Predictions")
    print("-" * 65)

    for description, weather, prediction in zip(
        descriptions, weather_inputs, predictions
    ):
        print(
            f"{description}: "
            f"High={weather['temperature_2m_max']}°C, "
            f"Low={weather['temperature_2m_min']}°C, "
            f"Rain={weather['precipitation_sum']} mm, "
            f"Wind={weather['wind_speed_10m_max']} km/h | "
            f"Prediction={prediction.label}, "
            f"Probability={prediction.probability:.1%}"
        )


if __name__ == "__main__":
    main()

# The guard prevents main() from running automatically when another
# script imports this module to reuse a function or class.



# Task 6: Reflection
#
# 1. Selecting the features in a fixed order prevents a bug where
#    temperature values could be interpreted as wind speed or rain
#    because the caller supplied the columns in a different order.
#    This ensures the model receives the features in the order it
#    expects.
#
# 2. Training a tiny model in a test fixture makes tests independent
#    of the real saved model file. Tests are more reliable and
#    repeatable because they create their own test data and do not
#    depend on a separate file being present or updated.
#
# 3. WeatherClassifier is ready for a pipeline because it accepts
#    batches of weather records and returns structured Prediction
#    objects. Before production, I would add logging and error
#    handling to help monitor failures and diagnose problems.
