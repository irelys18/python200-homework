
from dataclasses import dataclass

# Task 1
@dataclass
class Prediction:
    """Store the weather classification and probability for a day.

    Attributes:
        label: The prediction, either "good" or "skip".
        probability: The probability that the day is good for running,
            represented as a number between 0.0 and 1.0.
    """

    label: str
    probability: float


# Task 2
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class Prediction:
    """Store a weather classification and its probability.

    Attributes:
        label: Either "good" or "skip".
        probability: Probability that the day is good for running,
            between 0.0 and 1.0.
    """

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved weather model and predict running conditions."""

    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str | Path) -> None:
        """Load the saved machine learning pipeline.

        Args:
            model_path: Path to the saved .pkl model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {path}"
            )

        self.model: Any = joblib.load(path)

    def predict(
        self, days: list[dict[str, float]]
    ) -> list[Prediction]:
        """Predict whether each day's weather is good for running.

        Args:
            days: List of dictionaries containing the four required
                weather features.

        Returns:
            A list of Prediction objects in the same order as the input.

        Raises:
            ValueError: If any day is missing a required feature.
        """
        if not days:
            return []

        # Check that every day contains all required features.
        for index, day in enumerate(days):
            missing = [
                feature
                for feature in self.FEATURES
                if feature not in day
            ]

            if missing:
                raise ValueError(
                    f"Day {index} is missing required features: "
                    f"{', '.join(missing)}"
                )

        # Select features in the exact order expected by the model.
        weather_data = pd.DataFrame(
            days, columns=self.FEATURES
        )

        # Get predictions and probabilities from the saved pipeline.
        labels = self.model.predict(weather_data)
        probabilities = self.model.predict_proba(weather_data)[:, 1]

        # Convert raw model outputs into Prediction objects.
        results = []

        for label, probability in zip(labels, probabilities):
            results.append(
                Prediction(
                    label="good" if label == 1 else "skip",
                    probability=float(probability),
                )
            )

        return results


# Task 3
from .classifier import Prediction, WeatherClassifier

__all__ = ["WeatherClassifier", "Prediction"]
