
import joblib
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from weather_model import Prediction, WeatherClassifier


@pytest.fixture
def classifier(tmp_path):
    """Create and load a small model for testing."""
    features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    X_train = pd.DataFrame(
        [
            [25.0, 15.0, 0.0, 10.0],
            [-2.0, -8.0, 10.0, 40.0],
        ],
        columns=features,
    )
    y_train = [1, 0]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "test_weather_model.pkl"
    joblib.dump(model, model_path)

    return WeatherClassifier(model_path)


@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 25.0,
                "temperature_2m_min": 15.0,
                "precipitation_sum": 0.0,
                "wind_speed_10m_max": 10.0,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": -2.0,
                "temperature_2m_min": -8.0,
                "precipitation_sum": 10.0,
                "wind_speed_10m_max": 40.0,
            },
            "skip",
        ),
    ],
)
def test_labels_for_good_and_bad_days(classifier, day, expected_label):
    # This test should fail if every prediction is "skip".
    result = classifier.predict([day])
    assert result[0].label == expected_label


def test_probability_is_between_zero_and_one(classifier):
    day = {
        "temperature_2m_max": 25.0,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0,
    }

    result = classifier.predict([day])

    assert 0.0 <= result[0].probability <= 1.0


def test_three_days_return_predictions_in_order(classifier):
    good_day = {
        "temperature_2m_max": 25.0,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0,
    }
    bad_day = {
        "temperature_2m_max": -2.0,
        "temperature_2m_min": -8.0,
        "precipitation_sum": 10.0,
        "wind_speed_10m_max": 40.0,
    }

    results = classifier.predict([good_day, bad_day, good_day])

    assert len(results) == 3
    assert all(isinstance(item, Prediction) for item in results)
    assert [item.label for item in results] == ["good", "skip", "good"]


def test_empty_days_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises_value_error(classifier):
    incomplete_day = {
        "temperature_2m_max": 25.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0,
    }

    with pytest.raises(ValueError, match="temperature_2m_min"):
        classifier.predict([incomplete_day])


def test_missing_model_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(FileNotFoundError, match="not found"):
        WeatherClassifier(missing_path)
