
import json
import joblib
import pandas as pd
from pathlib import Path

# Task 7
model_path = Path("models") / "weather_classifier.pkl"
metadata_path = Path("models") / "weather_classifier_metadata.json"

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)


print("Weather Classifier Information")
print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test accuracy: {metadata['test_accuracy']:.4f}")


features = metadata["features"]

weather_days = pd.DataFrame([
    {
        "day": "Clearly good",
        "temperature_2m_max": 20.0,
        "temperature_2m_min": 12.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0,
    },
    {
        "day": "Clearly bad",
        "temperature_2m_max": 35.0,
        "temperature_2m_min": 25.0,
        "precipitation_sum": 10.0,
        "wind_speed_10m_max": 40.0,
    },
    {
        "day": "Borderline: cool",
        "temperature_2m_max": 8.0,
        "temperature_2m_min": 1.0,
        "precipitation_sum": 1.0,
        "wind_speed_10m_max": 15.0,
    },
    {
        "day": "Borderline: windy",
        "temperature_2m_max": 24.0,
        "temperature_2m_min": 15.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 28.0,
    },
])


X = weather_days[features]

# Predict labels and probabilities.
predictions = model.predict(X)
probabilities = model.predict_proba(X)
good_class_index = list(model.classes_).index(1)
good_probabilities = probabilities[:, good_class_index]

print("\nHypothetical Weather Predictions")

for index, row in weather_days.iterrows():
    label = "Good for running" if predictions[index] == 1 else "Skip"
    print(f"\nDay: {row['day']}")
    for feature in features:
        print(f"  {feature}: {row[feature]}")
    print(f"  Predicted label: {label}")
    print(f"  Probability of good conditions: {good_probabilities[index]:.1%}")


# Task 8
# 1. The borderline cool day and borderline windy day both received
#    a 100% probability of good running conditions. The model appears
#    confident based on its KNN probability estimate, but this does
#    not guarantee that the prediction is correct or that the weather
#    is safe. If the model predicts 0.52, I would treat the result as
#    uncertain because the estimated probability is close to 50%.
#    I would check the actual weather conditions before recommending
#    a run rather than relying only on the model prediction.
#
# 2. If predict.py ran before train_classifier.py had ever been run,
#    the saved model file and metadata file would be missing. Loading
#    them would raise a FileNotFoundError. I could make the error
#    message more helpful by checking whether both files exist first
#    and explaining that the model must be trained and saved before
#    predictions can be made.
#
# 3. In a cloud pipeline, predict.py would need to accept new daily
#    weather data automatically instead of using four hardcoded
#    hypothetical examples. It could read data from the pipeline
#    input and produce predictions in a structured output format.
