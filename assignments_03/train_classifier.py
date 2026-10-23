

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# Task 1
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 35.73,
    "longitude": -78.85,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

# City: Apex, North Carolina, USA
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Weather data shape:", df.shape)
print(df.head())


# Task 2: 
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

class_counts = df["good_for_running"].value_counts().reindex(
    [1, 0], fill_value=0
)
good_days = int(class_counts[1])
skip_days = int(class_counts[0])
total_days = len(df)
good_fraction = good_days / total_days

print("\nTask 2 - Class balance")
print(f"Good days (1): {good_days}")
print(f"Skip days (0): {skip_days}")
print(f"Fraction of days that are good: {good_fraction:.1%}")


# About 42.7% of days are classified as good for running.
# This seems reasonable for Apex, NC, because some days may be too
# cold, too hot, rainy, or windy to meet our running criteria.
# The classes are not severely imbalanced, but skip days are more common.
# Accuracy, precision, recall, and F1-score will help evaluate the model.


# Task 3
features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

k_values = [3, 5, 7, 9, 11, 15]
cv_results = []

for k in k_values:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    mean_score = scores.mean()
    cv_results.append((k, mean_score))
    print(f"k={k}: mean 5-fold CV accuracy = {mean_score:.4f}")

best_k, best_cv_accuracy = max(cv_results, key=lambda result: result[1])
print(f"\nBest k: {best_k}")
print(f"Best mean CV accuracy: {best_cv_accuracy:.4f}")

best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Skip", "Good for running"],
        zero_division=0,
    )
)


# The best k was 3, with a mean cross-validation accuracy of 93.13%.
# The model achieved 93.15% test accuracy.
# Recall for good running days was 97%, meaning the model identified
# most suitable days. Precision was 88%, so some predicted good days
# were actually skip days. Precision matters because weather conditions
# should be suitable before recommending an outdoor run.



# Task 4
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000)),
])

logistic_pipeline.fit(X_train, y_train)

y_pred_logistic = logistic_pipeline.predict(X_test)

print("\nTask 4 - Logistic Regression Results")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_logistic):.4f}")
print(
    f"Precision: {precision_score(y_test, y_pred_logistic, pos_label=1):.4f}"
)
print(f"Recall: {recall_score(y_test, y_pred_logistic, pos_label=1):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_logistic, pos_label=1):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred_logistic,
        labels=[0, 1],
        target_names=["Skip", "Good for running"],
        zero_division=0,
    )
)

logistic_model = logistic_pipeline.named_steps["logistic_regression"]

print("\nFeature coefficients:")
for feature, coefficient in zip(features, logistic_model.coef_[0]):
    print(f"{feature}: {coefficient:.4f}")


# Logistic Regression achieved 84.93% test accuracy, lower than KNN's 93.15%.
# Precipitation has the most negative coefficient (-3.0184), so it
# most strongly pushes predictions toward skip, followed by maximum
# temperature (-0.9902). Minimum temperature has a small negative
# coefficient, while wind speed has a positive coefficient (0.3606).
# The wind coefficient is unexpected given our labeling rules, showing
# that learned relationships may not perfectly match the rule-based labels.



# Task 5
# KNN achieved 93.15% test accuracy, higher than Logistic Regression's
# 84.93%, so KNN performed better on this test set.
#
# For a running app, accuracy is important, but precision and recall
# also matter. A false positive means recommending a run on a day
# labeled unsuitable, while a false negative means missing a day
# labeled suitable. False positives may be more concerning because
# unsuitable weather could make an outdoor run uncomfortable or unsafe.
# Therefore, I would pay close attention to precision for good-running
# predictions while also monitoring recall so that too many good days
# are not missed.
#
# I would deploy KNN with k=3 because it achieved higher test accuracy
# (93.15%) and higher recall for good-running days (97%) than Logistic
# Regression (84.93% accuracy and 77.42% recall). KNN is simple and
# fast enough for this small dataset, although it stores the training
# data for predictions. Logistic Regression is more interpretable
# through its coefficients and naturally provides probability scores.
# KNN can also provide probability estimates through predict_proba,
# so confidence scores are possible with either model.


# Task 6

# The chosen model is the best KNN pipeline from Task 3.
# best_model is already fitted on the training data.

y_pred_best = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_best)
test_f1 = f1_score(y_test, y_pred_best, pos_label=1)


models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)


model_path = models_dir / "weather_classifier.pkl"
joblib.dump(best_model, model_path)

label_rule = (
    "good_for_running=1 when temperature_2m_max is between 7 and 26 "
    "degrees Celsius inclusive, temperature_2m_min >= 0 degrees Celsius, "
    "precipitation_sum < 3 mm, and wind_speed_10m_max < 30 km/h; "
    "otherwise good_for_running=0."
)

metadata = {
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "city": "Apex, North Carolina, USA",
    "data_year": 2023,
    "features": features,
    "target": "good_for_running",
    "label_rule": label_rule,
    "chosen_model": "KNeighborsClassifier",
    "best_k": best_k,
    "test_accuracy": round(test_accuracy, 4),
    "test_f1": round(test_f1, 4),
}

metadata_path = models_dir / "weather_classifier_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nTask 6 - Model saved successfully")
print(f"Model file: {model_path}")
print(f"Metadata file: {metadata_path}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
