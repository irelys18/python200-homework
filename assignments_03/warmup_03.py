from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing Q1

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Q1 - Means of scaled training features:")
print(pd.DataFrame(X_train_scaled, columns=numeric).mean())

# Fit the scaler only on the training data to prevent data leakage from the test set.


# Preprocessing Q2
encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print("\nQ2 = Encoded season array Shape:")
print(season_encoded.shape)

print("\nQ2 = Category names:")
print(encoder.get_feature_names_out(["season"]))

# The single season column becomes 4 columns because the dataset contains
# 4 unique categories: fall, spring, summer, and winter.
# Each category gets its own binary column.

# Preprocessing Q3
numeric_features = numeric
categorical_features = ["season"]

X_mixed = df[numeric_features + categorical_features]
y_mixed = df["good_for_running"]

X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(
    X_mixed,
    y_mixed,
    test_size=0.2,
    random_state=42,
    stratify=y_mixed,
)

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("season", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

preprocessing_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("logistic_regression", LogisticRegression(max_iter=1000)),
])

preprocessing_pipeline.fit(X_train_mixed, y_train_mixed)
y_pred_mixed = preprocessing_pipeline.predict(X_test_mixed)

print("\nPreprocessing Q3 - Test Accuracy:")
print(accuracy_score(y_test_mixed, y_pred_mixed))

# The ColumnTransformer scales numeric features and one-hot encodes season.
# The pipeline applies these preprocessing steps consistently during training
# and prediction.


# KNN Q1
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)

print("\nKNN Q1 - Test Accuracy:")
print(accuracy_score(y_test, y_pred_knn))

print("\nKNN Q1 - Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Scaling helps KNN compare features on similar numerical scales.


# KNN Q2
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}

for k in k_values:
    candidate_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        candidate_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    cv_results[k] = scores.mean()
    print(f"KNN Q2 - k={k}: Mean CV accuracy = {scores.mean():.4f}")

best_k = max(cv_results, key=cv_results.get)
print("Best k from cross-validation:", best_k)

# k=7 achieved the highest mean cross-validation accuracy (93.96%).
# I would select k=7 based on cross-validation performance.

# KNN Q3
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)

y_pred_raw = knn_raw.predict(X_test)
accuracy_raw = accuracy_score(y_test, y_pred_raw)

print("\nQ3 - KNN accuracy without scaling:", accuracy_raw)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print("Q3 - KNN accuracy with scaling:", accuracy_scaled)


# Scaling improved accuracy from 92.5% to 95.83%.
# KNN relies on distances, so scaling prevents features with larger
# numerical ranges from dominating the distance calculations.


# Logistic Regression Q1
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000))
])

logistic_pipeline.fit(X_train, y_train)

y_pred_logistic = logistic_pipeline.predict(X_test)


print("\nLogistic Regression Q1 - Test Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Precision:", precision_score(y_test, y_pred_logistic, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_logistic, pos_label=1))
print("F1:", f1_score(y_test, y_pred_logistic, pos_label=1))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))


# Logistic Regression Q2
feature_names = X_train.columns
coefficients = logistic_pipeline.named_steps["logistic_regression"].coef_[0]

print("\nLogistic Regression Q2 - Feature Coefficients")
for feature, coefficient in zip(feature_names, coefficients):
    print(f"{feature}: {coefficient:.4f}")

# Wind speed (-2.4438) and precipitation (-1.8891) push most strongly
# toward class 0 ("skip"). This matches my intuition because strong winds
# and rain can make running less comfortable and safe. Minimum temperature
# pushes toward class 1 ("good"), while maximum temperature has a small
# negative effect.


# Logistic Regression Q3
probabilities = logistic_pipeline.predict_proba(X_test)

good_class_index = list(logistic_pipeline.classes_).index(1)

good_probabilities = probabilities[:5, good_class_index]

print("\nLogistic Regression Q3 - Probability of good for first five test days:")
print(good_probabilities.round(2))

# A probability near 0.5 means the model is less confident because
# it considers "good" and "skip" nearly equally likely.


# Evaluation Q1
y_pred_logistic = logistic_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred_logistic, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"]
)

disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()

print("\nEvaluation Q1 - Confusion Matrix:")
print(cm)
print("False positives:", cm[0, 1])
print("False negatives:", cm[1, 0])

# The model made 9 false positives and 7 false negatives.
# False positives may matter more for a running app because users could
# be encouraged to run in unsuitable or potentially unsafe weather.


# Joblib  Q1
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "warmup_model.pkl"

joblib.dump(logistic_pipeline, model_path)
print("\nJoblib Q1 - Model saved to:", model_path)

loaded_model = joblib.load(model_path)
loaded_predictions = loaded_model.predict(X_test)

assert (loaded_predictions == y_pred_logistic).all()

print("Reloaded model predictions match the original predictions.")

# Saving the entire pipeline preserves the scaler and classifier together.
# This helps ensure that future predictions use the same preprocessing.

