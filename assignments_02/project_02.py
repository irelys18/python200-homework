import requests
import pandas as pd
import matplotlib.pyplot as plt

# City chosen: Apex, North Carolina
url = "https://archive-api.open-meteo.com/v1/archive"

params = {"latitude": 35.73, 
          "longitude": -78.85, 
          "start_date": "2023-01-01", 
          "end_date": "2023-12-31", 
          "daily": [
              "temperature_2m_max", 
              "temperature_2m_min", 
              "precipitation_sum",
              "wind_speed_10m_max",
        ], 
        "timezone": "America/New_York"
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Dataset shape:", df.shape)
print(df.head())

# Task 2
print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# No missing values were found, so no imputation or row removal is needed.

numeric_df = df.select_dtypes(include = "number")

correlations = numeric_df.corr()["temperature_2m_max"]
correlations = correlations.drop("temperature_2m_max")
correlations = correlations.sort_values(ascending=False)

print("\nCorrelation with daily high, strongest first:")
print(correlations)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"],
    alpha=0.6
)

plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.title("Daily Low vs. Daily High Temperature")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("assignments_02/outputs/low_vs_high.png")
plt.show()

# Observation: The scatter plot shows whether higher daily lows
# tend to occur alongside higher daily highs.


# Task 3
# June, July, and August are considered summer months.
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("\nSummer and non summer days:")
print(df["is_summer"].value_counts().sort_index().rename(
    index={0: "Non summer days", 1: "Summer days"}
))

# I expect is_summer to help predict daily high temperatures because
# summer days are generally warmer than days in other seasons.

# Task 4
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

x = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nBaseline Model Results:")
print("Slope:", baseline_model.coef_[0])
print("RMSE:", rmse)
print("R2:", r2)

# RMSE is measured in degrees Celsius. Our predictions are typically
# off by about 3.15°C, although individual errors can vary.

# Task 5
feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer"
]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model =LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print("\nFull Model Results:")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.3f}°C")

print(f"Baseline test R2: {r2:.4f}")
print(f"Change in test R2: {test_r2 - r2:+.4f}")

if test_r2 > r2:
    print("The full model improved test R2 over the baseline.")
elif test_r2 < r2:
    print("The full model lowered test R2 compared to the baseline.")
else:
    print("The full model did not change test R2.")

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:.3f}")



# Interpretation: Daily low temperature and is_summer have positive
# coefficients, so they raise predicted highs when other features are held
# constant. Precipitation and wind speed have negative coefficients,
# so they lower predicted highs. These are associations, not causal effects.

# Train R2 (0.8728) and test R2 (0.8996) are reasonably close.
# Test R2 is slightly higher, suggesting this test split may be easier
# to predict; there is no obvious sign of overfitting from these scores.

# The full model's test R2 (0.8996) is higher than the baseline (0.8597),
# so adding precipitation, wind speed, and is_summer improved performance.

# Task 6
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_test_pred, alpha=0.7)

min_temp = min(y_test.min(), y_test_pred.min())
max_temp = max(y_test.max(), y_test_pred.max())
plt.plot([min_temp, max_temp], [min_temp, max_temp], linestyle="--")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Actual Daily High (°C)")
plt.ylabel("Predicted Daily High (°C)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("assignments_02/outputs/predicted_vs_actual_full_model.png")
plt.show()


# Plot observation: Most predictions are close to the diagonal.
# The model tends to overpredict some lower temperatures and
# underpredict several temperatures in the middle-to-high range.
# Errors vary across the range rather than being perfectly even.

# The dataset contains 365 daily observations. The 20% test set contains 73 days.
# The full model's test RMSE is 2.669°C, meaning prediction errors are
# typically around 2.67°C in size, although individual errors vary.
# The full model's test R² is 0.8996, explaining about 90% of the
# variation in daily high temperatures on the test set.
# The largest coefficient by absolute numerical value is temperature_2m_min
# (+0.925), which is associated with higher predicted daily highs.
# One surprising result is that test R² (0.8996) is higher than train R²
# (0.8728), which can happen when the test set is easier to predict.