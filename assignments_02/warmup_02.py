# --- scikit-learn API ---
# Q1

import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1,1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)
prediction_6 = model.predict(np.array([[6]]))
prediction_18 = model.predict(np.array([[18]]))

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for 6:", prediction_6)
print("Prediction for 18:", prediction_18)


# Q2

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("Reshaped to 2D:", x_2d.shape)

# scikit-learn needs X to be 2D so it 
# can organize the data into rows and features.


# --- Linear Regression ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")
plt.savefig("assignments_02/outputs/high_vs_low.png")
plt.show()

# # I can see two color groups, which suggests that is_rainy may 
# be related to temperature patterns.


# Q2
temp_min_2d = temp_min.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(temp_min_2d, temp_max, test_size=0.2, random_state=42)  

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q3
model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

r2_single = model.score(X_test, y_test)
rmse_single = np.sqrt(mean_squared_error(y_test, y_pred))

print("Test R2 with temp_min:", r2_single)
print("Test RMSE with temp_min:", rmse_single)

# For every 1-degree increase in the daily low, the predicted 
# daily high increases by about .94 degrees.


# Q4
X_full = np.column_stack((temp_min, is_rainy))

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split( 
    X_full, temp_max, test_size=0.2, random_state=42 )

model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)

r2_full = model_full.score(X_full_test, y_full_test)
print("Test R² with temp_min and is_rainy:", r2_full)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

print(f"Single-feature model R²: {r2_single:.4f}") 
print(f"Two-feature model R²: {r2_full:.4f}") 
print(f"Change in R²: {r2_full - r2_single:+.4f}") 

if r2_full > r2_single: 
    print("Adding is_rainy improved the model.") 
elif r2_full < r2_single: 
    print("Adding is_rainy lowered the model's R².") 
else: 
    print("Adding is_rainy did not change the model's R².")

# Holding the daily low temperature constant, rainy days are predicted
# to have a daily high about 2.92 degrees lower than non-rainy days.


# Q5
y_pred_full = model_full.predict(X_full_test)
plt.scatter(y_full_test, y_pred_full)
plt.title("Predicted vs Actual Daily Highs")
plt.xlabel("Actual Daily High")
plt.ylabel("Predicted Daily High")
plt.plot([y_full_test.min(), y_full_test.max()], [y_full_test.min(), y_full_test.max()])
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()

# A point above the diagonal means the actual value is higher than the
# prediction, so the model underestimated. A point below the diagonal
# means the actual value is lower than the prediction, so the model
# overestimated.