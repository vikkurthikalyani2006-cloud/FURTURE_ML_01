import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# STEP 1: GENERATE SYNTHETIC HISTORICAL DATA
# -------------------------------------------------------------
print("--- Step 1: Generating Historical Sales Data ---")
np.random.seed(42)
date_range = pd.date_range(start="2021-01-01", end="2025-12-01", freq="MS")

# Create a baseline trend with clear seasonal components (peaks in December)
base_sales = 50000
trend = np.linspace(0, 25000, len(date_range))
seasonality = 15000 * np.sin(2 * np.pi * date_range.month / 12)
noise = np.random.normal(0, 3000, len(date_range))

total_sales = base_sales + trend + seasonality + noise

df = pd.DataFrame({"Date": date_range, "Sales": total_sales})
df.set_index("Date", inplace=True)
print(f"Data generated successfully. Total records: {len(df)}")

# -------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING (LAG & ROLLING WINDOWS)
# -------------------------------------------------------------
print("\n--- Step 2: Engineering Time Series Features ---")
# Create time-based calendar features
df['Month'] = df.index.month
df['Year'] = df.index.year

# Create Lag features (using past months' values to predict future ones)
df['Lag_1'] = df['Sales'].shift(1)
df['Lag_2'] = df['Sales'].shift(2)
df['Lag_12'] = df['Sales'].shift(12)  # Same month last year

# Create Rolling Windows features
df['Rolling_Mean_3'] = df['Sales'].shift(1).rolling(window=3).mean()
df['Rolling_Std_3'] = df['Sales'].shift(1).rolling(window=3).std()

# Drop rows containing NaN values resulting from shifts
df.dropna(inplace=True)
print(f"Feature engineering complete. Available records: {len(df)}")

# -------------------------------------------------------------
# STEP 3: TEMPORAL TRAIN-TEST SPLIT
# -------------------------------------------------------------
print("\n--- Step 3: Performing Temporal Data Split ---")
# Split by time instead of randomly to avoid data leakage
split_date = "2025-01-01"
train = df.loc[df.index < split_date]
test = df.loc[df.index >= split_date]

features = ['Month', 'Year', 'Lag_1', 'Lag_2', 'Lag_12', 'Rolling_Mean_3', 'Rolling_Std_3']
target = 'Sales'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print(f"Training Period: {train.index.min().strftime('%Y-%m')} to {train.index.max().strftime('%Y-%m')}")
print(f"Testing Period: {test.index.min().strftime('%Y-%m')} to {test.index.max().strftime('%Y-%m')}")

# -------------------------------------------------------------
# STEP 4: MODEL TRAINING
# -------------------------------------------------------------
print("\n--- Step 4: Training XGBoost Regressor Model ---")
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
print("Model training completed successfully.")

# -------------------------------------------------------------
# STEP 5: MODEL EVALUATION
# -------------------------------------------------------------
print("\n--- Step 5: Evaluating Forecast Performance ---")
predictions = model.predict(X_test)
test_results = test.copy()
test_results['Predictions'] = predictions

# Metrics calculation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared Score (R²): {r2:.4f}")

# -------------------------------------------------------------
# STEP 6: VISUALIZING THE RESULTS
# -------------------------------------------------------------
print("\n--- Step 6: Generating Evaluation Plot ---")
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Sales'], label='Historical Training Data', color='blue')
plt.plot(test.index, test['Sales'], label='Actual Sales (Test)', color='green', marker='o')
plt.plot(test.index, test_results['Predictions'], label='Forecasted Sales', color='red', linestyle='--', marker='x')
plt.title('Sales Demand Forecasting Model Evaluation')
plt.xlabel('Timeline')
plt.ylabel('Sales Volume ($)')
plt.legend()
plt.grid(True)
plt.savefig('sales_forecast_evaluation.png')
print("Evaluation chart saved as 'sales_forecast_evaluation.png'")