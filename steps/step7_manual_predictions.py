import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 7 - MAKE PREDICTIONS MANUALLY
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_encoded.csv"
)

OUTPUTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step7_manual_predictions.csv"
)


# ------------------------------------------------------------
# 2. CREATE OUTPUTS FOLDER
# ------------------------------------------------------------

OUTPUTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 70)
print("STEP 7 - MAKE PREDICTIONS MANUALLY")
print("=" * 70)


# ------------------------------------------------------------
# 4. SELECT X AND y
# ------------------------------------------------------------

X = df["area_sqft"].to_numpy(
    dtype=float
)

y = df["price_usd"].to_numpy(
    dtype=float
)


print("\nSelected variables:")

print("X = area_sqft")
print("y = price_usd")


# ------------------------------------------------------------
# 5. CALCULATE MEAN VALUES
# ------------------------------------------------------------

x_mean = np.mean(X)

y_mean = np.mean(y)


# ------------------------------------------------------------
# 6. CALCULATE SLOPE MANUALLY
# ------------------------------------------------------------

x_difference = (
    X - x_mean
)

y_difference = (
    y - y_mean
)


numerator = np.sum(
    x_difference
    * y_difference
)

denominator = np.sum(
    x_difference ** 2
)


slope = (
    numerator
    / denominator
)


# ------------------------------------------------------------
# 7. CALCULATE INTERCEPT MANUALLY
# ------------------------------------------------------------

intercept = (
    y_mean
    - slope * x_mean
)


print("\n" + "=" * 70)
print("REGRESSION EQUATION")
print("=" * 70)

print(
    f"Slope = {slope:,.4f}"
)

print(
    f"Intercept = {intercept:,.2f}"
)

print(
    "\nPredicted Price = "
    f"{intercept:,.2f} "
    f"+ ({slope:,.4f} × Area)"
)


# ------------------------------------------------------------
# 8. MAKE PREDICTIONS MANUALLY
# ------------------------------------------------------------

predicted_prices = (
    intercept
    + slope * X
)


# ------------------------------------------------------------
# 9. ADD PREDICTIONS TO DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "house_id": df["house_id"],
        "area_sqft": X,
        "actual_price_usd": y,
        "predicted_price_usd": predicted_prices,
    }
)


# ------------------------------------------------------------
# 10. ROUND PREDICTED PRICES
# ------------------------------------------------------------

results["predicted_price_usd"] = (
    results["predicted_price_usd"]
    .round(2)
)


# ------------------------------------------------------------
# 11. DISPLAY FIRST 10 PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 MANUAL PREDICTIONS")
print("=" * 70)

print(
    results.head(10)
)


# ------------------------------------------------------------
# 12. SHOW ONE PREDICTION CALCULATION
# ------------------------------------------------------------

example_area = X[0]

example_actual_price = y[0]

example_predicted_price = (
    intercept
    + slope * example_area
)


print("\n" + "=" * 70)
print("ONE PREDICTION EXPLAINED")
print("=" * 70)

print(
    f"House area = "
    f"{example_area:,.0f} sq ft"
)

print(
    f"Actual price = "
    f"${example_actual_price:,.2f}"
)

print("\nFormula:")

print(
    "Predicted Price = "
    "Intercept + Slope × Area"
)

print("\nCalculation:")

print(
    f"Predicted Price = "
    f"{intercept:,.2f} "
    f"+ ({slope:,.4f} × {example_area:,.0f})"
)

print(
    f"Predicted Price = "
    f"${example_predicted_price:,.2f}"
)


# ------------------------------------------------------------
# 13. TEST WITH CUSTOM EXAMPLE AREAS
# ------------------------------------------------------------

example_areas = [
    1000,
    1500,
    2000,
    2500,
    3000,
]


print("\n" + "=" * 70)
print("PREDICTIONS FOR SAMPLE HOUSE AREAS")
print("=" * 70)


for area in example_areas:

    prediction = (
        intercept
        + slope * area
    )

    print(
        f"{area:>4} sq ft "
        f"-> "
        f"${prediction:,.2f}"
    )


# ------------------------------------------------------------
# 14. SAVE RESULTS
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("PREDICTIONS FILE SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)