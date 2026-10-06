import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 6 - CALCULATE SLOPE AND INTERCEPT MANUALLY
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

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

OUTPUT_PLOT = (
    PLOTS_FOLDER
    / "step6_best_fit_regression_line.png"
)


# ------------------------------------------------------------
# 2. CREATE PLOTS FOLDER
# ------------------------------------------------------------

PLOTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 70)
print("STEP 6 - CALCULATE SLOPE AND INTERCEPT MANUALLY")
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
# 5. CALCULATE MEAN OF X AND y
# ------------------------------------------------------------

x_mean = np.mean(X)

y_mean = np.mean(y)


print("\n" + "=" * 70)
print("MEAN VALUES")
print("=" * 70)

print(
    f"Mean of X (area) = "
    f"{x_mean:.2f} sq ft"
)

print(
    f"Mean of y (price) = "
    f"${y_mean:,.2f}"
)


# ------------------------------------------------------------
# 6. CALCULATE DEVIATIONS FROM THE MEAN
# ------------------------------------------------------------

x_difference = (
    X - x_mean
)

y_difference = (
    y - y_mean
)


# ------------------------------------------------------------
# 7. CALCULATE SLOPE
# ------------------------------------------------------------

# Formula:
#
# slope =
# Σ((x - x_mean) * (y - y_mean))
# --------------------------------
# Σ((x - x_mean)^2)


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
# 8. CALCULATE INTERCEPT
# ------------------------------------------------------------

# Formula:
#
# intercept =
# y_mean - slope * x_mean


intercept = (
    y_mean
    - slope * x_mean
)


# ------------------------------------------------------------
# 9. DISPLAY CALCULATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SLOPE CALCULATION")
print("=" * 70)

print(
    f"Numerator = {numerator:,.2f}"
)

print(
    f"Denominator = {denominator:,.2f}"
)

print(
    f"Slope = {slope:,.4f}"
)


print("\n" + "=" * 70)
print("INTERCEPT CALCULATION")
print("=" * 70)

print(
    f"Intercept = {intercept:,.2f}"
)


# ------------------------------------------------------------
# 10. DISPLAY FINAL REGRESSION EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL REGRESSION EQUATION")
print("=" * 70)

print(
    "Predicted Price = "
    f"{intercept:,.2f} "
    f"+ ({slope:,.4f} × Area)"
)


# ------------------------------------------------------------
# 11. EXPLAIN SLOPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT DOES THE SLOPE MEAN?")
print("=" * 70)

print(
    f"For every additional 1 square foot, "
    f"the predicted house price changes by "
    f"approximately ${slope:,.2f}."
)


# ------------------------------------------------------------
# 12. EXPLAIN INTERCEPT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT DOES THE INTERCEPT MEAN?")
print("=" * 70)

print(
    "The intercept is the predicted price "
    "when house area is 0 square feet."
)

print(
    f"Intercept = ${intercept:,.2f}"
)

print(
    "For this dataset, a 0 sq ft house is not "
    "a realistic case, so the intercept is mainly "
    "part of the mathematical regression equation."
)


# ------------------------------------------------------------
# 13. MAKE ONE EXAMPLE PREDICTION
# ------------------------------------------------------------

example_area = 2000


example_prediction = (
    intercept
    + slope * example_area
)


print("\n" + "=" * 70)
print("EXAMPLE PREDICTION")
print("=" * 70)

print(
    f"House area = "
    f"{example_area} sq ft"
)

print(
    f"Predicted price = "
    f"${example_prediction:,.2f}"
)


# ------------------------------------------------------------
# 14. CALCULATE PREDICTIONS FOR REGRESSION LINE
# ------------------------------------------------------------

predicted_prices = (
    intercept
    + slope * X
)


# ------------------------------------------------------------
# 15. SORT VALUES FOR A CLEAN LINE
# ------------------------------------------------------------

sorted_indices = np.argsort(
    X
)

X_sorted = X[
    sorted_indices
]

predicted_sorted = predicted_prices[
    sorted_indices
]


# ------------------------------------------------------------
# 16. CREATE GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    X,
    y,
    alpha=0.6,
    label="Actual House Prices"
)


plt.plot(
    X_sorted,
    predicted_sorted,
    linewidth=2,
    label="Best-Fit Regression Line"
)


plt.xlabel(
    "House Area (Square Feet)"
)

plt.ylabel(
    "House Price (USD)"
)

plt.title(
    "Manual Linear Regression - Area vs Price"
)

plt.legend()

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 17. SAVE GRAPH
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)


print("\n" + "=" * 70)
print("PLOT SAVED")
print("=" * 70)

print(
    OUTPUT_PLOT
)


# ------------------------------------------------------------
# 18. SHOW GRAPH
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 19. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)