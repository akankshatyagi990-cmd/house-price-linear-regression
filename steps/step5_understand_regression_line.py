import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 5 - UNDERSTAND THE REGRESSION LINE
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
    / "step5_regression_line.png"
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


print("\n" + "=" * 60)
print("STEP 5 - UNDERSTAND THE REGRESSION LINE")
print("=" * 60)


# ------------------------------------------------------------
# 4. SELECT X AND y
# ------------------------------------------------------------

X = df["area_sqft"]

y = df["price_usd"]


print("\nSelected variables:")

print("X = area_sqft")

print("y = price_usd")


# ------------------------------------------------------------
# 5. CREATE A BEST-FIT LINE
# ------------------------------------------------------------

# np.polyfit finds the slope and intercept
# of the best-fit straight line.
#
# We are using it here only to VISUALIZE the line.
#
# In Step 6, we will calculate slope and intercept
# manually and understand the mathematics.

slope, intercept = np.polyfit(
    X,
    y,
    1
)


# ------------------------------------------------------------
# 6. CREATE X VALUES FOR THE LINE
# ------------------------------------------------------------

line_x = np.linspace(
    X.min(),
    X.max(),
    200
)


# ------------------------------------------------------------
# 7. CALCULATE y VALUES FOR THE LINE
# ------------------------------------------------------------

line_y = (
    slope * line_x
    + intercept
)


# ------------------------------------------------------------
# 8. CREATE THE GRAPH
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


# Actual houses
plt.scatter(
    X,
    y,
    alpha=0.6,
    label="Actual House Prices"
)


# Regression line
plt.plot(
    line_x,
    line_y,
    linewidth=2,
    label="Regression Line"
)


# ------------------------------------------------------------
# 9. ADD LABELS
# ------------------------------------------------------------

plt.xlabel(
    "House Area (Square Feet)"
)

plt.ylabel(
    "House Price (USD)"
)

plt.title(
    "Understanding the Linear Regression Line"
)

plt.legend()

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 10. SAVE GRAPH
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)


# ------------------------------------------------------------
# 11. DISPLAY EXPLANATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("WHAT DOES THE LINE MEAN?")
print("=" * 60)

print(
    "Each dot represents one actual house."
)

print(
    "The line represents the general relationship "
    "between house area and house price."
)

print(
    "The line does not pass through every point."
)

print(
    "Instead, it tries to stay as close as possible "
    "to all the data points."
)

print(
    "For a new house area, the regression line can "
    "be used to estimate its price."
)


# ------------------------------------------------------------
# 12. SHOW EXAMPLE PREDICTION FROM THE LINE
# ------------------------------------------------------------

example_area = 2000

example_prediction = (
    slope * example_area
    + intercept
)


print("\n" + "=" * 60)
print("EXAMPLE")
print("=" * 60)

print(
    f"House area: {example_area} sq ft"
)

print(
    f"Estimated price from regression line: "
    f"${example_prediction:,.2f}"
)


# ------------------------------------------------------------
# 13. SAVE AND SHOW GRAPH
# ------------------------------------------------------------

print("\nPlot saved at:")

print(
    OUTPUT_PLOT
)


plt.show()


# ------------------------------------------------------------
# 14. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 60)