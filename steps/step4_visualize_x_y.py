import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 4 - VISUALIZE X VS y
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
    / "step4_area_vs_price.png"
)


# ------------------------------------------------------------
# 2. CREATE PLOTS FOLDER IF NEEDED
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
print("STEP 4 - VISUALIZE X VS y")
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
# 5. DISPLAY FIRST 10 X-y PAIRS
# ------------------------------------------------------------

xy_data = pd.DataFrame(
    {
        "Area Sq Ft (X)": X,
        "House Price (y)": y,
    }
)


print("\nFIRST 10 X-y PAIRS")
print("=" * 60)

print(
    xy_data.head(10)
)


# ------------------------------------------------------------
# 6. CREATE SCATTER PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    X,
    y,
    alpha=0.7,
)


# ------------------------------------------------------------
# 7. ADD GRAPH LABELS
# ------------------------------------------------------------

plt.xlabel(
    "House Area (Square Feet)"
)

plt.ylabel(
    "House Price (USD)"
)

plt.title(
    "House Area vs House Price"
)


# ------------------------------------------------------------
# 8. ADD GRID
# ------------------------------------------------------------

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 9. SAVE GRAPH
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 60)
print("PLOT CREATED")
print("=" * 60)

print(
    OUTPUT_PLOT
)


# ------------------------------------------------------------
# 10. DISPLAY GRAPH
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 11. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)