import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 8 - CALCULATE RESIDUALS
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_manual_predictions.csv"
)

OUTPUTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step8_residuals.csv"
)

OUTPUT_PLOT = (
    PLOTS_FOLDER
    / "step8_residual_plot.png"
)


# ------------------------------------------------------------
# 2. CREATE FOLDERS IF NEEDED
# ------------------------------------------------------------

OUTPUTS_FOLDER.mkdir(
    exist_ok=True
)

PLOTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD STEP 7 PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print("\n" + "=" * 70)
print("STEP 8 - CALCULATE RESIDUALS")
print("=" * 70)


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "house_id",
    "area_sqft",
    "actual_price_usd",
    "predicted_price_usd",
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ------------------------------------------------------------
# 5. CALCULATE RESIDUAL
# ------------------------------------------------------------

# Formula:
#
# Residual =
# Actual Price - Predicted Price


df["residual"] = (
    df["actual_price_usd"]
    - df["predicted_price_usd"]
)


# ------------------------------------------------------------
# 6. DISPLAY FIRST 10 RESULTS
# ------------------------------------------------------------

print("\nFIRST 10 RESIDUALS")
print("=" * 70)

print(
    df[
        [
            "house_id",
            "area_sqft",
            "actual_price_usd",
            "predicted_price_usd",
            "residual",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 7. EXPLAIN ONE RESIDUAL
# ------------------------------------------------------------

example = df.iloc[0]


print("\n" + "=" * 70)
print("ONE RESIDUAL EXPLAINED")
print("=" * 70)

print(
    f"House ID = {example['house_id']}"
)

print(
    f"Area = {example['area_sqft']:,.0f} sq ft"
)

print(
    f"Actual Price = "
    f"${example['actual_price_usd']:,.2f}"
)

print(
    f"Predicted Price = "
    f"${example['predicted_price_usd']:,.2f}"
)

print("\nFormula:")

print(
    "Residual = Actual Price - Predicted Price"
)

print("\nCalculation:")

print(
    f"Residual = "
    f"{example['actual_price_usd']:,.2f} "
    f"- {example['predicted_price_usd']:,.2f}"
)

print(
    f"Residual = "
    f"${example['residual']:,.2f}"
)


# ------------------------------------------------------------
# 8. INTERPRET THE EXAMPLE
# ------------------------------------------------------------

print("\nInterpretation:")


if example["residual"] > 0:

    print(
        "Positive residual:"
    )

    print(
        "The actual house price was higher "
        "than the predicted price."
    )


elif example["residual"] < 0:

    print(
        "Negative residual:"
    )

    print(
        "The actual house price was lower "
        "than the predicted price."
    )


else:

    print(
        "Residual is 0:"
    )

    print(
        "The prediction exactly matched "
        "the actual price."
    )


# ------------------------------------------------------------
# 9. RESIDUAL SUMMARY
# ------------------------------------------------------------

mean_residual = (
    df["residual"].mean()
)

minimum_residual = (
    df["residual"].min()
)

maximum_residual = (
    df["residual"].max()
)


print("\n" + "=" * 70)
print("RESIDUAL SUMMARY")
print("=" * 70)

print(
    f"Mean Residual = "
    f"${mean_residual:,.2f}"
)

print(
    f"Minimum Residual = "
    f"${minimum_residual:,.2f}"
)

print(
    f"Maximum Residual = "
    f"${maximum_residual:,.2f}"
)


# ------------------------------------------------------------
# 10. SAVE RESIDUAL DATA
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("RESIDUAL FILE SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 11. CREATE RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    df["area_sqft"],
    df["residual"],
    alpha=0.7,
)


# Zero residual line
plt.axhline(
    y=0,
    linewidth=2,
    linestyle="--",
)


# ------------------------------------------------------------
# 12. ADD LABELS
# ------------------------------------------------------------

plt.xlabel(
    "House Area (Square Feet)"
)

plt.ylabel(
    "Residual (Actual Price - Predicted Price)"
)

plt.title(
    "Residual Plot - House Price Linear Regression"
)

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 13. SAVE PLOT
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 70)
print("RESIDUAL PLOT SAVED")
print("=" * 70)

print(
    OUTPUT_PLOT
)


# ------------------------------------------------------------
# 14. SHOW PLOT
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)