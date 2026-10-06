import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 10 - EVALUATE THE LINEAR REGRESSION MODEL
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step9_test_data.csv"
)

OUTPUTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

EVALUATION_FILE = (
    OUTPUTS_FOLDER
    / "step10_evaluation_results.csv"
)

METRICS_FILE = (
    OUTPUTS_FOLDER
    / "step10_metrics.csv"
)

OUTPUT_PLOT = (
    PLOTS_FOLDER
    / "step10_actual_vs_predicted.png"
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
# 3. LOAD TEST DATA FROM STEP 9
# ------------------------------------------------------------

df = pd.read_csv(
    TEST_DATA_FILE
)


print("\n" + "=" * 70)
print("STEP 10 - EVALUATE THE MODEL")
print("=" * 70)


# ------------------------------------------------------------
# 4. SELECT ACTUAL AND PREDICTED VALUES
# ------------------------------------------------------------

actual = df[
    "actual_price_usd"
].to_numpy(
    dtype=float
)

predicted = df[
    "predicted_price_usd"
].to_numpy(
    dtype=float
)


print("\nNumber of test houses:")

print(
    len(actual)
)


# ------------------------------------------------------------
# 5. CALCULATE RESIDUALS
# ------------------------------------------------------------

# Residual:
#
# Actual Price - Predicted Price


residuals = (
    actual
    - predicted
)


# ------------------------------------------------------------
# 6. CALCULATE ABSOLUTE ERRORS
# ------------------------------------------------------------

absolute_errors = np.abs(
    residuals
)


# ------------------------------------------------------------
# 7. CALCULATE SQUARED ERRORS
# ------------------------------------------------------------

squared_errors = (
    residuals ** 2
)


# ------------------------------------------------------------
# 8. CALCULATE MAE
# ------------------------------------------------------------

# MAE =
# average of absolute errors


mae = np.mean(
    absolute_errors
)


# ------------------------------------------------------------
# 9. CALCULATE MSE
# ------------------------------------------------------------

# MSE =
# average of squared errors


mse = np.mean(
    squared_errors
)


# ------------------------------------------------------------
# 10. CALCULATE RMSE
# ------------------------------------------------------------

# RMSE =
# square root of MSE


rmse = np.sqrt(
    mse
)


# ------------------------------------------------------------
# 11. CALCULATE R-SQUARED MANUALLY
# ------------------------------------------------------------

# R² =
#
# 1 - (Sum of Squared Residuals /
#      Total Sum of Squares)


actual_mean = np.mean(
    actual
)


sum_squared_residuals = np.sum(
    squared_errors
)


total_sum_squares = np.sum(
    (
        actual
        - actual_mean
    ) ** 2
)


r_squared = (
    1
    - (
        sum_squared_residuals
        / total_sum_squares
    )
)


# ------------------------------------------------------------
# 12. DISPLAY METRICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL EVALUATION METRICS")
print("=" * 70)


print(
    f"MAE  = ${mae:,.2f}"
)

print(
    f"MSE  = {mse:,.2f}"
)

print(
    f"RMSE = ${rmse:,.2f}"
)

print(
    f"R²   = {r_squared:.4f}"
)


# ------------------------------------------------------------
# 13. EXPLAIN MAE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT DOES MAE MEAN?")
print("=" * 70)

print(
    f"On average, the model's predicted house price "
    f"is approximately ${mae:,.2f} away "
    f"from the actual house price."
)


# ------------------------------------------------------------
# 14. EXPLAIN RMSE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT DOES RMSE MEAN?")
print("=" * 70)

print(
    f"RMSE = ${rmse:,.2f}"
)

print(
    "RMSE gives larger errors more importance "
    "because the errors are squared before averaging."
)


# ------------------------------------------------------------
# 15. EXPLAIN R-SQUARED
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT DOES R-SQUARED MEAN?")
print("=" * 70)


if r_squared >= 0.80:

    print(
        "The model explains a large portion of "
        "the variation in house prices."
    )

elif r_squared >= 0.50:

    print(
        "The model explains a moderate portion of "
        "the variation in house prices."
    )

elif r_squared >= 0:

    print(
        "The model explains only a limited portion of "
        "the variation in house prices."
    )

else:

    print(
        "The model performs worse than simply predicting "
        "the average test-set house price."
    )


print(
    f"\nR² Score = {r_squared:.4f}"
)


# ------------------------------------------------------------
# 16. CREATE DETAILED EVALUATION DATA
# ------------------------------------------------------------

evaluation_results = df.copy()


evaluation_results[
    "residual"
] = residuals


evaluation_results[
    "absolute_error"
] = absolute_errors


evaluation_results[
    "squared_error"
] = squared_errors


# ------------------------------------------------------------
# 17. ROUND VALUES
# ------------------------------------------------------------

evaluation_results[
    "residual"
] = (
    evaluation_results[
        "residual"
    ].round(2)
)


evaluation_results[
    "absolute_error"
] = (
    evaluation_results[
        "absolute_error"
    ].round(2)
)


evaluation_results[
    "squared_error"
] = (
    evaluation_results[
        "squared_error"
    ].round(2)
)


# ------------------------------------------------------------
# 18. DISPLAY FIRST 10 EVALUATION ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 TEST RESULTS")
print("=" * 70)

print(
    evaluation_results.head(10)
)


# ------------------------------------------------------------
# 19. SAVE DETAILED RESULTS
# ------------------------------------------------------------

evaluation_results.to_csv(
    EVALUATION_FILE,
    index=False
)


# ------------------------------------------------------------
# 20. CREATE METRICS DATAFRAME
# ------------------------------------------------------------

metrics = pd.DataFrame(
    {
        "metric": [
            "MAE",
            "MSE",
            "RMSE",
            "R_squared",
        ],
        "value": [
            mae,
            mse,
            rmse,
            r_squared,
        ],
    }
)


# ------------------------------------------------------------
# 21. SAVE METRICS
# ------------------------------------------------------------

metrics.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# 22. CREATE ACTUAL VS PREDICTED PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    actual,
    predicted,
    alpha=0.7,
)


# ------------------------------------------------------------
# 23. CREATE PERFECT-PREDICTION LINE
# ------------------------------------------------------------

minimum_value = min(
    actual.min(),
    predicted.min()
)

maximum_value = max(
    actual.max(),
    predicted.max()
)


plt.plot(
    [
        minimum_value,
        maximum_value,
    ],
    [
        minimum_value,
        maximum_value,
    ],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)


# ------------------------------------------------------------
# 24. ADD LABELS
# ------------------------------------------------------------

plt.xlabel(
    "Actual House Price (USD)"
)

plt.ylabel(
    "Predicted House Price (USD)"
)

plt.title(
    "Actual vs Predicted House Prices"
)

plt.legend()

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 25. SAVE PLOT
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    EVALUATION_FILE
)

print(
    METRICS_FILE
)

print(
    OUTPUT_PLOT
)


# ------------------------------------------------------------
# 26. SHOW PLOT
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 27. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 10 COMPLETED SUCCESSFULLY")
print("=" * 70)