import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 9 - TRAIN / TEST SPLIT
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

TRAIN_OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step9_train_data.csv"
)

TEST_OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step9_test_data.csv"
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
print("STEP 9 - TRAIN / TEST SPLIT")
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
# 5. CHECK TOTAL NUMBER OF ROWS
# ------------------------------------------------------------

total_rows = len(df)


print("\n" + "=" * 70)
print("ORIGINAL DATASET")
print("=" * 70)

print(
    f"Total rows = {total_rows}"
)


# ------------------------------------------------------------
# 6. CREATE RANDOM INDEXES
# ------------------------------------------------------------

# Setting a random seed makes sure
# we get the same split every time.

np.random.seed(
    42
)


indices = np.arange(
    total_rows
)


np.random.shuffle(
    indices
)


# ------------------------------------------------------------
# 7. CALCULATE SPLIT POINT
# ------------------------------------------------------------

train_ratio = 0.80


train_size = int(
    total_rows
    * train_ratio
)


# ------------------------------------------------------------
# 8. SPLIT INDEXES
# ------------------------------------------------------------

train_indices = (
    indices[:train_size]
)

test_indices = (
    indices[train_size:]
)


# ------------------------------------------------------------
# 9. CREATE TRAINING DATA
# ------------------------------------------------------------

X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


# ------------------------------------------------------------
# 10. CREATE TEST DATA
# ------------------------------------------------------------

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]


# ------------------------------------------------------------
# 11. DISPLAY SPLIT SIZE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training rows = {len(X_train)}"
)

print(
    f"Testing rows = {len(X_test)}"
)

print(
    f"Training percentage = "
    f"{len(X_train) / total_rows * 100:.0f}%"
)

print(
    f"Testing percentage = "
    f"{len(X_test) / total_rows * 100:.0f}%"
)


# ------------------------------------------------------------
# 12. CALCULATE TRAINING MEANS
# ------------------------------------------------------------

x_train_mean = np.mean(
    X_train
)

y_train_mean = np.mean(
    y_train
)


# ------------------------------------------------------------
# 13. CALCULATE TRAINING SLOPE
# ------------------------------------------------------------

x_difference = (
    X_train
    - x_train_mean
)

y_difference = (
    y_train
    - y_train_mean
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
# 14. CALCULATE TRAINING INTERCEPT
# ------------------------------------------------------------

intercept = (
    y_train_mean
    - slope * x_train_mean
)


# ------------------------------------------------------------
# 15. DISPLAY TRAINED MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINED USING TRAINING DATA ONLY")
print("=" * 70)

print(
    f"Slope = {slope:,.4f}"
)

print(
    f"Intercept = {intercept:,.2f}"
)

print("\nRegression Equation:")

print(
    "Predicted Price = "
    f"{intercept:,.2f} "
    f"+ ({slope:,.4f} × Area)"
)


# ------------------------------------------------------------
# 16. MAKE PREDICTIONS ON TEST DATA
# ------------------------------------------------------------

y_test_predicted = (
    intercept
    + slope * X_test
)


# ------------------------------------------------------------
# 17. CREATE TRAINING DATAFRAME
# ------------------------------------------------------------

train_data = pd.DataFrame(
    {
        "area_sqft": X_train,
        "actual_price_usd": y_train,
    }
)


# ------------------------------------------------------------
# 18. CREATE TEST DATAFRAME
# ------------------------------------------------------------

test_data = pd.DataFrame(
    {
        "area_sqft": X_test,
        "actual_price_usd": y_test,
        "predicted_price_usd": y_test_predicted,
    }
)


test_data["predicted_price_usd"] = (
    test_data["predicted_price_usd"]
    .round(2)
)


# ------------------------------------------------------------
# 19. DISPLAY SAMPLE TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 TRAINING ROWS")
print("=" * 70)

print(
    train_data.head(10)
)


# ------------------------------------------------------------
# 20. DISPLAY SAMPLE TEST PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 TEST PREDICTIONS")
print("=" * 70)

print(
    test_data.head(10)
)


# ------------------------------------------------------------
# 21. SAVE TRAINING DATA
# ------------------------------------------------------------

train_data.to_csv(
    TRAIN_OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 22. SAVE TEST DATA
# ------------------------------------------------------------

test_data.to_csv(
    TEST_OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    TRAIN_OUTPUT_FILE
)

print(
    TEST_OUTPUT_FILE
)


# ------------------------------------------------------------
# 23. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)