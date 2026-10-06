import pandas as pd
from pathlib import Path


# ============================================================
# STEP 1 - ENCODE LOCATION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_linear_regression_practice.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_encoded.csv"
)


# ------------------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print("\n" + "=" * 60)
print("HOUSE PRICE DATASET")
print("=" * 60)


print("\nFIRST 5 ROWS:")

print(
    df.head()
)


# ------------------------------------------------------------
# 3. CHECK ORIGINAL LOCATION VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ORIGINAL LOCATION VALUES")
print("=" * 60)


print(
    df["location"].unique()
)


print("\nOriginal location data type:")

print(
    df["location"].dtype
)


print("\nLocation counts:")

print(
    df["location"].value_counts()
)


# ------------------------------------------------------------
# 4. CREATE ENCODING MAPPING
# ------------------------------------------------------------

location_mapping = {
    "Rural": 0,
    "Suburban": 1,
    "Urban": 2,
}


print("\n" + "=" * 60)
print("ENCODING RULE")
print("=" * 60)


print("Rural     = 0")
print("Suburban  = 1")
print("Urban     = 2")


# ------------------------------------------------------------
# 5. ENCODE LOCATION COLUMN
# ------------------------------------------------------------

df["location"] = df["location"].map(
    location_mapping
)


# ------------------------------------------------------------
# 6. VIEW DATA AFTER ENCODING
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA AFTER ENCODING")
print("=" * 60)


print(
    df.head()
)


# ------------------------------------------------------------
# 7. CHECK ENCODED VALUES
# ------------------------------------------------------------

print("\nEncoded location values:")

print(
    df["location"].unique()
)


print("\nNew location data type:")

print(
    df["location"].dtype
)


# ------------------------------------------------------------
# 8. CHECK FOR ENCODING ERRORS
# ------------------------------------------------------------

missing_values = (
    df["location"]
    .isnull()
    .sum()
)


print("\n" + "=" * 60)
print("ENCODING VALIDATION")
print("=" * 60)


print(
    f"Missing values after encoding: "
    f"{missing_values}"
)


if missing_values == 0:

    print(
        "Encoding successful."
    )

else:

    print(
        "WARNING: Some location values "
        "were not encoded."
    )


# ------------------------------------------------------------
# 9. CHECK ENCODED CATEGORY COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ENCODED LOCATION COUNTS")
print("=" * 60)


print(
    df["location"]
    .value_counts()
    .sort_index()
)


print("\nMeaning:")

print("0 = Rural")
print("1 = Suburban")
print("2 = Urban")


# ------------------------------------------------------------
# 10. CHECK ALL DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN DATA TYPES")
print("=" * 60)


print(
    df.dtypes
)


# ------------------------------------------------------------
# 11. SAVE ENCODED DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 60)
print("ENCODED FILE CREATED")
print("=" * 60)


print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 12. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 60)
