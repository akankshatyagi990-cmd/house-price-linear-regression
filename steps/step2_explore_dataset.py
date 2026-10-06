import pandas as pd
from pathlib import Path


# ============================================================
# STEP 2 - EXPLORE THE HOUSE PRICE DATASET
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_encoded.csv"
)


# ------------------------------------------------------------
# 2. LOAD ENCODED DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 60)
print("STEP 2 - HOUSE PRICE DATASET EXPLORATION")
print("=" * 60)


# ------------------------------------------------------------
# 3. VIEW FIRST 5 ROWS
# ------------------------------------------------------------

print("\nFIRST 5 ROWS")
print("=" * 60)

print(
    df.head()
)


# ------------------------------------------------------------
# 4. VIEW LAST 5 ROWS
# ------------------------------------------------------------

print("\nLAST 5 ROWS")
print("=" * 60)

print(
    df.tail()
)


# ------------------------------------------------------------
# 5. CHECK DATASET SIZE
# ------------------------------------------------------------

rows, columns = df.shape


print("\nDATASET SIZE")
print("=" * 60)

print(
    f"Number of rows: {rows}"
)

print(
    f"Number of columns: {columns}"
)


# ------------------------------------------------------------
# 6. CHECK COLUMN NAMES
# ------------------------------------------------------------

print("\nCOLUMN NAMES")
print("=" * 60)

for column in df.columns:

    print(
        column
    )


# ------------------------------------------------------------
# 7. CHECK DATA TYPES
# ------------------------------------------------------------

print("\nDATA TYPES")
print("=" * 60)

print(
    df.dtypes
)


# ------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("=" * 60)

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# 9. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

duplicate_rows = (
    df.duplicated().sum()
)


print("\nDUPLICATE ROWS")
print("=" * 60)

print(
    f"Duplicate rows: {duplicate_rows}"
)


# ------------------------------------------------------------
# 10. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nDESCRIPTIVE STATISTICS")
print("=" * 60)

print(
    df.describe().round(2)
)


# ------------------------------------------------------------
# 11. CHECK MINIMUM VALUES
# ------------------------------------------------------------

print("\nMINIMUM VALUES")
print("=" * 60)

print(
    df.select_dtypes(
        include="number"
    ).min()
)


# ------------------------------------------------------------
# 12. CHECK MAXIMUM VALUES
# ------------------------------------------------------------

print("\nMAXIMUM VALUES")
print("=" * 60)

print(
    df.select_dtypes(
        include="number"
    ).max()
)


# ------------------------------------------------------------
# 13. CHECK MEAN VALUES
# ------------------------------------------------------------

print("\nMEAN VALUES")
print("=" * 60)

print(
    df.select_dtypes(
        include="number"
    )
    .mean()
    .round(2)
)


# ------------------------------------------------------------
# 14. CHECK LOCATION DISTRIBUTION
# ------------------------------------------------------------

print("\nLOCATION DISTRIBUTION")
print("=" * 60)

print(
    df["location"]
    .value_counts()
    .sort_index()
)


print("\nEncoding meaning:")

print("0 = Rural")
print("1 = Suburban")
print("2 = Urban")


# ------------------------------------------------------------
# 15. CHECK HOUSE PRICE RANGE
# ------------------------------------------------------------

minimum_price = (
    df["price_usd"].min()
)

maximum_price = (
    df["price_usd"].max()
)

average_price = (
    df["price_usd"].mean()
)


print("\n" + "=" * 60)
print("HOUSE PRICE SUMMARY")
print("=" * 60)


print(
    f"Minimum House Price = "
    f"${minimum_price:,.0f}"
)

print(
    f"Maximum House Price = "
    f"${maximum_price:,.0f}"
)

print(
    f"Average House Price = "
    f"${average_price:,.0f}"
)


# ------------------------------------------------------------
# 16. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 60)