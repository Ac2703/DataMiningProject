import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

RAW_DATA_PATH = "large_art_ecommerce_dataset.csv"
CLEANED_DATA_PATH = "cleaned_dataset.csv"


def clean_dataset():
    """Load the raw dataset, perform lightweight cleaning, and export it."""
    print("Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")

    # ============================================
    # 1. STANDARDIZE STRING VALUES
    # ============================================
    print("\n" + "=" * 50)
    print("1. STANDARDIZING STRING COLUMNS")
    print("=" * 50)

    string_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"String columns detected: {len(string_cols)}")
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    # ============================================
    # 2. HANDLE MISSING VALUES
    # ============================================
    print("\n" + "=" * 50)
    print("2. HANDLING MISSING DATA")
    print("=" * 50)

    missing_before = df.isnull().sum()
    print("\nMissing values before handling:")
    print(missing_before[missing_before > 0])

    # Replace common textual placeholders with NaN
    df.replace(
        ["N/A", "n/a", "NA", "na", "None", "none", "", "nan", "NaN"],
        np.nan,
        inplace=True,
    )

    if "Reproduction Type" in df.columns:
        df["Reproduction Type"] = df["Reproduction Type"].fillna("Unknown")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    missing_after = df.isnull().sum()
    print("\nMissing values after handling:")
    print(missing_after[missing_after > 0])

    # ============================================
    # 3. REMOVE DUPLICATES
    # ============================================
    print("\n" + "=" * 50)
    print("3. REMOVING DUPLICATES")
    print("=" * 50)

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # ============================================
    # 4. DATA TYPE CONSISTENCY
    # ============================================
    print("\n" + "=" * 50)
    print("4. DATA TYPE CONSISTENCY")
    print("=" * 50)

    numeric_cols = ["Price ($)", "Delivery (days)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
            print(f"{col}: converted to numeric.")

    # ============================================
    # 5. SAVE CLEANED DATA
    # ============================================
    print("\n" + "=" * 50)
    print("SAVING CLEANED DATA")
    print("=" * 50)

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"✓ Saved cleaned dataset to {CLEANED_DATA_PATH}")

    print("\n" + "=" * 50)
    print("CLEANING COMPLETE!")
    print("=" * 50)
    print("\nSummary:")
    print(f"  - Original dataset shape: {original_shape}")
    print(f"  - Cleaned dataset shape: {df.shape}")
    print(f"  - Missing values handled: {missing_before.sum()} → {missing_after.sum()}")
    print(f"  - Duplicates removed: {duplicate_count}")


if __name__ == "__main__":
    clean_dataset()
