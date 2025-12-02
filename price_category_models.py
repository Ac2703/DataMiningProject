import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEANED_DATA_PATH = Path("cleaned_dataset.csv")
RAW_DATA_PATH = Path("large_art_ecommerce_dataset.csv")
PRICE_COLUMN = "Price ($)"
PRICE_CATEGORY_COLUMN = "Price Category"


def ensure_clean_data():
    """Run preprocessing if the cleaned dataset is not available."""
    if CLEANED_DATA_PATH.exists():
        print("Found existing cleaned dataset.")
        return pd.read_csv(CLEANED_DATA_PATH)

    print("Cleaned dataset not found. Running preprocess_data.py ...")
    from preprocess_data import clean_dataset

    clean_dataset()
    return pd.read_csv(CLEANED_DATA_PATH)


def assign_price_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Low/Medium/High category based on tertiles of the price distribution."""
    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"{PRICE_COLUMN} column is required to create price categories.")

    price_series = df[PRICE_COLUMN].astype(float)
    tertiles = price_series.quantile([0, 1 / 3, 2 / 3, 1]).values
    # Ensure strictly increasing bin edges
    tertiles[0] = price_series.min() - 1
    tertiles[-1] = price_series.max() + 1

    labels = ["Low", "Medium", "High"]
    df[PRICE_CATEGORY_COLUMN] = pd.cut(price_series, bins=tertiles, labels=labels, include_lowest=True)
    print("\nPrice category distribution:")
    print(df[PRICE_CATEGORY_COLUMN].value_counts().sort_index())
    return df


def build_preprocessor(df: pd.DataFrame):
    """Create a ColumnTransformer that encodes categorical features and scales numeric ones."""
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in {PRICE_COLUMN, PRICE_CATEGORY_COLUMN}]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != PRICE_COLUMN]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    return preprocessor, categorical_cols + numeric_cols


def train_linear_regression(X_train, y_train, X_test, category_to_int):
    """Train a linear regression model and convert predictions back to categories."""
    preprocessor, feature_cols = build_preprocessor(X_train)

    y_train_numeric = y_train.map(category_to_int)

    lin_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    lin_pipeline.fit(X_train, y_train_numeric)
    numeric_preds = lin_pipeline.predict(X_test).round().clip(0, len(category_to_int) - 1)
    inv_map = {v: k for k, v in category_to_int.items()}
    pred_categories = pd.Series(numeric_preds).map(inv_map)
    return pred_categories, lin_pipeline


def train_random_forest(X_train, y_train, X_test):
    """Train a random forest classifier to predict price categories."""
    preprocessor, feature_cols = build_preprocessor(X_train)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    rf_pipeline.fit(X_train, y_train)
    predictions = rf_pipeline.predict(X_test)
    return predictions, rf_pipeline


def main():
    df = ensure_clean_data()
    df = assign_price_categories(df)

    if df[PRICE_CATEGORY_COLUMN].isnull().any():
        raise ValueError("Price category contains NaN values. Check the binning logic.")

    feature_columns = [col for col in df.columns if col not in {PRICE_COLUMN, PRICE_CATEGORY_COLUMN}]
    X = df[feature_columns]
    y = df[PRICE_CATEGORY_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    category_to_int = {label: idx for idx, label in enumerate(sorted(y.unique()))}

    # Linear Regression (converted to categories)
    print("\n" + "=" * 50)
    print("LINEAR REGRESSION (PRICE CATEGORY VIA REGRESSION)")
    print("=" * 50)
    lin_preds, lin_model = train_linear_regression(X_train, y_train, X_test, category_to_int)
    print(classification_report(y_test, lin_preds, zero_division=0))
    print(f"Linear Regression Accuracy: {accuracy_score(y_test, lin_preds):.4f}")

    # Random Forest Classifier
    print("\n" + "=" * 50)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 50)
    rf_preds, rf_model = train_random_forest(X_train, y_train, X_test)
    print(classification_report(y_test, rf_preds, zero_division=0))
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")


if __name__ == "__main__":
    main()

