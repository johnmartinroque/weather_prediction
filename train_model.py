"""
train_model.py
----------------
Loads weather_forecast_data.csv, cleans it (drops Daily_Summary if present),
trains a Random Forest classifier to predict Rain / No Rain, saves the
trained model + feature list to models/, and saves evaluation charts to
visualizations/.

Run from the project root:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "weather_forecast_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

TARGET_COL = "Rain"
DROP_COLS = ["Daily_Summary"]  # dropped if present in the dataset


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop Daily_Summary (and any other unwanted columns) if present
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    return df


def main():
    df = load_data()
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())

    # -----------------------------------------------------------------
    # Prepare features / target
    # -----------------------------------------------------------------
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
    df[TARGET_COL] = df[TARGET_COL].map({"rain": 1, "no rain": 0})

    if df[TARGET_COL].isna().any():
        raise ValueError(
            "Target column contains values other than 'rain' / 'no rain'. "
            "Please check the Rain column."
        )

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------------------------------------------
    # Train model
    # -----------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["no rain", "rain"]))

    # -----------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # 1. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr_df = df.copy()
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # 2. Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="steelblue")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # 3. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["no rain", "rain"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 4. Class balance
    plt.figure(figsize=(5, 4))
    df[TARGET_COL].map({1: "rain", 0: "no rain"}).value_counts().plot(
        kind="bar", color=["#4C72B0", "#DD8452"]
    )
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    print(f"Saved visualizations to: {VIZ_DIR}")

    # -----------------------------------------------------------------
    # Save model + feature list
    # -----------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "rain_model.pkl")
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
