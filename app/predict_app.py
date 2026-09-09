"""
predict_app.py
----------------
Interactive CLI app: loads the trained model from models/rain_model.pkl
and lets the user type in weather values to get a Rain / No Rain prediction.

Shows the observed min/max range for each feature (from the training
data, if available) so the user knows what values are reasonable to enter.

Run from the project root:
    python app/predict_app.py
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "weather.csv")

# Fallback ranges (computed from the sample dataset), used if the
# original training CSV isn't available at DATA_PATH.
FALLBACK_RANGES = {
    "Temperature": (10.47, 34.71),
    "Humidity": (31.03, 99.54),
    "Wind_Speed": (0.58, 18.63),
    "Cloud_Cover": (0.79, 99.91),
    "Pressure": (980.11, 1049.74),
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_model.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


def load_ranges(feature_cols):
    """Return {feature: (min, max)} using the training data if present,
    otherwise fall back to hardcoded ranges from the sample dataset."""
    ranges = dict(FALLBACK_RANGES)

    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            for col in feature_cols:
                if col in df.columns:
                    ranges[col] = (df[col].min(), df[col].max())
        except Exception:
            pass  # silently keep fallback ranges if the file can't be read

    return {col: ranges.get(col) for col in feature_cols}


def get_float_input(prompt, value_range=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a valid number.")
            continue

        if value_range is not None:
            low, high = value_range
            if not (low <= value <= high):
                print(
                    f"  Warning: {value} is outside the typical observed "
                    f"range ({low:.2f} to {high:.2f}). Continuing anyway."
                )
        return value


def main():
    model, feature_cols = load_model()
    ranges = load_ranges(feature_cols)

    print("=" * 50)
    print("  RAIN PREDICTION")
    print("=" * 50)
    print("Enter values for the following features.")
    print("Typical ranges (from the training data) are shown as a guide:\n")
    for col in feature_cols:
        r = ranges.get(col)
        if r:
            print(f"  - {col}: {r[0]:.2f} to {r[1]:.2f}")
        else:
            print(f"  - {col}: (no range info available)")
    print()

    while True:
        values = {}
        for col in feature_cols:
            r = ranges.get(col)
            hint = f" [{r[0]:.2f} - {r[1]:.2f}]" if r else ""
            values[col] = get_float_input(f"{col}{hint}: ", r)

        input_df = pd.DataFrame([values], columns=feature_cols)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label = "RAIN" if pred == 1 else "NO RAIN"
        confidence = proba[pred] * 100

        print(f"\nPrediction: {label}  (confidence: {confidence:.1f}%)\n")

        again = input("Predict another? (y/n): ").strip().lower()
        if again != "y":
            break
        print()

    print("Goodbye!")


if __name__ == "__main__":
    main()