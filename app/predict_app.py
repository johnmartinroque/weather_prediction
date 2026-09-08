"""
predict_app.py
----------------
Interactive CLI app: loads the trained model from models/rain_model.pkl
and lets the user type in weather values to get a Rain / No Rain prediction.

Run from the project root:
    python app/predict_app.py
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain_model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_model.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


def get_float_input(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number.")


def main():
    model, feature_cols = load_model()

    print("=" * 50)
    print("  RAIN PREDICTION")
    print("=" * 50)
    print(f"Enter values for: {', '.join(feature_cols)}\n")

    while True:
        values = {}
        for col in feature_cols:
            values[col] = get_float_input(f"{col}: ")

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
