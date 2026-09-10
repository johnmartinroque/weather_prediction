"""
streamlit_app.py
------------------
Streamlit interface for the rain prediction model.
Loads the trained model from models/rain_model.pkl and lets the user
enter weather values (via sliders bounded by the observed data range)
to get a Rain / No Rain prediction.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

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


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run train_model.py first."
        )
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


@st.cache_data
def load_ranges(feature_cols):
    """Return {feature: (min, max, mean)} using the training data if
    present, otherwise fall back to hardcoded ranges from the sample."""
    ranges = {col: (*FALLBACK_RANGES.get(col, (0.0, 100.0)),
                     sum(FALLBACK_RANGES.get(col, (0.0, 100.0))) / 2)
              for col in feature_cols}

    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            for col in feature_cols:
                if col in df.columns:
                    ranges[col] = (
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].mean()),
                    )
        except Exception:
            pass  # keep fallback ranges if the file can't be read

    return ranges


def main():
    st.set_page_config(page_title="Rain Prediction", page_icon="🌧️", layout="centered")

    st.title("🌧️ Rain Prediction")
    st.write(
        "Adjust the weather values below and click **Predict** to see "
        "whether the model expects rain."
    )

    model, feature_cols = load_model()
    ranges = load_ranges(feature_cols)

    st.subheader("Weather inputs")
    values = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_cols):
        low, high, mean = ranges[col]
        with cols[i % 2]:
            values[col] = st.slider(
                col,
                min_value=round(low, 2),
                max_value=round(high, 2),
                value=round(mean, 2),
                help=f"Observed range: {low:.2f} to {high:.2f}",
            )

    st.divider()

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values], columns=feature_cols)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        label = "RAIN" if pred == 1 else "NO RAIN"
        confidence = proba[pred] * 100

        if pred == 1:
            st.success(f"### ☔ {label}  \nConfidence: {confidence:.1f}%")
        else:
            st.info(f"### ☀️ {label}  \nConfidence: {confidence:.1f}%")

        with st.expander("Input values used"):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()