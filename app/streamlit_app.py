"""
streamlit_app.py
------------------
Streamlit interface for the rain prediction models.
Always runs every trained algorithm saved by train_model.py into
models/ (logistic regression, decision tree, random forest, gradient
boosting) on the same input and shows all of their predictions side
by side. Weather values are entered via sliders bounded by the
observed data range.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os
import glob
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "weather_forecast_data.csv")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")

# Fallback ranges (computed from a sample of the dataset), used only if
# the original training CSV isn't available at DATA_PATH.
FALLBACK_RANGES = {
    "Temperature": (10.47, 34.71),
    "Humidity": (31.03, 99.54),
    "Wind_Speed": (0.58, 18.63),
    "Cloud_Cover": (0.79, 99.91),
    "Pressure": (980.11, 1049.74),
}


@st.cache_data
def discover_model_names():
    """Find every per-algorithm model saved by train_model.py.
    Excludes rain_model.pkl, which is just a copy of the best one."""
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl")))
    names = [
        os.path.splitext(os.path.basename(p))[0]
        for p in paths
        if os.path.basename(p) != "rain_model.pkl"
    ]
    if not names:
        st.error(f"No trained models found in `{MODELS_DIR}`. Run train_model.py first.")
        st.stop()
    return names


@st.cache_resource
def load_all_models(names):
    """Load every model, skipping (with a warning) any that fail -
    e.g. a scikit-learn version mismatch on one algorithm shouldn't
    take down the whole app."""
    models = {}
    feature_cols = None
    failed = []
    for name in names:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        try:
            bundle = joblib.load(path)
            models[name] = bundle["model"]
            feature_cols = bundle["feature_cols"]
        except Exception as e:
            failed.append((name, str(e)))
    return models, feature_cols, failed


@st.cache_data
def load_comparison():
    """Optional accuracy/F1 table saved by train_model.py."""
    if os.path.exists(COMPARISON_PATH):
        try:
            return pd.read_csv(COMPARISON_PATH).set_index("algorithm")
        except Exception:
            return None
    return None


@st.cache_data
def load_ranges(feature_cols):
    """Return {feature: (min, max, mean)} using the training data if
    present, otherwise fall back to hardcoded ranges from the sample."""
    ranges = {
        col: (*FALLBACK_RANGES.get(col, (0.0, 100.0)),
              sum(FALLBACK_RANGES.get(col, (0.0, 100.0))) / 2)
        for col in feature_cols
    }

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


def pretty(name):
    return name.replace("_", " ").title()


def main():
    st.set_page_config(page_title="Rain Prediction", page_icon="🌧️", layout="centered")

    st.title("🌧️ Rain Prediction")
    st.write(
        "Adjust the weather values below and click **Predict** to see what "
        "every trained model expects."
    )

    model_names = discover_model_names()
    models, feature_cols, failed = load_all_models(tuple(model_names))

    if failed:
        for name, err in failed:
            st.warning(
                f"Couldn't load **{pretty(name)}** (skipped): {err.splitlines()[0]}"
            )

    if not models:
        st.error("No models could be loaded. Check the warnings above.")
        st.stop()

    if feature_cols is None:
        st.stop()

    comparison = load_comparison()
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

        rows = []
        for name, model in models.items():
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            row = {
                "Algorithm": pretty(name),
                "Prediction": "☔ RAIN" if pred == 1 else "☀️ NO RAIN",
                "Confidence": f"{proba[pred] * 100:.1f}%",
            }
            if comparison is not None and name in comparison.index:
                row["Test accuracy"] = f"{comparison.loc[name, 'accuracy'] * 100:.1f}%"
            rows.append(row)

        results_df = pd.DataFrame(rows)
        st.subheader("Results")
        st.table(results_df.set_index("Algorithm"))

        rain_votes = sum(1 for r in rows if "RAIN" in r["Prediction"] and "NO" not in r["Prediction"])
        st.caption(f"{rain_votes} of {len(rows)} models predict rain.")

        with st.expander("Input values used"):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()