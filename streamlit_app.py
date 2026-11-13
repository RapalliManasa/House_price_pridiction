import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Urban House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dt_pipeline.pkl"
DATA_PATH = BASE_DIR / "indian_urban_house_price_practice_100k.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
reference_df = load_reference_data()

# These are the exact columns used while training the saved model.
target_col = "Price_INR_Lakhs"
excluded_cols = ["Property_ID", target_col, "Price_Per_SqFt_INR"]
feature_cols = [c for c in reference_df.columns if c not in excluded_cols]

st.title("🏠 Indian Urban House Price Prediction")
st.write("Enter the property details below and click **Predict House Price**.")

st.info("This app uses your saved Decision Tree pipeline (dt_pipeline.pkl). The predicted value is in Indian lakhs.")

with st.form("prediction_form"):
    user_input = {}

    # Two-column form layout
    left, right = st.columns(2)
    for idx, col in enumerate(feature_cols):
        target_container = left if idx % 2 == 0 else right
        series = reference_df[col]

        with target_container:
            if (pd.api.types.is_object_dtype(series) or
                pd.api.types.is_categorical_dtype(series) or
                pd.api.types.is_bool_dtype(series)):
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    options = ["Unknown"]
                user_input[col] = st.selectbox(col.replace("_", " "), options)
            else:
                values = pd.to_numeric(series, errors="coerce").dropna()
                default = float(values.median()) if not values.empty else 0.0
                is_integer = pd.api.types.is_integer_dtype(series)
                if is_integer:
                    user_input[col] = st.number_input(
                        col.replace("_", " "),
                        value=int(round(default)),
                        step=1
                    )
                else:
                    user_input[col] = st.number_input(
                        col.replace("_", " "),
                        value=default,
                        step=0.1,
                        format="%.2f"
                    )

    submitted = st.form_submit_button("🔍 Predict House Price", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([[user_input[c] for c in feature_cols]], columns=feature_cols)
    try:
        prediction = float(model.predict(input_df)[0])
        st.success("Prediction completed!")
        st.metric("Estimated House Price", f"₹ {prediction:,.2f} Lakhs")
        st.caption(f"Approximate amount: ₹ {prediction * 100000:,.0f}")
    except Exception as error:
        st.error("Prediction failed. Check that the model and input columns match.")
        st.exception(error)

with st.expander("About this model"):
    st.write(
        "The application loads the Decision Tree regression pipeline saved by "
        "train1.py. Preprocessing (missing-value imputation, scaling and one-hot "
        "encoding) is handled inside the saved pipeline."
    )
