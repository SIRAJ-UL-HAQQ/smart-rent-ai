from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing_utils import squeeze_text_column


MODEL_PATH = Path(__file__).resolve().parent / "rental_ridge_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if MODEL_PATH.stat().st_size < 1000:
        raise RuntimeError(
            "The saved model file looks truncated/corrupted. "
            "Please rerun 03_model_training.py to regenerate rental_ridge_model.joblib."
        )

    try:
        return joblib.load(MODEL_PATH)
    except EOFError as exc:
        raise RuntimeError(
            "The saved model file is unreadable because it was truncated. "
            "Please rerun 03_model_training.py to regenerate rental_ridge_model.joblib."
        ) from exc


def build_input_frame():
    beds = st.sidebar.number_input("Beds", min_value=1, max_value=10, value=2)
    baths = st.sidebar.number_input("Baths", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
    sqft = st.sidebar.number_input("Sqft", min_value=300, max_value=5000, value=1200, step=50)
    neighborhood = st.sidebar.selectbox(
        "Neighborhood",
        ["Downtown", "Midtown", "Uptown", "Old Town", "Riverside"],
    )
    description = st.sidebar.text_area(
        "Description",
        value="Spacious 2bd with in-unit laundry and garage parking",
        height=120,
    )

    has_laundry = st.sidebar.checkbox("In-unit laundry")
    has_parking = st.sidebar.checkbox("Garage parking")
    pet_friendly = st.sidebar.checkbox("Pet-friendly")
    has_balcony = st.sidebar.checkbox("Balcony")
    hardwood_floors = st.sidebar.checkbox("Hardwood floors")

    input_df = pd.DataFrame(
        [{
            "beds": int(beds),
            "baths": float(baths),
            "sqft": int(sqft),
            "neighborhood": neighborhood,
            "latitude": 40.7300,
            "longitude": -73.9900,
            "description": description,
            "has_laundry": int(has_laundry),
            "has_parking": int(has_parking),
            "pet_friendly": int(pet_friendly),
            "has_balcony": int(has_balcony),
            "hardwood_floors": int(hardwood_floors),
        }]
    )

    return input_df


st.title("Rental Price Estimator")
st.write("Estimate the monthly rent for a rental listing using the saved Ridge pipeline.")

model = load_model()
input_df = build_input_frame()

if st.button("Estimate Rent"):
    prediction = model.predict(input_df)[0]
    st.metric("Estimated Monthly Rent", f"${prediction:,.0f}")
