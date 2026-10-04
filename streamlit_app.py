"""
streamlit_app.py
----------------
Streamlit front-end for the House Price Prediction app.

Before running this, make sure:
  1. You have trained the model:  python src/train.py
  2. The Flask API is running:    python app.py

Start the UI:
    streamlit run streamlit_app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

FLASK_URL = "http://127.0.0.1:5000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon=":house:",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("House Price Predictor")
st.markdown(
    """
    Enter the details of a house below and click **Predict Price** to get an
    estimated median house value based on the
    [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

    > **How it works:** Your inputs are sent to a Flask API that loads a
    > trained Random Forest model and returns the predicted price.
    """
)
st.divider()

# ---------------------------------------------------------------------------
# Input form  (st.form renders all widgets together reliably in Streamlit 1.40+)
# ---------------------------------------------------------------------------
st.subheader("House Features")

with st.form("input_form"):

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.slider(
            "Median Income (x$10,000)",
            min_value=0.5, max_value=15.0, value=5.0, step=0.1,
            help="Median income of the block group, in tens of thousands of dollars.",
        )
        house_age = st.slider(
            "House Age (years)",
            min_value=1.0, max_value=52.0, value=20.0, step=1.0,
            help="Median age of houses in the block group.",
        )
        ave_rooms = st.slider(
            "Average Rooms per House",
            min_value=1.0, max_value=20.0, value=5.0, step=0.1,
            help="Average number of rooms per household.",
        )
        ave_bedrms = st.slider(
            "Average Bedrooms per House",
            min_value=0.5, max_value=5.0, value=1.0, step=0.1,
            help="Average number of bedrooms per household.",
        )

    with col2:
        population = st.slider(
            "Block Population",
            min_value=50.0, max_value=10000.0, value=1000.0, step=50.0,
            help="Total population of the block group.",
        )
        ave_occup = st.slider(
            "Average Occupancy",
            min_value=1.0, max_value=10.0, value=2.5, step=0.1,
            help="Average number of household members.",
        )
        latitude = st.slider(
            "Latitude",
            min_value=32.54, max_value=41.95, value=37.0, step=0.01,
            help="Geographic latitude of the block group (California range).",
        )
        longitude = st.slider(
            "Longitude",
            min_value=-124.35, max_value=-114.31, value=-119.0, step=0.01,
            help="Geographic longitude of the block group (California range).",
        )

    submitted = st.form_submit_button(
        "Predict Price", use_container_width=True, type="primary"
    )

# ---------------------------------------------------------------------------
# Prediction  (runs only when the form is submitted)
# ---------------------------------------------------------------------------
if submitted:

    payload = {
        "MedInc":     float(med_inc),
        "HouseAge":   float(house_age),
        "AveRooms":   float(ave_rooms),
        "AveBedrms":  float(ave_bedrms),
        "Population": float(population),
        "AveOccup":   float(ave_occup),
        "Latitude":   float(latitude),
        "Longitude":  float(longitude),
    }

    # Check API health first
    try:
        health_resp = requests.get(f"{FLASK_URL}/health", timeout=3)
        health_resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the Flask API. "
            "Make sure it is running with:  `python app.py`"
        )
        st.stop()
    except Exception as exc:
        st.error(f"API health check failed: {exc}")
        st.stop()

    # Call /predict
    with st.spinner("Calculating..."):
        try:
            response = requests.post(
                f"{FLASK_URL}/predict",
                json=payload,
                timeout=10,
            )
            result = response.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if response.status_code == 200:
        predicted_price = result["predicted_price"]

        st.success("Prediction complete!")
        st.metric(
            label="Estimated Median House Value",
            value=predicted_price,
        )

        with st.expander("Inputs used for this prediction"):
            input_df = pd.DataFrame(list(payload.items()), columns=["Feature", "Value"])
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    else:
        st.error(f"API error: {result.get('error', 'Unknown error')}")

# ---------------------------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------------------------
st.divider()
with st.expander("What factors matter most? (Feature Importances)"):

    model_path = os.path.join(os.path.dirname(__file__), "model", "house_price_model.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude",
        ]
        fi_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="#3b82f6")
        ax.set_xlabel("Importance Score")
        ax.set_title("Random Forest - Feature Importances")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Train the model first (`python src/train.py`) to see feature importances.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Python | scikit-learn | Flask | Streamlit | "
    "Data: California Housing Dataset (sklearn)"
)
