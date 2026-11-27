import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_dataset():
    return pd.read_csv("winequality-red.csv")

model, scaler = load_artifacts()
df = load_dataset()
features = joblib.load("feature_names.pkl")

st.title("🍷 Wine Quality Prediction")
st.write("Predict whether red wine is **Good** or **Bad** using a trained KNN Classifier.")
st.caption("Adjust the physicochemical properties below, then click Predict.")

values = []
with st.form("prediction_form"):
    for feature in features:
        values.append(
            st.slider(
                feature.title(),
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean()),
            )
        )
    submitted = st.form_submit_button("🍷 Predict Quality", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([values], columns=features)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    confidence = float(np.max(probabilities) * 100)

    if prediction == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")
    st.metric("Model confidence", f"{confidence:.2f}%")

with st.expander("Preview the dataset"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.caption("Built with Python, Streamlit, pandas and a KNN classification model.")
