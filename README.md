# 🍷 Wine Quality Prediction

A Streamlit web application that predicts whether red wine is classified as **Good** or **Bad** using a trained K-Nearest Neighbors (KNN) classifier.

## Features
- Interactive sliders for 11 physicochemical wine properties
- KNN-based binary quality prediction
- Prediction confidence from model probabilities
- Dataset preview

## Run locally
1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Project files
- `app.py` — Streamlit application
- `knn_model.pkl` — trained KNN classifier
- `scaler.pkl` — fitted StandardScaler
- `feature_names.pkl` — ordered model feature names
- `winequality-red.csv` — red wine dataset
- `Wine_Quality_Predictor.ipynb` — model development notebook

## Deployment
Deploy this repository using [Streamlit Community Cloud](https://share.streamlit.io/). Select branch `main` and entrypoint file `app.py`.

## Note
The saved scikit-learn model was created with scikit-learn 1.7.2, so `requirements.txt` pins that version for compatibility.
