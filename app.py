# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Title
# -----------------------------
st.title("🏠 Housing Price Prediction using Ridge Regression")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload housing.csv file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload the housing.csv dataset to continue")
    st.stop()

# -----------------------------
# Load dataset (SAFE)
# -----------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Error reading CSV file")
    st.stop()

# -----------------------------
# Check required column
# -----------------------------
if "median_house_value" not in df.columns:
    st.error("Dataset must contain 'median_house_value' column")
    st.stop()

# -----------------------------
# Preview
# -----------------------------
st.subheader("Dataset Preview")
st.write(df.head())

# -----------------------------
# Missing Values
# -----------------------------
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# Convert Categorical (SAFE)
# -----------------------------
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# -----------------------------
# Statistics
# -----------------------------
st.subheader("Statistical Summary")
st.write(df.describe())

# -----------------------------
# Correlation Heatmap (SAFE)
# -----------------------------
st.subheader("Correlation Heatmap")

try:
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)
except:
    st.warning("Heatmap could not be displayed")

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

feature_columns = X.columns

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Ridge Model
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"MSE: {mse:.2f}")
st.write(f"R² Score: {r2:.4f}")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance (Coefficients)")

coeff_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.write(coeff_df)

# -----------------------------
# Actual vs Predicted Plot (SAFE)
# -----------------------------
st.subheader("Actual vs Predicted")

try:
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred)
    ax2.set_xlabel("Actual Prices")
    ax2.set_ylabel("Predicted Prices")
    st.pyplot(fig2)
except:
    st.warning("Could not display scatter plot")

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("Predict House Price")

median_income = st.number_input("Median Income", 0.0, 15.0, 5.0)
housing_median_age = st.number_input("House Age", 1, 100, 20)
total_rooms = st.number_input("Total Rooms", 1.0, 10000.0, 2000.0)
total_bedrooms = st.number_input("Total Bedrooms", 1.0, 5000.0, 500.0)
population = st.number_input("Population", 1, 50000, 3000)
households = st.number_input("Households", 1, 5000, 1000)
latitude = st.number_input("Latitude", 30.0, 45.0, 35.0)
longitude = st.number_input("Longitude", -130.0, -110.0, -120.0)

# -----------------------------
# Prediction Logic (SAFE)
# -----------------------------
if st.button("Predict Price"):

    try:
        input_dict = {col: 0 for col in feature_columns}

        input_dict.update({
            "median_income": median_income,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "latitude": latitude,
            "longitude": longitude
        })

        input_df = pd.DataFrame([input_dict])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error("Prediction failed. Please check input values.")
