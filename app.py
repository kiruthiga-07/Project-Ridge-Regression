import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏠 Housing Price Prediction (Ridge Regression)")

# -----------------------------
# Load dataset
# -----------------------------
try:
    df = pd.read_csv("housing.csv")
except:
    st.error("❌ Upload housing.csv to your GitHub repo")
    st.stop()

st.write("Dataset Preview", df.head())

# -----------------------------
# Handle missing values
# -----------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# Convert categorical
# -----------------------------
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# -----------------------------
# Features & target
# -----------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

feature_columns = X.columns  # IMPORTANT

# -----------------------------
# Split + Scale
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)

st.subheader("Model Performance")
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R²:", r2_score(y_test, y_pred))

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)

# -----------------------------
# Prediction UI
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

if st.button("Predict"):
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

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
