import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Housing Ridge Regression", layout="wide")

@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame, housing.feature_names

# Header
st.title("🏡 California Housing Price Predictor")
st.markdown("Using Ridge Regression to handle multicollinearity and predict housing values.")

# Load data
df, feature_names = load_data()

# Sidebar - Hyperparameters
st.sidebar.header("Model Settings")
alpha = st.sidebar.slider("Ridge Alpha (Penalty)", 0.01, 20.0, 1.0)

# Prepare Data
X = df[feature_names]
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (Critical for Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Model
model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics Display
m1, m2 = st.columns(2)
m1.metric("R² Score (Accuracy)", f"{r2_score(y_test, y_pred):.4f}")
m2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.4f}")

# Visualizations
st.subheader("Model Insights")
col1, col2 = st.columns(2)

with col1:
    st.write("**Feature Importance (Coefficients)**")
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coeff': model.coef_}).sort_values(by='Coeff')
    fig1, ax1 = plt.subplots()
    sns.barplot(data=coef_df, x='Coeff', y='Feature', palette='viridis', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("**Correlation Heatmap**")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# Custom Prediction Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Live Prediction")
user_inputs = []
for col in feature_names:
    val = st.sidebar.number_input(f"{col}", value=float(df[col].median()))
    user_inputs.append(val)

if st.sidebar.button("Predict Price"):
    scaled_input = scaler.transform([user_inputs])
    prediction = model.predict(scaled_input)[0]
    st.sidebar.success(f"Estimated Value: ${prediction * 100000:,.2f}")
