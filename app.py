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
st.set_page_config(page_title="CA Housing Ridge Analysis", layout="wide")

@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame, housing.feature_names

# --- DATA LOADING ---
df, feature_names = load_data()

st.title("🏡 California Housing Ridge Regression Tool")
st.markdown("---")

# --- SIDEBAR: INPUTS & MODEL TUNING ---
st.sidebar.header("🕹️ Model Controls")
alpha = st.sidebar.slider("Ridge Penalty (Alpha)", 0.1, 50.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.header("📍 Predict for a New Area")

# Custom Inputs with logical rounding
user_input_dict = {}
for col in feature_names:
    median_val = float(df[col].median())
    if col in ['Population', 'HouseAge']:
        # Use integers for population and age
        val = st.sidebar.number_input(f"{col}", value=int(median_val), step=1)
    else:
        # Keep decimals for income and coordinates
        val = st.sidebar.number_input(f"{col}", value=median_val, format="%.4f")
    user_input_dict[col] = val

# --- MODEL LOGIC ---
X = df[feature_names]
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- DASHBOARD LAYOUT ---
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Model Accuracy (R²)", f"{r2_score(y_test, y_pred):.3f}")
col2.metric("Prediction Error (MSE)", f"{mean_squared_error(y_test, y_pred):.3f}")
col3.metric("Regularization", f"Alpha: {alpha}")

# --- NEW VISUALIZATIONS ---
tab1, tab2, tab3 = st.tabs(["🌍 California Price Map", "📊 Analysis Graphs", "📈 Accuracy Check"])

with tab1:
    st.subheader("Geographic Distribution of Prices")
    fig_map, ax_map = plt.subplots(figsize=(10, 7))
    # Scatter plot using Lat/Long
    scatter = ax_map.scatter(df['Longitude'], df['Latitude'], alpha=0.4,
                            c=df['MedHouseVal'], cmap='jet', s=df['Population']/100)
    plt.colorbar(scatter, label='Median House Value ($100k)')
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    st.pyplot(fig_map)
    st.info("Bigger bubbles = Higher Population | Redder color = Higher Price")

with tab2:
    sub1, sub2 = st.columns(2)
    with sub1:
        st.write("**Feature Weights**")
        coef_df = pd.DataFrame({'Feature': feature_names, 'Impact': model.coef_}).sort_values(by='Impact')
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(data=coef_df, x='Impact', y='Feature', palette='magma', ax=ax_bar)
        st.pyplot(fig_bar)
    with sub2:
        st.write("**Feature Correlations**")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df.corr(), cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

with tab3:
    st.subheader("Actual vs. Predicted Prices")
    fig_res, ax_res = plt.subplots()
    ax_res.scatter(y_test, y_pred, alpha=0.3, color='green')
    ax_res.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax_res.set_xlabel("Actual Values")
    ax_res.set_ylabel("Predicted Values")
    st.pyplot(fig_res)

# --- PREDICTION BUTTON ---
if st.sidebar.button("✨ Calculate Estimate"):
    input_df = pd.DataFrame([user_input_dict])
    input_scaled = scaler.transform(input_df)
    final_pred = model.predict(input_scaled)[0]
    
    st.sidebar.markdown("---")
    st.sidebar.success(f"### Predicted Price: ${final_pred * 100000:,.2f}")
    if final_pred > df['MedHouseVal'].mean():
        st.sidebar.write("🏠 This is above the state average.")
    else:
        st.sidebar.write("📉 This is below the state average.")
