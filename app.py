import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

# Page Configuration
st.set_page_config(
    page_title="Smart Crop Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Model
try:
    model = joblib.load('crop_yield_prediction_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Encode categories
states = ["Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi"]
seasons = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"]
crops = ["Rice", "Wheat", "Maize", "Barley", "Soybean", "Banana", "Sugarcane", "Turmeric"]

district_map = {
    "Andhra Pradesh": ["ANANTAPUR", "CHITTOOR", "EAST GODAVARI", "GUNTUR", "KADAPA"],
    "Assam": ["Baksa", "Barpeta"],
    "Bihar": ["Araria", "Arwal"],
    "Chhattisgarh": ["Balod", "Bastar"],
    "Delhi": ["Central Delhi", "East Delhi"]
}

label_encoders = {category: LabelEncoder() for category in ["State", "District", "Season", "Crop"]}
label_encoders["State"].fit(states)
label_encoders["District"].fit([d for districts in district_map.values() for d in districts])
label_encoders["Season"].fit(seasons)
label_encoders["Crop"].fit(crops)

# Styling
st.markdown("<h1 style='text-align:center; color:#2E7D32;'>🌾 Smart Crop Predictor</h1>", unsafe_allow_html=True)

# Sidebar for Inputs
st.sidebar.header("📍 Location & Season")
state = st.sidebar.selectbox("State", states)
district = st.sidebar.selectbox("District", district_map[state])
season = st.sidebar.selectbox("Season", seasons)
crop_year = st.sidebar.number_input("Crop Year", min_value=2000, max_value=3000, value=2026, step=1)

st.sidebar.header("🌾 Crop Selection")
crop = st.sidebar.selectbox("Select Crop", crops)

st.sidebar.header("🌦 Environmental Factors")
temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0, step=0.1)
humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 60.0, step=0.1)
soil_moisture = st.sidebar.slider('Soil Moisture (%)', 0.0, 100.0, 50.0, step=0.1)
area = st.sidebar.number_input('Area (acres)', min_value=0.1, max_value=1000.0, value=4.0, step=0.1)

# Encode User Inputs
state_encoded = label_encoders["State"].transform([state])[0]
district_encoded = label_encoders["District"].transform([district])[0]
season_encoded = label_encoders["Season"].transform([season])[0]
crop_encoded = label_encoders["Crop"].transform([crop])[0]

# Predict Crop Yield
if st.button('🚜 Predict Crop Yield', key="predict_main"):
    input_data = np.array([[temperature, humidity, soil_moisture, area, crop_encoded, state_encoded, district_encoded, season_encoded]])

    with st.spinner("Predicting... Please wait ⏳"):
        time.sleep(1)  # Simulate processing time
        prediction = model.predict(input_data)[0]

    # Categorize yield
    if prediction < 1.5:
        yield_category = "Low"
        recommendation = "Consider using organic fertilizers and optimizing irrigation methods."
    elif 1.5 <= prediction < 3.5:
        yield_category = "Moderate"
        recommendation = "Yield is moderate. Improve soil health and monitor moisture levels."
    else:
        yield_category = "High"
        recommendation = "Great yield potential! Maintain proper fertilization and irrigation techniques."

    # Additional Environmental Recommendations
    env_recommendations = []
    if temperature > 35:
        env_recommendations.append("🌡️ Temperature is high. Use mulching and shade to reduce heat stress.")
    if humidity > 80:
        env_recommendations.append("💧 High humidity detected. Monitor for fungal infections and use suitable fungicides.")
    if soil_moisture < 30:
        env_recommendations.append("🌱 Low soil moisture. Consider drip irrigation for optimal growth.")

    # Best Harvesting Time
    harvest_time = {
        "Rice": "September - November",
        "Wheat": "March - May",
        "Maize": "October - November",
        "Barley": "April - June",
        "Soybean": "September - October",
        "Banana": "Year-round",
        "Sugarcane": "November - April",
        "Turmeric": "January - February"
    }
    best_harvest = harvest_time.get(crop, "Seasonal")

    # Fertilizer Recommendation
    fertilizer_suggestions = {
        "Rice": "Use Nitrogen, Phosphorus, and Potassium fertilizers in balanced amounts.",
        "Wheat": "Apply Urea, DAP, and MOP for better yield.",
        "Maize": "Ensure adequate Nitrogen supply for healthy growth.",
        "Soybean": "Use Phosphorus-rich fertilizers for improved pod formation.",
        "Banana": "Regular application of organic compost and potassium-rich fertilizers is recommended.",
        "Sugarcane": "Use nitrogen-based fertilizers along with micronutrients.",
        "Turmeric": "Apply farmyard manure and phosphorus fertilizers for better root development."
    }
    fertilizer_recommendation = fertilizer_suggestions.get(crop, "Use organic fertilizers to improve soil health.")

    # Display Results
    st.markdown(f"""
        <div style='background-color:#f1f8e9; padding:20px; border-radius:10px; text-align:center;'>
            <h2>🌾 Estimated Crop Yield: <b>{prediction:.2f} Tons</b></h2>
            <h3 style='color: {"red" if yield_category == "Low" else "orange" if yield_category == "Moderate" else "green"}'>
                {yield_category} Yield
            </h3>
            <p><b>📌 Recommendation:</b> {recommendation}</p>
        </div>
    """, unsafe_allow_html=True)

    # Additional Recommendations
    if env_recommendations:
        st.markdown("### 🌍 Environmental Advice")
        for rec in env_recommendations:
            st.write(f"✅ {rec}")

    st.markdown(f"### ⏳ Best Harvesting Time: {best_harvest}")
    st.markdown(f"### 🌱 Fertilizer Recommendation: {fertilizer_recommendation}")

# Footer
st.markdown("<p style='text-align:center; color:#888888; margin-top:30px;'>🌱 Powered by MJMA</p>", unsafe_allow_html=True)
