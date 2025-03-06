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

# UI Styling
st.markdown("""
    <style>
        /* Page Background */
        body {
            background: #f1f8e9;
            font-family: 'Arial', sans-serif;
        }

        /* Title */
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2E7D32;
            animation: fadeIn 1s ease-in-out;
        }

        /* Main Prediction Box */
        .prediction-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #2E7D32;
            animation: bounceIn 1s ease-in-out;
        }

        /* Information Box */
        .info-box {
            background: linear-gradient(to right, #e8f5e9, #f1f8e9); 
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 6px solid #2E7D32;
            box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.1);
            animation: fadeInUp 1.5s ease-in-out;
        }

        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes bounceIn {
            0% { transform: scale(0.8); opacity: 0; }
            60% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(1); }
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Recommendation Box */
        .recommendation-box {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 15px;
            animation: fadeInUp 1.5s ease-in-out;
        }

        /* Button Styles */
        .predict-button {
            background-color: #2E7D32;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .predict-button:hover {
            background-color: #1B5E20;
            transform: scale(1.05);
        }

        /* DataFrame Styles */
        .dataframe {
            margin-top: 30px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='title'>🌾 Smart Crop Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the best crop yield based on your location and environmental factors.</p>", unsafe_allow_html=True)

# Function to generate recommendations based on predicted yield
def get_recommendations(predicted_yield, crop):
    """
    Generate recommendations based on predicted crop yield.
    """
    if predicted_yield < 2:
        recommendation = f"💧 Watering and fertilization might be required for {crop} to boost yield."
    elif predicted_yield < 4:
        recommendation = f"🌱 {crop} yield is average. Ensure proper irrigation and pest control."
    elif predicted_yield < 6:
        recommendation = f"🌾 Great yield for {crop}. Maintain current agricultural practices."
    else:
        recommendation = f"🚜 Excellent yield for {crop}. Consider optimizing harvesting and distribution."
    
    return recommendation

# Sidebar Inputs
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

# Display Selected Inputs
st.subheader("📝 Selected Inputs")
data = {
    "Parameter": [
        "🌍 State", "🏙 District", "🌱 Season", "📅 Crop Year", "🌾 Crop", 
        "🌡 Temperature (°C)", "💧 Humidity (%)", "🌿 Soil Moisture (%)", "🌾 Area (acres)"
    ],
    "Value": [
        state, district, season, crop_year, crop, temperature, humidity, soil_moisture, area
    ]
}

# Ensure values are strings for compatibility
df = pd.DataFrame(data)
df['Value'] = df['Value'].apply(lambda x: str(x) if isinstance(x, (int, float)) else x)

st.dataframe(df, height=350, width=600, use_container_width=True)

# Predict Crop Yield
if st.button('🚜 Predict Crop Yield', key="predict_main"):
    input_data = np.array([[temperature, humidity, soil_moisture, area, label_encoders["Crop"].transform([crop])[0], label_encoders["State"].transform([state])[0], label_encoders["District"].transform([district])[0], label_encoders["Season"].transform([season])[0]]])

    with st.spinner("Predicting... Please wait ⏳"):
        time.sleep(1)  # Simulate processing time
        prediction = model.predict(input_data)
        predicted_yield = prediction[0]

    # Display Yield Prediction
    st.markdown(f"""
        <div class='prediction-box'>
            🌾 Estimated Crop Yield: <b>{predicted_yield:.2f}</b> Tons
        </div>
    """, unsafe_allow_html=True)

    # Display Recommendations
    recommendations = get_recommendations(predicted_yield, crop)
    st.markdown(f"""
        <div class='recommendation-box'>
            <h4>🌱 Recommended Actions:</h4>
            <p>{recommendations}</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align:center; color:#888888; margin-top:30px;'>🌱 Powered by MJMA</p>", unsafe_allow_html=True)
