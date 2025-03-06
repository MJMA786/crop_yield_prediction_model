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
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #2E7D32;
        }
        .prediction-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #2E7D32;
            animation: bounceIn 0.8s ease-in-out;
        }
        .recommendation-box {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='title'>🌾 Smart Crop Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the best crop yield based on your location and environmental factors.</p>", unsafe_allow_html=True)


# Enhanced Information Box (Styled & Aligned)

st.markdown( """ <style> 
.info-box { 
background: linear-gradient(to right, #e8f5e9, #f1f8e9); 
padding: 25px; 
border-radius: 15px; 
margin-bottom: 20px; 
border-left: 6px solid #2E7D32; 
box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.1); }
.info-title { 
font-size: 24px; 
font-weight: bold; 
color: #1B5E20; 
text-align: center; 
margin-bottom: 12px; } 
.info-section { 
font-size: 18px; 
font-weight: bold; 
color: #388E3C; 
margin-top: 15px; } 
.info-list { 
padding-left: 25px; 
font-size: 16px; 
line-height: 1.6; 
color: #333; } 
.highlight { 
color: #1B5E20; 
font-weight: bold; 
} </style>
<div class='info-box'>
    <div class='info-title'>🌾 Smart Crop Predictor – ML-Powered Yield Estimator</div>
    <p>Welcome to <b class='highlight'>Smart Crop Predictor</b>, an advanced <b>Machine Learning (ML)</b> application designed to help farmers, researchers, and agronomists make data-driven crop yield predictions.</p>
    <div class='info-section'>🚀 Key Features</div>
    <ul class='info-list'>
        <li>✅ <b>State & District:</b> Location-based crop productivity insights.</li>
        <li>✅ <b>Season Selection:</b> Choose the right season for optimal yield.</li>
        <li>✅ <b>Crop Type:</b> Identify yield variations for different crops.</li>
        <li>✅ <b>Environmental Factors:</b> Temperature, humidity, and soil conditions.</li>
        <li>✅ <b>Farm Area:</b> Yield estimation based on cultivated land size.</li>
    </ul>
    <div class='info-section'>🧠 How It Works</div>
    <ol class='info-list'>
        <li>1️⃣ Enter your <b>location</b>, <b>season</b>, and <b>crop details</b>.</li>
        <li>2️⃣ Adjust key environmental parameters.</li>
        <li>3️⃣ Click <b>‘Predict Crop Yield’</b> to generate an estimate.</li>
        <li>4️⃣ Use insights for smarter agricultural decisions!</li>
    </ol>
    <div class='info-section'>🔬 Why This Matters?</div>
    <ul class='info-list'>
        <li>✅ Supports <b>data-driven farming</b> for improved yields.</li>
        <li>✅ Helps <b>farmers optimize resources</b> and maximize profits.</li>
        <li>✅ Aids policymakers in <b>sustainable agriculture planning</b>.</li>
    </ul>
    <div class='info-section'>🧑‍💻 How the ML Model Works</div>
    <p>The prediction model is built on <b class='highlight'>historical crop yield data</b> and advanced machine learning techniques.</p>
    <ul class='info-list'>
        <li>🔹 <b>Model Type:</b> Supervised Learning (e.g., <b>Random Forest</b>).</li>
        <li>🔹 <b>Key Inputs:</b> 
            <ul>
                <li>Climate Factors: Temperature, Humidity, Rainfall.</li>
                <li>Soil Conditions: Moisture level, Cultivation area.</li>
                <li>Geographical Data: State, District, Season.</li>
                <li>Crop Type: Encoded for accurate predictions.</li>
            </ul>
        </li>
    </ul>
    <div class='info-section'>🌱 Smart Farming, Smarter Decisions!</div>
    <p>⚡ <b>Leverage AI-powered precision farming for better yields & sustainability.</b></p>
</div>
""",unsafe_allow_html=True)

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

data = { "Parameter": [ "🌍 State", "🏙 District", "🌱 Season", "📅 Crop Year", "🌾 Crop", "🌡 Temperature (°C)", "💧 Humidity (%)", "🌿 Soil Moisture (%)", "🌾 Area (acres)" ], "Value": [state, district, season, crop_year, crop, temperature, humidity, soil_moisture, area] }

df = pd.DataFrame(data) 
st.dataframe(df, height=350, width=600)

# Encode User Inputs
state_encoded = label_encoders["State"].transform([state])[0]
district_encoded = label_encoders["District"].transform([district])[0]
season_encoded = label_encoders["Season"].transform([season])[0]
crop_encoded = label_encoders["Crop"].transform([crop])[0]

# Recommendation System
def get_recommendations(predicted_yield, crop):
    if predicted_yield < 2:
        return f"🚜 **Low Yield Expected!** \n- Use organic fertilizers and irrigation. \n- Consider **drought-resistant varieties** of {crop}. \n- Improve **soil moisture retention** by adding mulch."
    elif 2 <= predicted_yield < 4:
        return f"🌱 **Moderate Yield Expected.** \n- Optimize **fertilizer** application based on soil tests. \n- Implement **pest control measures** to protect {crop}. \n- Maintain proper **crop rotation** to improve soil health."
    else:
        return f"🌾 **High Yield Expected!** \n- Ensure regular **water supply** to sustain yield. \n- Store harvested {crop} properly to **avoid post-harvest losses**. \n- Consider **selling in bulk** for better market rates."

# Predict Crop Yield
if st.button('🚜 Predict Crop Yield', key="predict_main"):
    input_data = np.array([[temperature, humidity, soil_moisture, area, crop_encoded, state_encoded, district_encoded, season_encoded]])

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
