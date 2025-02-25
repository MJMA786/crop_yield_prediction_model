# 🌾 Smart Crop Predictor

## 🚀 Overview
The **Smart Crop Predictor** is a Machine Learning-powered application designed to assist farmers, agronomists, and researchers in predicting crop yield based on environmental and regional factors. The app leverages a trained ML model to analyze key parameters such as temperature, humidity, soil moisture, and geographical location to estimate crop yield.

## 🎯 Features
- ✅ **Location-Based Prediction:** Select **State** and **District** for location-specific insights.
- ✅ **Seasonal Variations:** Choose the best season for an accurate yield estimation.
- ✅ **Crop Selection:** Supports multiple crop types with unique growth patterns.
- ✅ **Environmental Factors:** Predict yield based on **Temperature, Humidity, and Soil Moisture**.
- ✅ **Farm Area Calculation:** Estimates yield based on the total area of cultivation.

## 🏗 Tech Stack
- **Python** (Streamlit for UI, Pandas & NumPy for data processing)
- **Machine Learning Model:** Random Forest Regressor
- **Scikit-Learn** for preprocessing and modeling
- **Git LFS** for handling large files (dataset and model)

## 📌 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/MJMA786/crop_yield_prediction_model
cd crop_yield_prediction
```

### 2️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed, then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run app.py
```
The app will launch in your default browser.

## 📊 How It Works
1. Select **State, District, Season, and Crop Type**.
2. Adjust environmental factors: **Temperature, Humidity, Soil Moisture**.
3. Click **Predict Crop Yield** to estimate yield in tons.
4. Use the results to make **data-driven farming decisions**.

## 🖼 Screenshots


## 📁 Project Structure
```
📦 crop_yield_prediction
├── 📄 app.py                 # Main application file
├── 📄 requirements.txt       # Required dependencies
├── 📄 crop_yield_prediction_model.pkl  # Trained ML model
├── 📊 Crop Prediction dataset.xlsx  # Dataset used for training
├── 📄 rfr(deployment_code).ipynb  # Jupyter Notebook for ML model training
├── 📄 README.md   # Project Documentation
├── 📄 LICENSE  # MIT license        
```

## 🏆 Future Enhancements
- 📌 Support for more crop varieties
- 📌 Integration of real-time weather API
- 📌 Advanced ML models for improved accuracy

## 🤝 Contributing
Feel free to fork this repository, raise issues, or submit PRs to improve the project!

## 📜 License
This project is licensed under the **MIT License**.

---
🚀 **Developed & Maintained by MJMA**

