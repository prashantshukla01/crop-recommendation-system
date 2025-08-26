# 🌱 Crop Recommendation System

A machine learning-powered web application that recommends optimal crops based on location-specific climate and soil conditions.

## ✨ Features

- **Dual ML Architecture**: Predicts soil nutrients (NPK) and recommends crops
- **Real-Time Data**: Fetches historical weather data from Open-Meteo APIs
- **Location-Based**: Works with city names, coordinates, or current GPS location
- **Web Interface**: Beautiful, responsive Flask web application
- **Detailed Analytics**: Shows weather patterns, soil composition, and confidence scores

## 🛠️ Technology Stack

**Backend**: Python, Flask, Scikit-Learn, Pandas, NumPy  
**Frontend**: HTML, CSS, JavaScript  
**ML Models**: Random Forest Regressor & Classifier  
**APIs**: Open-Meteo Weather & Geocoding APIs

## 📦 File Structure

```bash
crop-recommendation-system/
├── app.py                 # Main Flask application
├── train_models.py        # Model training script
├── data_fetcher.py        # API integration and data processing
├── model/                 # Trained ML models
├── static/                # CSS styles
├── templates/             # HTML templates
└── data/                  # Dataset directory
```

## 📊 How It Works

Input Location: User enters location or uses GPS
Data Fetching: Gets historical weather and soil data from APIs
NPK Prediction: ML model predicts soil nutrient levels
Crop Recommendation: Second ML model suggests optimal crops
Results Display: Shows detailed analysis with confidence scores
🌐 Usage Examples

"Bangalore, India"
"12.9716, 77.5946"
"New York, USA"
"London, UK"
The system handles both precise coordinates and city names, with fallback to simulated data when APIs are unavailable.

Perfect for farmers, agricultural students, and gardening enthusiasts! 🌾
