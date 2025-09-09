from flask import Flask, jsonify, request, render_template, session, redirect, url_for
import joblib
import numpy as np
from data_fetcher import SoilDataFetcher
from database import DatabaseManager
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'soil-analysis-secret-key-2024')

# Add custom JSON filter for templates
@app.template_filter('tojson')
def to_json(obj):
    return json.dumps(obj)

# Global variables for models and database
models_loaded = False
crop_model = None
label_encoder = None
data_fetcher = SoilDataFetcher()
db = DatabaseManager()

def get_current_user():
    """Get current user from session or default user"""
    user_id = session.get('user_id')
    if user_id:
        return db.get_user(user_id=user_id)
    else:
        # Use default demo user
        user = db.get_default_user()
        if user:
            session['user_id'] = user['id']
        return user

def load_models():
    """Load crop prediction model and label encoder"""
    global crop_model, label_encoder, models_loaded
    
    try:
        crop_model = joblib.load('model/crop_predictor.joblib')
        label_encoder = joblib.load('model/label_encoder.joblib')
        models_loaded = True
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page with API documentation and interface"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard with analysis history and interactive map"""
    user = get_current_user()
    if not user:
        return redirect(url_for('index'))
    
    # Get user statistics and analysis history
    stats = db.get_dashboard_stats(user['id'])
    analyses = db.get_user_analyses(user['id'], limit=10)
    locations = db.get_analysis_locations(user['id'])
    
    return render_template('dashboard.html', 
                         user=user, 
                         stats=stats, 
                         analyses=analyses,
                         locations=locations)

@app.route('/profile')
def profile():
    """User profile page"""
    user = get_current_user()
    if not user:
        return redirect(url_for('index'))
    
    preferences = db.get_user_preferences(user['id'])
    return render_template('profile.html', user=user, preferences=preferences)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'service': 'soil-analysis-api'
    })

@app.route('/analyze-soil', methods=['POST'])
def analyze_soil():
    """Main endpoint for soil analysis and crop recommendation"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please restart the service.'}), 500
        
    try:
        # Get location from request (support both JSON and form data)
        if request.is_json:
            data = request.get_json()
            location = data.get('location')
        else:
            location = request.form.get('location')

        if not location:
            return jsonify({
                'success': False,
                'error': 'Location parameter is required',
                'example': {'location': 'Delhi, India'}
            }), 400

        # Fetch comprehensive soil and weather data
        soil_weather_data, location_name = data_fetcher.get_location_data(location)

        # Get coordinates for database storage
        lat, lng, _ = data_fetcher.geocode_location(location)

        # Prepare input for crop prediction (simplified direct approach)
        crop_input = np.array([[
            soil_weather_data.get('N', 40.0),
            soil_weather_data.get('P', 30.0), 
            soil_weather_data.get('K', 25.0),
            soil_weather_data['temperature'],
            soil_weather_data['humidity'],
            soil_weather_data.get('ph', 6.5),
            soil_weather_data['rainfall']
        ]])

        # Predict crop
        crop_prediction = crop_model.predict(crop_input)[0]
        crop_name = label_encoder.inverse_transform([crop_prediction])[0]

        # Get prediction probabilities for confidence and alternatives
        probabilities = crop_model.predict_proba(crop_input)[0]
        confidence = float(max(probabilities) * 100)
        
        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[::-1][:3]
        alternatives = []
        for i, idx in enumerate(top_indices[1:]):
            alt_crop = label_encoder.inverse_transform([idx])[0]
            alt_confidence = float(probabilities[idx] * 100)
            alternatives.append({
                "name": alt_crop,
                "confidence": round(alt_confidence, 1)
            })

        # Prepare comprehensive response
        result = {
            'success': True,
            'location': location_name,
            'analysis': {
                'primary_recommendation': {
                    'crop': crop_name,
                    'confidence': round(confidence, 1)
                },
                'alternative_crops': alternatives,
                'soil_analysis': {
                    'ph': round(soil_weather_data.get('ph', 6.5), 2),
                    'sand_percent': round(soil_weather_data.get('sand', 35.0), 1),
                    'silt_percent': round(soil_weather_data.get('silt', 35.0), 1),
                    'clay_percent': round(soil_weather_data.get('clay', 30.0), 1),
                    'organic_carbon': round(soil_weather_data.get('organic_carbon', 1.5), 2),
                    'cation_exchange_capacity': round(soil_weather_data.get('cec', 15.0), 1)
                },
                'climate_data': {
                    'temperature_celsius': round(soil_weather_data['temperature'], 1),
                    'annual_rainfall_mm': round(soil_weather_data['rainfall'], 0),
                    'humidity_percent': round(soil_weather_data['humidity'], 1)
                },
                'nutrient_estimates': {
                    'nitrogen_kg_ha': round(soil_weather_data.get('N', 40.0), 1),
                    'phosphorus_kg_ha': round(soil_weather_data.get('P', 30.0), 1),
                    'potassium_kg_ha': round(soil_weather_data.get('K', 25.0), 1)
                }
            }
        }
        
        # Save analysis to database
        user = get_current_user()
        if user:
            db.save_analysis(
                user['id'], location_name, lat, lng, 
                result['analysis'], crop_name, confidence
            )
            db.update_last_login(user['id'])
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in soil analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}',
            'location': location if 'location' in locals() else 'unknown'
        }), 500

@app.route('/api/analysis/<int:analysis_id>')
def get_analysis_details(analysis_id):
    """Get detailed analysis data by ID"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
            
        # Get user's analyses to ensure they own this analysis
        analyses = db.get_user_analyses(user['id'])
        analysis = None
        for a in analyses:
            if a['id'] == analysis_id:
                analysis = a
                break
                
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404
            
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Initialize models on module load
print("Initializing models...")
load_models()

if __name__ == '__main__':
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5002))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting Soil Analysis API on {host}:{port}")
    app.run(host=host, port=port, debug=debug_mode)
# hello

