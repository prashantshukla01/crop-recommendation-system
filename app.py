from flask import Flask, render_template, jsonify, request
import joblib
import json
import numpy as np
from data_fetcher import WeatherDataFetcher

app= Flask(__name__)

#load models and metadata
def load_models():
    #load npk predictors
    npk_models={}
    for nutrient in ['N','P','K']:
        npk_models[nutrient]= joblib.load(f'model/{nutrient}_predictor.joblib')

    #load crop predictor
    crop_model = joblib.load('model/crop_predictor.joblib')
    label_encoder = joblib.load('model/label_encoder.joblib')

    #load features list
    with open('model/api_features.json','r') as f:
        api_features = json.load(f)

    return npk_models,crop_model,label_encoder,api_features

#initialise
npk_models, crop_model, label_encoder, API_FEATURES = load_models()
data_fetcher = WeatherDataFetcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    try:
        location = request.form.get("location")

        if not location:
            return jsonify({'error': 'please enter a location'})

        #fetch data from open meteo apis
        api_features,location_name = data_fetcher.get_location_data(location)

        #prepare input for npk prediction
        npk_input = np.array([[api_features.get(feat,0) for feat in API_FEATURES]])

        # Predict NPK values (Stage 1)
        predicted_n = npk_models['N'].predict(npk_input)[0]
        predicted_p = npk_models['P'].predict(npk_input)[0]
        predicted_k = npk_models['K'].predict(npk_input)[0]

        #prepare input for crop prediction
        crop_input = np.array([[
            predicted_n,predicted_p,predicted_k,
            api_features['temperature'],
            api_features['humidity'],
            api_features.get('ph',6.5),  # default if ph is not available
            api_features['rainfall']
        ]])

        #predict crop
        crop_prediction = crop_model.predict(crop_input)[0]
        crop_name = label_encoder.inverse_transform([crop_prediction])[0]

        #get prediction probabilities

        probabilities = crop_model.predict_proba(crop_input)[0]
        confidence = max(probabilities)*100

        #prepare response
        result = {
            'location': location_name,
            'recommended_crop': crop_name,
            'confidence': f"{confidence:.1f}%",
            'weather_data': {
                'temperature': f"{api_features['temperature']:.1f}°C",
                'rainfall': f"{api_features['rainfall']:.0f} mm/year",
                'humidity': f"{api_features['humidity']:.1f}%"
            },
            'soil_data': {k: f"{v:.2f}" for k, v in api_features.items()
                          if k in ['ph', 'sand', 'silt', 'clay', 'organic_carbon']},
            'predicted_npk': {
                'N': f"{predicted_n:.1f}",
                'P': f"{predicted_p:.1f}",
                'K': f"{predicted_k:.1f}"
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error':str(e)})

if __name__== '__main__':
    app.run(debug=True)










