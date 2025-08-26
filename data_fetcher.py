import requests
import pandas as pd
from datetime import datetime, timedelta


class WeatherDataFetcher:
    def __init__(self):
        # Historical weather ke liye archive API (sahi endpoint)
        self.weather_base_url = "https://archive-api.open-meteo.com/v1/archive"

        # Soil ke liye Open-Meteo se direct nahi aata (placeholder rakha h)
        self.soil_base_url = "https://api.open-meteo.com/v1/forecast"

    def geocode_location(self, location_name):
        """Convert location name to coordinates"""
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": location_name, "count": 1, "language": "en"}
        try:
            response = requests.get(geocode_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return result['latitude'], result['longitude'], result['name']
            else:
                raise Exception('Location not found. Please try with different location name.')
        except requests.exceptions.RequestException as e:
            raise Exception(f"Geocoding failed: {str(e)}")

    def fetch_historical_weather(self, lat, lng, years=5):
        """Fetch historical weather data"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * years)

        params = {
            "latitude": lat,
            "longitude": lng,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": [
                "temperature_2m_mean",
                "precipitation_sum",
                "relative_humidity_2m_mean"
            ],
            "timezone": "auto"
        }

        try:
            response = requests.get(self.weather_base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Weather API error: {str(e)}")

    def fetch_soil_data(self, lat, lng):
        """Fetch soil data (currently Open-Meteo se sand/silt/clay nahi milte -> placeholder)"""
        try:
            # Placeholder response (API call remove kar diya kyunki Open-Meteo ye data nahi deta)
            return {
                "sand": 30.0,
                "silt": 40.0,
                "clay": 30.0
            }
        except Exception as e:
            raise Exception(f"Soil API error: {str(e)}")

    def process_weather_data(self, weather_json):
        """Process weather data into usable features"""
        daily_data = weather_json['daily']

        df = pd.DataFrame({
            'date': pd.to_datetime(daily_data['time']),
            'temperature': daily_data['temperature_2m_mean'],
            'precipitation': daily_data['precipitation_sum'],
            'humidity': daily_data['relative_humidity_2m_mean']
        })

        avg_temperature = df['temperature'].mean()
        annual_precipitation = df['precipitation'].sum() / (len(df) / 365)
        avg_humidity = df["humidity"].mean()

        return {
            "temperature": avg_temperature,
            "rainfall": annual_precipitation,
            "humidity": avg_humidity,
            "ph": 6.5  # default placeholder
        }

    def process_soil_data(self, soil_json):
        """Process soil data into usable features (sand, silt, clay required for model)"""
        soil_data = {}
        soil_data["sand"] = soil_json.get("sand", 30.0)
        soil_data["silt"] = soil_json.get("silt", 40.0)
        soil_data["clay"] = soil_json.get("clay", 30.0)
        return soil_data

    def get_location_data(self, location_input):
        """Main method to get all data for a location"""
        # Geocode first
        lat, lng, location_name = self.geocode_location(location_input)
        print(f"Fetching data for: {location_name} ({lat:.4f}, {lng:.4f})")

        # Fetch data from APIs
        weather_data = self.fetch_historical_weather(lat, lng)
        soil_data = self.fetch_soil_data(lat, lng)

        # Process data
        weather_features = self.process_weather_data(weather_data)
        soil_features = self.process_soil_data(soil_data)

        # Combine all features
        all_features = {**weather_features, **soil_features}

        return all_features, location_name
