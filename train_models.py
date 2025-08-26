import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import joblib
import json
from sklearn.preprocessing import LabelEncoder


# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    return df


# Train NPK prediction models (Stage 1)
def train_npk_models(df):
    print("\n" + "=" * 50)
    print("TRAINING NPK PREDICTION MODELS")
    print("=" * 50)

    # Features available from Open-Meteo APIs
    api_features = ['temperature', 'humidity', 'ph', 'rainfall']

    # Since your dataset doesn't have soil composition, we'll simulate it
    # In real usage, you'd get this from Open-Meteo Soil API
    np.random.seed(42)
    df['sand'] = np.random.uniform(5, 95, len(df))
    df['silt'] = (100 - df['sand']) * np.random.uniform(0.2, 0.8, len(df))
    df['clay'] = (100 - df['sand']) - df['silt']

    api_features += ['sand', 'silt', 'clay']

    npk_models = {}
    targets = ['N', 'P', 'K']

    for nutrient in targets:
        print(f"\nTraining {nutrient} predictor...")

        X = df[api_features]
        y = df[nutrient]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.2f}, R²: {r2:.4f}")

        npk_models[nutrient] = model
        joblib.dump(model, f'model/{nutrient}_predictor.joblib')

    # Save feature list for later use
    with open('model/api_features.json', 'w') as f:
        json.dump(api_features, f)

    return npk_models, api_features


# Train crop prediction model (Stage 2)
def train_crop_model(df):
    print("\n" + "=" * 50)
    print("TRAINING CROP PREDICTION MODEL")
    print("=" * 50)

    # Use all features including NPK
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    X = df[features]
    y = df['label']

    # Encode crop labels if needed
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model and label encoder
    joblib.dump(model, 'model/crop_predictor.joblib')
    joblib.dump(le, 'model/label_encoder.joblib')

    return model, le


if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data('data/crop_recommendation.csv')

    # Train both models
    npk_models, api_features = train_npk_models(df)
    crop_model, label_encoder = train_crop_model(df)

    print("\n" + "=" * 50)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 50)