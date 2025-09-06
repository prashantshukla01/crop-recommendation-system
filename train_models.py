import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder
import os

def load_and_prepare_data(file_path):
    """Load and prepare crop recommendation dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nUnique crops: {df['label'].nunique()}")
    print(f"Crop distribution:\n{df['label'].value_counts()}")
    
    return df
def create_model_directory():
    """Create model directory if it doesn't exist"""
    os.makedirs('model', exist_ok=True)


def train_optimized_crop_model(df):
    """Train optimized crop prediction model with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING OPTIMIZED CROP PREDICTION MODEL")
    print("="*60)

    # Features for crop prediction
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Validate that all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")

    X = df[features]
    y = df['label']
    
    print(f"Training with {len(X)} samples and {len(features)} features")
    print(f"Target classes: {y.nunique()}")

    # Encode crop labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Hyperparameter tuning for better accuracy
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    print("\nPerforming hyperparameter tuning...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Use best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

    # Create model directory
    create_model_directory()
    
    # Save model and encoder
    joblib.dump(best_model, 'model/crop_predictor.joblib')
    joblib.dump(le, 'model/label_encoder.joblib')
    
    print("\nModel and encoder saved successfully!")
    return best_model, le


def compare_datasets():
    """Compare old vs new dataset"""
    print("Comparing datasets...")
    
    # Load old dataset
    try:
        old_df = pd.read_csv('data/crop_recommendation.csv')
        print(f"Old dataset: {len(old_df)} samples, {old_df['label'].nunique()} crops")
        print(f"Old crop distribution (top 5):")
        print(old_df['label'].value_counts().head())
    except:
        print("Old dataset not found")
    
    # Load new dataset
    try:
        new_df = pd.read_csv('data/comprehensive_crop_dataset.csv')
        print(f"\nNew dataset: {len(new_df)} samples, {new_df['label'].nunique()} crops")
        print(f"New crop distribution (top 5):")
        print(new_df['label'].value_counts().head())
        print(f"\nAll crops in new dataset:")
        print(sorted(new_df['label'].unique()))
    except:
        print("New dataset not found")

if __name__ == "__main__":
    try:
        print("Starting Enhanced Crop Recommendation Model Training")
        print("=" * 60)
        
        # Compare datasets first
        compare_datasets()
        
        # Use comprehensive dataset if available
        dataset_path = 'data/comprehensive_crop_dataset.csv'
        if not os.path.exists(dataset_path):
            print(f"\nComprehensive dataset not found at {dataset_path}")
            dataset_path = 'data/crop_recommendation.csv'
            print(f"Falling back to {dataset_path}")
        
        # Load and prepare data
        df = load_and_prepare_data(dataset_path)
        
        # Train optimized crop model
        model, encoder = train_optimized_crop_model(df)
        
        print("\n" + "=" * 60)
        print("ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nFiles saved:")
        print("- model/crop_predictor.joblib (trained model)")
        print("- model/label_encoder.joblib (label encoder)")
        print(f"\nModel trained on {len(df)} samples with {df['label'].nunique()} crop types")
        print("\nYou can now run the Flask app with: python app.py")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
