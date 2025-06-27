import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

class TestDataValidation:
    def test_data_integrity(self):
        """Validate IRIS dataset structure and completeness"""
        data = pd.read_csv('data/iris.csv')
        
        # Check required columns exist
        required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        assert all(col in data.columns for col in required_cols)
        
        # Validate data types
        assert data['species'].dtype == 'object'
        assert all(data[col].dtype in ['float64', 'int64'] for col in required_cols[:-1])
        
        # Check for missing values
        assert not data.isnull().any().any()

    def test_model_performance(self):
        """Validate model accuracy meets minimum threshold"""
        # Load trained model
        model = joblib.load("artifacts/model.joblib")
        
        # Load test data (assuming split is reproducible)
        data = pd.read_csv('data/iris.csv')
        from sklearn.model_selection import train_test_split
        
        train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
        X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
        y_test = test.species
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Assert minimum accuracy threshold
        assert accuracy >= 0.85, f"Model accuracy {accuracy:.3f} below threshold"

    def test_feature_ranges(self):
        """Validate feature values are within expected ranges"""
        data = pd.read_csv('data/iris.csv')
        
        # Check realistic value ranges for iris dataset
        assert data['sepal_length'].between(4.0, 8.0).all()
        assert data['sepal_width'].between(2.0, 5.0).all()
        assert data['petal_length'].between(1.0, 7.0).all()
        assert data['petal_width'].between(0.1, 3.0).all()
