from django.db import models
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create your models here.

def generate_sample_data(n_samples=1000):
    """Generate sample data for training"""
    np.random.seed(42)
    
    # Generate features
    experience = np.random.uniform(0, 20, n_samples)
    education = np.random.randint(1, 5, n_samples)
    skills = np.random.randint(1, 10, n_samples)
    company_size = np.random.randint(1, 5, n_samples)
    location = np.random.randint(1, 5, n_samples)
    
    # Generate target variables
    # For polynomial regression (continuous salary)
    salary = 30000 + 2000 * experience + 5000 * education + 1000 * skills + np.random.normal(0, 5000, n_samples)
    
    # For logistic regression (salary ranges)
    salary_ranges = np.zeros(n_samples)
    salary_ranges[salary < 50000] = 0
    salary_ranges[(salary >= 50000) & (salary < 100000)] = 1
    salary_ranges[(salary >= 100000) & (salary < 150000)] = 2
    salary_ranges[salary >= 150000] = 3
    
    # For KNN (salary categories)
    salary_categories = np.zeros(n_samples)
    salary_categories[salary < 40000] = 0
    salary_categories[(salary >= 40000) & (salary < 80000)] = 1
    salary_categories[(salary >= 80000) & (salary < 120000)] = 2
    salary_categories[salary >= 120000] = 3
    
    # Create DataFrames
    X = pd.DataFrame({
        'experience': experience,
        'education': education,
        'skills': skills,
        'company_size': company_size,
        'location': location
    })
    
    y_polynomial = salary
    y_logistic = salary_ranges
    y_knn = salary_categories
    
    return X, y_polynomial, y_logistic, y_knn

def train_models():
    """Train all models and save them"""
    # Generate sample data
    X, y_polynomial, y_logistic, y_knn = generate_sample_data()
    
    # Split data for each model
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
        X[['experience', 'education', 'skills']], y_polynomial, test_size=0.2, random_state=42
    )
    
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X[['experience', 'education', 'skills', 'company_size']], y_logistic, test_size=0.2, random_state=42
    )
    
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X, y_knn, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_knn_scaled = scaler.fit_transform(X_train_knn)
    X_test_knn_scaled = scaler.transform(X_test_knn)
    
    # Train Polynomial Regression
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly_transformed = poly_features.fit_transform(X_train_poly)
    polynomial_model = LinearRegression()
    polynomial_model.fit(X_train_poly_transformed, y_train_poly)
    
    # Train Logistic Regression
    logistic_model = LogisticRegression(multi_class='multinomial')
    logistic_model.fit(X_train_log, y_train_log)
    
    # Train KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_knn_scaled, y_train_knn)
    
    # Save models and scaler
    joblib.dump(polynomial_model, 'models/polynomial_model.joblib')
    joblib.dump(poly_features, 'models/poly_features.joblib')
    joblib.dump(logistic_model, 'models/logistic_model.joblib')
    joblib.dump(knn_model, 'models/knn_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Print model performance metrics
    print("\nModel Performance Metrics:")
    print("-" * 50)
    
    # Polynomial Regression R² score
    poly_r2 = polynomial_model.score(poly_features.transform(X_test_poly), y_test_poly)
    print(f"Polynomial Regression R² Score: {poly_r2:.4f}")
    
    # Logistic Regression accuracy
    log_accuracy = logistic_model.score(X_test_log, y_test_log)
    print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")
    
    # KNN accuracy
    knn_accuracy = knn_model.score(X_test_knn_scaled, y_test_knn)
    print(f"KNN Accuracy: {knn_accuracy:.4f}")

def load_models():
    """Load all trained models"""
    models = {}
    
    # Load Polynomial Regression model and features
    models['polynomial'] = joblib.load('models/polynomial_model.joblib')
    models['poly_features'] = joblib.load('models/poly_features.joblib')
    
    # Load Logistic Regression model
    models['logistic'] = joblib.load('models/logistic_model.joblib')
    
    # Load KNN model and scaler
    models['knn'] = joblib.load('models/knn_model.joblib')
    models['scaler'] = joblib.load('models/scaler.joblib')
    
    return models
