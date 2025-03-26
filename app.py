from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from models import train_models, load_models
import os

app = Flask(__name__)

# Train models if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')
    train_models()

# Load the trained models
models = load_models()

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict_polynomial', methods=['POST'])
def predict_polynomial():
    data = request.get_json()
    
    # Extract features
    features = np.array([
        float(data['experience']),
        float(data['education']),
        float(data['skills'])
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = models['polynomial'].predict(features)[0]
    
    return jsonify({
        'predicted_salary': float(prediction)
    })

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    data = request.get_json()
    
    # Extract features
    features = np.array([
        float(data['experience']),
        float(data['education']),
        float(data['skills']),
        float(data['company_size'])
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = models['logistic'].predict(features)[0]
    
    # Map prediction to salary range
    ranges = {
        0: "Low ($0-$50,000)",
        1: "Medium ($50,000-$100,000)",
        2: "High ($100,000-$150,000)",
        3: "Very High ($150,000+)"
    }
    
    return jsonify({
        'predicted_range': ranges[prediction]
    })

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    data = request.get_json()
    
    # Extract features
    features = np.array([
        float(data['experience']),
        float(data['education']),
        float(data['skills']),
        float(data['company_size']),
        float(data['location'])
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = models['knn'].predict(features)[0]
    
    # Map prediction to salary category
    categories = {
        0: "Entry Level",
        1: "Mid Level",
        2: "Senior Level",
        3: "Executive Level"
    }
    
    return jsonify({
        'predicted_category': categories[prediction]
    })

if __name__ == '__main__':
    app.run(debug=True) 