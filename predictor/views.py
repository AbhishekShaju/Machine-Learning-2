from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .models import train_models, load_models
import os

# Train models if they don't exist
if not os.path.exists('models'):
    os.makedirs('models')
    train_models()

# Load the trained models
models = load_models()

def home(request):
    return render(request, 'predictor/home.html')

def polynomial(request):
    return render(request, 'predictor/polynomial.html')

def logistic(request):
    return render(request, 'predictor/logistic.html')

def knn(request):
    return render(request, 'predictor/knn.html')

@csrf_exempt
def predict_polynomial(request):
    if request.method == 'POST':
        data = request.POST
        
        # Extract features
        features = np.array([
            float(data['experience']),
            float(data['education']),
            float(data['skills'])
        ]).reshape(1, -1)
        
        # Transform features
        features_transformed = models['poly_features'].transform(features)
        
        # Make prediction
        prediction = models['polynomial'].predict(features_transformed)[0]
        
        return JsonResponse({
            'predicted_salary': float(prediction)
        })
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def predict_logistic(request):
    if request.method == 'POST':
        data = request.POST
        
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
        
        return JsonResponse({
            'predicted_range': ranges[prediction]
        })
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def predict_knn(request):
    if request.method == 'POST':
        data = request.POST
        
        # Extract features
        features = np.array([
            float(data['experience']),
            float(data['education']),
            float(data['skills']),
            float(data['company_size']),
            float(data['location'])
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = models['scaler'].transform(features)
        
        # Make prediction
        prediction = models['knn'].predict(features_scaled)[0]
        
        # Map prediction to salary category
        categories = {
            0: "Entry Level",
            1: "Mid Level",
            2: "Senior Level",
            3: "Executive Level"
        }
        
        return JsonResponse({
            'predicted_category': categories[prediction]
        })
    return JsonResponse({'error': 'Invalid request method'}, status=400)
