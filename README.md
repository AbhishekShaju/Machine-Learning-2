# Salary Prediction System

This web application predicts salary using three different machine learning techniques:
1. Polynomial Regression - Predicts exact salary amount
2. Logistic Regression - Predicts salary range
3. K-Nearest Neighbors (KNN) - Predicts salary category

## Features

- Modern and responsive UI
- Three different prediction models
- Real-time predictions
- Easy-to-use forms
- Beautiful styling with CSS

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. On the home page, select one of the three prediction techniques
2. Fill in the required information in the form
3. Click the predict button to get your salary prediction
4. View the results displayed on the page

## Technical Details

- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Machine Learning: scikit-learn
- Models:
  - Polynomial Regression (degree=3)
  - Logistic Regression (multinomial)
  - K-Nearest Neighbors (k=5)

## Note

This application uses sample data for training the models. In a production environment, you would want to:
1. Use real salary data
2. Implement proper data validation
3. Add user authentication
4. Add error handling
5. Implement proper security measures 