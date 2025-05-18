from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle

app = Flask(__name__)

# Constants for normalization
GLUCOSE_MAX = 300.0  # Maximum expected glucose level in mg/dL
BMI_MAX = 50.0      # Maximum expected BMI in kg/m²

# Load the model with error handling
try:
    if not os.path.exists('logistic_regression_model.pkl'):
        raise FileNotFoundError("Model file 'logistic_regression_model.pkl' not found")
    
    # Try to load the model
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Validate the model
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid scikit-learn model")
    
    print("Model loaded and validated successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def validate_input(value, min_val, max_val, field_name):
    try:
        float_val = float(value)
        if not (min_val <= float_val <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        return float_val
    except ValueError as e:
        raise ValueError(f"Invalid {field_name}: {str(e)}")

def normalize_value(value, max_value):
    """Normalize a value to 0-1 range by dividing by max value"""
    return float(value) / max_value

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            if model is None:
                return render_template('index.html', prediction_text="Error: Model not loaded properly. Please check the model file.")
            
            # Validate and convert inputs
            try:
                gender = validate_input(request.form['gender'], 0, 1, "Gender")
                age_band = validate_input(request.form['age_band'], 0, 4, "Age Band")
                hypertension = validate_input(request.form['hypertension'], 0, 1, "Hypertension")
                heart_disease = validate_input(request.form['heart_disease'], 0, 1, "Heart Disease")
                
                # Validate raw values
                glucose_level = validate_input(request.form['avg_glucose_level'], 40, 300, "Average Glucose Level")
                bmi_value = validate_input(request.form['bmi'], 10, 50, "BMI")
                
                # Normalize the values
                avg_glucose_level = normalize_value(glucose_level, GLUCOSE_MAX)
                bmi = normalize_value(bmi_value, BMI_MAX)
                
                smoking_status = validate_input(request.form['smoking_status'], 0, 3, "Smoking Status")
            except ValueError as e:
                return render_template('index.html', prediction_text=f"Input Error: {str(e)}")
            
            # Create feature array in the correct order
            features = np.array([
                gender, age_band, hypertension, heart_disease,
                avg_glucose_level, bmi, smoking_status
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]  # Get probability of stroke
            
            # Format the result with probability
            result = 'Stroke Risk Detected' if prediction == 1 else 'No Stroke Risk Detected'
            probability_text = f"Risk Probability: {probability:.1%}"
            
            return render_template('index.html', 
                                prediction_text=result,
                                probability_text=probability_text)
    except Exception as e:
        error_message = f"Error: {str(e)}. Please check your input values."
        return render_template('index.html', prediction_text=error_message)

if __name__ == "__main__":
    app.run(debug=True)
