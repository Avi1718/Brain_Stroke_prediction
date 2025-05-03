from flask import Flask, render_template, request
import numpy as np
import os
import pickle

app = Flask(__name__)

# Load the model with error handling
try:
    if not os.path.exists('stroke_model.pkl'):
        raise FileNotFoundError("Model file 'stroke_model.pkl' not found")
    
    # Try to load the model
    with open('stroke_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Validate the model
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid scikit-learn model")
    
    print("Model loaded and validated successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            if model is None:
                return render_template('index.html', prediction_text="Error: Model not loaded properly. Please check the model file.")
            
            # Get form values and convert to float
            gender = float(request.form['gender'])
            age = float(request.form['age'])
            hypertension = float(request.form['hypertension'])
            heart_disease = float(request.form['heart_disease'])
            ever_married = float(request.form['ever_married'])
            avg_glucose_level = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])
            smoking_status = float(request.form['smoking_status'])
            
            # Create feature array in the correct order
            features = np.array([gender, age, hypertension, heart_disease, ever_married, 
                       avg_glucose_level, bmi, smoking_status]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            result = 'Stroke Detected' if prediction == 1 else 'No Stroke Risk'
            return render_template('index.html', prediction_text=result)
    except Exception as e:
        error_message = f"Error: {str(e)}. Please check your input values."
        return render_template('index.html', prediction_text=error_message)

if __name__ == "__main__":
    app.run(debug=True)
