from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

# Create a simple logistic regression model
model = LogisticRegression(random_state=42)

# Create some dummy training data
X = np.random.rand(100, 7)  # 7 features: gender, age_band, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status
y = np.random.randint(0, 2, 100)  # Binary classification

# Train the model
model.fit(X, y)

# Save the model using pickle
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

# Verify the model was saved correctly
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        print("Model saved and verified successfully")
        print(f"Model type: {type(loaded_model)}")
        print(f"Model has predict method: {hasattr(loaded_model, 'predict')}")
        print(f"Number of features: {loaded_model.n_features_in_}")
except Exception as e:
    print(f"Error verifying model: {str(e)}") 