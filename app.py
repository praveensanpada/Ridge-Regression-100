from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Load the trained Ridge model if it exists
try:
    with open('ridge_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global model
    # Get the file from the request
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Prepare the data
    X = df.iloc[:, 0:2].values  # CGPA and IQ as features
    y = df.iloc[:, -1].values  # LPA as the target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Train the model using Ridge Regression
    model = Ridge(alpha=1.0)  # You can adjust the alpha parameter as needed
    model.fit(X_train, y_train)
    
    # Save the model
    with open('ridge_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Test the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return jsonify({'message': 'Model trained successfully!', 'mse': mse, 'r2_score': r2})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        return jsonify({'error': 'Model not trained yet. Please upload a dataset and train the model first.'})

    # Get the input values from the request
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])
    
    # Predict the LPA based on the input CGPA and IQ
    lpa = model.predict(np.array([[cgpa, iq]]))[0]
    
    return jsonify({'cgpa': cgpa, 'iq': iq, 'predicted_lpa': lpa})

if __name__ == '__main__':
    app.run(debug=True)
