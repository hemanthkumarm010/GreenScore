from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from predict import predict_sustainability  # Assuming predict.py contains the prediction function

app = Flask(__name__, static_folder='frontend')

# Load your model and the scaler used during training
model = joblib.load(r'C:\Users\DSATM\Desktop\sustainable-living-project\models\random_forest_model.joblib')
scaler = joblib.load(r'C:\Users\DSATM\Desktop\sustainable-living-project\models\scaler.joblib')  # Load the scaler if saved

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('frontend', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        user_input = request.get_json()
        print("Received user input:", user_input)

        # Call the predict function from predict.py
        prediction = predict_sustainability(model, user_input, scaler)

        # Return the prediction as JSON
        print("Prediction:", prediction)
        return jsonify({'prediction': prediction})

    except Exception as e:
        print("Error occurred:", e)  # Print any exceptions
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
