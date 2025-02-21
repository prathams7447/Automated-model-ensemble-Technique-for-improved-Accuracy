import pickle
import requests
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd



app = Flask(__name__)

# Load the trained model
with open("fraud_detection_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load PCA transformer 
with open("pca_transformer.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

# Load StandardScaler 
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

print(f"StandardScaler expects: {scaler.n_features_in_} features")
print(f"PCA expects: {pca.n_features_in_} features")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)

        print(f"Received input shape: {input_data.shape}") 

        # Extract 'Amount' from input (Assuming it's the first feature)
        amount = input_data[0, 0]

        log_amount = np.log1p(amount)

        # Remove 'Amount' from input before scaling
        input_without_amount = input_data[:, 1:]  # Exclude first column (Amount)

        if input_without_amount.shape[1] != scaler.n_features_in_:
            return jsonify({'error': f'Expected {scaler.n_features_in_} features for StandardScaler, but got {input_without_amount.shape[1]}'})

        input_scaled = scaler.transform(input_without_amount)

        input_scaled_with_log = np.append(input_scaled, [[log_amount]], axis=1)

        print(f"Updated input shape after adding log_amount: {input_scaled_with_log.shape}")

        if input_scaled_with_log.shape[1] != pca.n_features_in_:
            return jsonify({'error': f'Expected {pca.n_features_in_} features for PCA, but got {input_scaled_with_log.shape[1]}'})

        input_transformed = pca.transform(input_scaled_with_log)

        prediction = model.predict(input_transformed)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

