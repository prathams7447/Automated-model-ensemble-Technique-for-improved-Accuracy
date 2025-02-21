## Automated-model-ensemble-Technique-for-improved-Accuracy### Credit Card Fraud Detection

This project aims to develop a machine learning model to detect fraudulent credit card transactions using various models, including RandomForest, XGBoost, and GradientBoosting. The model uses a stacked ensemble approach to improve prediction accuracy.

## Credit Card Fraud Detection Dataset

The dataset used in this project is a credit card fraud detection dataset. It is preprocessed by handling missing values, outlier removal, feature scaling, and dimensionality reduction using PCA. The dataset is available in the repository.

### Files in the Repository:

- `train_fraud_detection_model.py`: Script to train the fraud detection model.
- `fraud_detection_api.py`: Flask API for serving predictions from the trained model.
- `fraud_detection_streamlit_app.py`: Streamlit app for user interaction with fraud detection API.
- `creditcard.csv`: Dataset used for model training.
- `scaler.pkl`, `pca_transformer.pkl`, `fraud_detection_model.pkl`: Saved models and transformations.

### Steps to Execute:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/prathams7447/Automated-model-ensemble-Technique-for-improved-Accuracy.git
   cd Automated-model-ensemble-Technique-for-improved-Accuracy

   ```

2. **Extract the Dataset**:
   Download and extract the ZIP file containing the dataset. Once extracted, you will find the creditcard.csv file in the directory.

3. **Install Required Dependencies: Install the required Python packages using**:
   pip install -r requirements.txt

4. **Train the Model: Run the following script to train the fraud detection model**:
   python train_fraud_detection_model.py

5. **Start the Flask API: Run the Flask API to serve the trained model for predictions**:
   python fraud_detection_api.py
   The API will run on http://localhost:5000.

6. **Start the Streamlit App**:
   streamlit run fraud_detection_streamlit_app.py
   Streamlit App will run on http://localhost:8501

## License

This project is licensed under the MIT License - see the LICENSE file for details.
