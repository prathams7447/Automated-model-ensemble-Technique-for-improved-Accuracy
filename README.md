# Automated-model-ensemble-Technique-for-improved-Accuracy# Credit Card Fraud Detection

This project aims to develop a machine learning model to detect fraudulent credit card transactions using various models, including RandomForest, XGBoost, and GradientBoosting. The model uses a stacked ensemble approach to improve prediction accuracy.

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

2. **Install Required Dependencies: Install the required Python packages using**:
pip install -r requirements.txt

3. **Train the Model: Run the following script to train the fraud detection model**:
python train_fraud_detection_model.py

4. **Start the Flask API: Run the Flask API to serve the trained model for predictions**:
python fraud_detection_api.py

5. **Start the Streamlit App**:
streamlit run fraud_detection_streamlit_app.py


# Credit Card Fraud Detection Dataset
The dataset used in this project is a credit card fraud detection dataset. It is preprocessed by handling missing values, outlier removal, feature scaling, and dimensionality reduction using PCA. The dataset is available in the repository.


# License
This project is licensed under the MIT License - see the LICENSE file for details.