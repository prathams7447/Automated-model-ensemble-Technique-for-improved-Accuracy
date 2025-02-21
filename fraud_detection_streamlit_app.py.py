
import streamlit as st
import requests
import numpy as np

st.title('Credit Card Fraud Detection')

# User Inputs
amount = st.number_input('Transaction Amount', min_value=1.0, value=100.0)

# Collect 29 more feature inputs
features = []
for i in range(29):
    features.append(st.number_input(f'Feature {i+1}', value=0.0))

# Combine all features (Amount as the first feature)
input_data = [amount] + features

if st.button('Check for Fraud'):
    response = requests.post('http://localhost:5000/predict', json={'input': [input_data]})
    result = response.json()

    if 'error' in result:
        st.write(f"Error: {result['error']}")
    else:
        st.write("Fraudulent Transaction" if result['prediction'] == 1 else "Legitimate Transaction")
