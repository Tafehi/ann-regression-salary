import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load model and encoders
model = load_model('regression_model.h5')

with open('standard_scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

with open('label_gender_encoder.pkl', 'rb') as file:
    label_gender_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Streamlit App Title
st.title('Salary Prediction App')

# Input Data
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_gender_encoder.classes_)
age = st.slider('Age', 18, 60, 30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited or not', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode the geography field
geo_encoded = onehot_encoder_geo.transform(np.array([[geography]])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1).drop(columns=['Geography'])

# Standardize the input data
input_data = standard_scaler.transform(input_data)

# Predict the salary
predicted_salary = model.predict(input_data)

# Display the result
st.write(f'The predicted salary is {predicted_salary[0][0]:.2f}')
