import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle



model = tf.keras.models.load_model('modelreg.h5')
with open('preprocessorreg.pkl','rb') as file:
    preprocessor= pickle.load(file)

st.title('customer churn predict app')

## user input 
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=92, value=40)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited= st.number_input("Exited")
tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data=pd.DataFrame(
    {
    'CreditScore':[credit_score], 
    'Geography': [geography] ,
    'Gender':[gender],
    'Age':[age], 
    'Tenure':[tenure], 
    'Balance':[balance],
    'NumOfProducts':[num_of_products], 
    'HasCrCard':[has_cr_card], 
    'IsActiveMember':[is_active_member],
    'Exited':[exited],
}   
)

if st.button("Predict"):
    try:
        # Apply preprocessing
        transformed_data = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)
        st.info(f"🟢 The estimated salary is  (salary: {prediction:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
