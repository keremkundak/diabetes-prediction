import sys
sys.path.append('src')

import streamlit as st
import pandas as pd
import joblib
from feature_engineering import create_features

# Modeli ve scaler'ı yükle
model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title('Diyabet Tahmin Uygulaması')

st.write('Lütfen kan değerlerinizi girin:')

# Kullanıcıdan inputları al
age = st.number_input('Yaş', min_value=0, max_value=120, value=25)
gender = st.selectbox('Cinsiyet', ['Erkek', 'Kadın'])
bmi = st.number_input('Vücut Kitle İndeksi (BMI)', min_value=0.0, value=25.0)
chol = st.number_input('Kolesterol (Chol)', min_value=0.0, value=200.0)
tg = st.number_input('Trigliserit (TG)', min_value=0.0, value=150.0)
hdl = st.number_input('HDL Kolesterol', min_value=0.0, value=50.0)
ldl = st.number_input('LDL Kolesterol', min_value=0.0, value=100.0)
cr = st.number_input('Kreatinin (Cr)', min_value=0.0, value=1.0)
bun = st.number_input('Kan Üre Azotu (BUN)', min_value=0.0, value=20.0)

# Tahmin butonu
if st.button('Tahmin Yap'):
    # Inputları dataframe'e çevir
    gender_map = {'Erkek': 1, 'Kadın': 0}
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_map[gender]],
        'BMI': [bmi],
        'Chol': [chol],
        'TG': [tg],
        'HDL': [hdl],
        'LDL': [ldl],
        'Cr': [cr],
        'BUN': [bun]
    })

    # Feature engineering
    input_data = create_features(input_data)

    # Veriyi ölçeklendir
    numerical_cols = [col for col in input_data.columns if input_data[col].dtype != 'object']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Tahmin yap
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write('## Tahmin Sonucu')
    if prediction[0] == 1:
        st.write('**Diyabet riski bulunmaktadır.**')
    else:
        st.write('**Diyabet riski bulunmamaktadır.**')
    
    st.write(f'Diyabet Olma Olasılığı: {prediction_proba[0][1]:.2f}')