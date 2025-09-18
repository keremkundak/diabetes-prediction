import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_engineering import create_features

def preprocess_data(df):
    """Veri ön işleme adımlarını uygular."""
    # Unnamed: 0 kolonunu sil
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Gender kolonunu sayısala çevir
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.lower()
        df['Gender'] = df['Gender'].map({'m': 1, 'f': 0})

    # Eksik değerleri ortalama ile doldurma
    for col in df.columns:
        if df[col].dtype != 'object' and col != 'Diagnosis':
            df[col] = df[col].fillna(df[col].mean())

    # Feature Engineering
    df = create_features(df)

    # Sayısal değişkenleri ölçeklendirme
    scaler = StandardScaler()
    numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'Diagnosis']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler
