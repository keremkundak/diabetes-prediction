import pandas as pd
import numpy as np

def create_features(df):
    """Yeni özellikler oluşturur."""
    print("Number of zeros in HDL:", (df['HDL'] == 0).sum())
    df['BMI_Age'] = df['BMI'] * df['Age']
    # Replace 0 in HDL with a small number to avoid division by zero
    df['HDL'] = df['HDL'].replace(0, 1e-6)
    df['Chol_HDL_ratio'] = df['Chol'] / df['HDL']
    return df