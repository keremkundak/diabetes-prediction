import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """Veriyi yükler."""
    return pd.read_csv(path)

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Veriyi train ve test olarak ayırır."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
