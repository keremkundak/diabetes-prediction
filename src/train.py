import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import load_data, split_data
from data_preprocessing import preprocess_data

# Veriyi yükle ve ön işle
df = load_data('data/raw/Diabetes Classification.csv')
df_processed, scaler = preprocess_data(df)

# Scaler'ı kaydet
joblib.dump(scaler, 'models/scaler.pkl')

# Veriyi ayır
X_train, X_test, y_train, y_test = split_data(df_processed, 'Diagnosis')

# Modeller
models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# MLflow deneyi başlat
mlflow.set_experiment("Diabetes Prediction")

# Modelleri eğit, değerlendir ve logla
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f'{name} modeli eğitiliyor...')
        model.fit(X_train, y_train)

        # Değerlendir
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        }

        # Parametreleri, metrikleri ve modeli logla
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)

# Neural Network modelini eğit, değerlendir ve logla
with mlflow.start_run(run_name='neural_network'):
    print('neural_network modeli eğitiliyor...')
    nn_model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Değerlendir
    y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32")
    y_proba_nn = nn_model.predict(X_test)
    nn_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_nn),
        'Precision': precision_score(y_test, y_pred_nn),
        'Recall': recall_score(y_test, y_pred_nn),
        'F1-Score': f1_score(y_test, y_pred_nn),
        'ROC AUC': roc_auc_score(y_test, y_proba_nn)
    }

    # Metrikleri ve modeli logla
    mlflow.log_metrics(nn_metrics)
    mlflow.keras.log_model(nn_model, "neural_network")

print("Modeller başarıyla eğitildi ve MLflow'a loglandı.")
