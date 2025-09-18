import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_data, split_data
from data_preprocessing import preprocess_data

# Veriyi yükle ve ön işle
df = load_data('data/raw/Diabetes Classification.csv')
df_processed = preprocess_data(df)

# Veriyi ayır
X_train, X_test, y_train, y_test = split_data(df_processed, 'Diagnosis')

# Modelleri yükle
models = {}
for name in ['logistic_regression', 'random_forest', 'xgboost']:
    models[name] = joblib.load(f'models/{name}.pkl')

# Metrikleri hesapla
metrics = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    }

# Neural network modelini yükle ve değerlendir
nn_model = load_model('models/nn_model.h5')
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32")
y_proba_nn = nn_model.predict(X_test)
metrics['neural_network'] = {
    'Accuracy': accuracy_score(y_test, y_pred_nn),
    'Precision': precision_score(y_test, y_pred_nn),
    'Recall': recall_score(y_test, y_pred_nn),
    'F1-Score': f1_score(y_test, y_pred_nn),
    'ROC AUC': roc_auc_score(y_test, y_proba_nn)
}


# Metrikleri CSV'ye kaydet
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv('reports/metrics_table.csv')

# Model karşılaştırma grafiği
metrics_df['F1-Score'].plot(kind='bar', figsize=(10, 6))
plt.title('Model F1-Skor Karşılaştırması')
plt.ylabel('F1-Skor')
plt.xticks(rotation=45)
plt.savefig('reports/model_comparison.png')

print('Değerlendirme tamamlandı. Raporlar oluşturuldu.')
