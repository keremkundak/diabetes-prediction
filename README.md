# Diyabet Tahmin Modeli

Bu proje, [health test by blood dataset](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset) veri setini kullanarak diyabet hastalığını tahmin etmeyi amaçlamaktadır. Proje kapsamında çeşitli makine öğrenmesi modelleri denenmiş ve en iyi sonuçları veren model seçilmiştir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

1.  **Veri setini `data/raw/` klasörüne indirin.**

2.  **Modelleri eğitmek ve deneyleri MLflow'a loglamak için:**

    ```bash
    python src/train.py
    ```

## Raporlar ve Deney Takibi (MLflow)

Deney sonuçlarını, metrikleri ve modelleri görmek için MLflow UI'ı başlatın:

```bash
mlflow ui
```

Tarayıcınızda `http://127.0.0.1:5000` adresini açın.

## Streamlit Web Arayüzü

İnteraktif tahminler yapmak için Streamlit uygulamasını çalıştırın:

```bash
streamlit run app.py
```

## FastAPI

Programatik tahminler için FastAPI uygulamasını çalıştırın:

```bash
uvicorn api:app --reload
```

API dokümantasyonuna `http://127.0.0.1:8000/docs` adresinden ulaşabilirsiniz.

Örnek bir `curl` isteği:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "Age": 55,
  "Gender": "Erkek",
  "BMI": 30.0,
  "Chol": 250.0,
  "TG": 180.0,
  "HDL": 40.0,
  "LDL": 160.0,
  "Cr": 1.2,
  "BUN": 25.0
}' http://127.0.0.1:8000/predict
```
