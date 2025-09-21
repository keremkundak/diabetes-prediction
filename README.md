# Diabetes Prediction Model

This project aims to predict diabetes using the Pima Indians Diabetes dataset. Various machine learning models were tested within the scope of the project, and the model that gave the best results was selected.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1.  **Download the dataset to the `data/raw/` folder.**

2.  **To train the models and log experiments to MLflow:**

    ```bash
    python src/train.py
    ```

## Reports and Experiment Tracking (MLflow)

To see the experiment results, metrics, and models, start the MLflow UI:

```bash
mlflow ui
```

Open `http://127.0.0.1:5000` in your browser.

## Streamlit Web Interface

To make interactive predictions, run the Streamlit application:

```bash
streamlit run app.py
```

## FastAPI

To make programmatic predictions, run the FastAPI application:

```bash
uvicorn api:app --reload
```

You can access the API documentation at `http://1227.0.0.1:8000/docs`.

An example `curl` request:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "Age": 55,
  "Gender": "Male",
  "BMI": 30.0,
  "Chol": 250.0,
  "TG": 180.0,
  "HDL": 40.0,
  "LDL": 160.0,
  "Cr": 1.2,
  "BUN": 25.0
}' http://127.0.0.1:8000/predict
```