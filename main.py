# app.py — Cleaned and test-friendly version

from logging_config import get_logger
logger = get_logger(__name__)

from typing import List
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

import joblib
import uvicorn

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------------------
# Prometheus metrics
# ---------------------
REQUEST_COUNT = Counter("request_count", "Total number of requests", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of requests", ["endpoint"])

# ---------------------
# MLflow setup
# ---------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("DecisionTree_Experiment")

logger.info("Loading dataset")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------
# Train and log model
# ---------------------
with mlflow.start_run() as run:
    model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))

    input_example = [X_test[0].tolist()]
    signature = mlflow.models.infer_signature(X_test, preds)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="decision_tree_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="DecisionTreeClassifier",
    )

    joblib.dump(model, "model.pkl")
    logger.info("Model saved as model.pkl")

# ---------------------
# Manage MLflow model aliases
# ---------------------
client = MlflowClient()
versions = client.search_model_versions("name='DecisionTreeClassifier'")
latest_version = max(versions, key=lambda v: int(v.version)).version if versions else None

if latest_version:
    try:
        client.set_model_version_alias("DecisionTreeClassifier", latest_version, "production")
        logger.info(f"Set alias 'production' -> version {latest_version}")
    except Exception as e:
        logger.warning(f"Could not set alias: {e}")

# ---------------------
# Load production model
# ---------------------
mlflow_model = None
if latest_version:
    try:
        import mlflow.pyfunc
        mlflow_model = mlflow.pyfunc.load_model("models:/DecisionTreeClassifier/production")
        logger.info("Loaded model from registry (production stage).")
    except Exception as e:
        logger.warning(f"Failed to load registry model: {e}. Using local model.")
        mlflow_model = joblib.load("model.pkl")
else:
    mlflow_model = joblib.load("model.pkl")
    logger.info("Loaded local model (no registry version found).")

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="Decision Tree Classifier API")

class PredictRequest(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(request: PredictRequest):
    """Make a prediction for a single input sample"""
    if mlflow_model is None:
        return {"error": "Model not loaded"}

    try:
        preds = mlflow_model.predict([request.features])
        prediction = int(preds[0])
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": "Prediction failed", "details": str(e)}

# ---------------------
# Middleware for Prometheus
# ---------------------
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    REQUEST_COUNT.labels(endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    return response

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# ---------------------
# Run app
# ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
