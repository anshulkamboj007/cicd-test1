from logging_config import get_logger

logger = get_logger(__name__)

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import uvicorn

from prometheus_client import Counter, Histogram, generate_latest
import time

# 🔹 Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total number of requests", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of requests", ["endpoint"])

# 🔹 MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("DecisionTree_Experiment")

logger.info("Loading dataset")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))

    # 🔹 Updated MLflow signature API
    input_example = [X_test[0].tolist()]
    signature = mlflow.models.infer_signature(X_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="decision_tree_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="DecisionTreeClassifier"
    )

    joblib.dump(model, "model.pkl")
    logger.info("Model saved as model.pkl")

# 🔹 Alias-based registry management
client = MlflowClient()

# Get the latest version by alias "latest"
latest = client.get_model_version_by_alias("DecisionTreeClassifier", "latest")

# Assign aliases instead of stages
client.set_model_version_alias(
    name="DecisionTreeClassifier",
    version=latest.version,
    alias="staging"
)
logger.info(f"Registered DecisionTreeClassifier version {latest.version} as '@staging'")

client.set_model_version_alias(
    name="DecisionTreeClassifier",
    version=latest.version,
    alias="production"
)
logger.info(f"Registered DecisionTreeClassifier version {latest.version} as '@production'")

# 🔹 Load production model via alias
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/DecisionTreeClassifier@production")

# 🔹 FastAPI app
app = FastAPI(title="Decision Tree Classifier API")

class PredictRequest(BaseModel):
    """Single sample for prediction"""
    features: list[float]

@app.post("/predict")
def predict(request: PredictRequest):
    model = joblib.load("model.pkl")
    prediction = model.predict([request.features])
    return {"prediction": int(prediction[0])}

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
    return generate_latest()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
