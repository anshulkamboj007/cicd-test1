# app.py  -- updated, non-deprecated usage
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

# Optional: keep a local copy for quick load/fallback if you want
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
MLFLOW_DB_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_DB_URI)
mlflow.set_experiment("DecisionTree_Experiment")

logger.info("Loading dataset")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train + log model to MLflow
with mlflow.start_run() as run:
    model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))

    # signature and example
    input_example = [X_test[0].tolist()]
    signature = mlflow.models.infer_signature(X_test, preds)

    mlflow.log_metric("accuracy", acc)

    # This will both log and register (if registered_model_name provided)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="decision_tree_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="DecisionTreeClassifier",  # optional: registers the model
    )

    # Save a local copy (optional fallback)
    joblib.dump(model, "model.pkl")
    logger.info("Model saved locally as model.pkl")
    logger.info(f"MLflow run_id: {run.info.run_id}")

# ---------------------
# Mlflow client: find latest version and set aliases (if desired)
# ---------------------
client = MlflowClient()

# Get all versions for this registered model, pick the highest version number
# (search_model_versions returns dict-like objects)
all_versions = client.search_model_versions(f"name = 'DecisionTreeClassifier'")
if not all_versions:
    logger.warning("No registered versions found for 'DecisionTreeClassifier'")
    latest_version = None
else:
    # Convert versions to ints to pick the max
    latest_version_obj = max(all_versions, key=lambda v: int(v.version))
    latest_version = latest_version_obj.version
    logger.info(f"Latest registered model version found: {latest_version}")

    # Optionally set aliases (staging / production) for the latest version
    # Note: set_model_version_alias is still valid API
    try:
        client.set_model_version_alias(
            name="DecisionTreeClassifier",
            version=latest_version,
            alias="staging",
        )
        logger.info(f"Set alias 'staging' -> version {latest_version}")
        client.set_model_version_alias(
            name="DecisionTreeClassifier",
            version=latest_version,
            alias="production",
        )
        logger.info(f"Set alias 'production' -> version {latest_version}")
    except Exception as e:
        logger.warning(f"Could not set aliases: {e}")

# ---------------------
# Load production model once at startup
# ---------------------
# Prefer the Models Registry URI. Use 'production' stage.
# The canonical model URI is "models:/<name>/<stage>"
mlflow_model = None
if latest_version is not None:
    model_uri = "models:/DecisionTreeClassifier/production"
    try:
        # mlflow.pyfunc.load_model returns a pyfunc wrapper with predict(df) -> np.array
        import mlflow.pyfunc
        mlflow_model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model from '{model_uri}'")
    except Exception as e:
        logger.warning(f"Failed to load model from registry URI '{model_uri}': {e}")
        # fallback: try loading local joblib
        try:
            mlflow_model = joblib.load("model.pkl")
            logger.info("Loaded local joblib fallback model ('model.pkl').")
        except Exception as err:
            logger.error(f"Failed to load local fallback model: {err}")
            mlflow_model = None
else:
    # if nothing registered, try local fallback
    try:
        mlflow_model = joblib.load("model.pkl")
        logger.info("Loaded local fallback model ('model.pkl').")
    except Exception as err:
        logger.error(f"Failed to load any model: {err}")
        mlflow_model = None

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="Decision Tree Classifier API")

class PredictRequest(BaseModel):
    """Single sample for prediction"""
    features: List[float]

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict endpoint expects a single sample: { "features": [f1, f2, f3, f4] }
    """
    if mlflow_model is None:
        return {"error": "Model not loaded"}

    # If the loaded model is a pyfunc model, it prefers a 2D array or pandas.DataFrame
    try:
        # Try pyfunc predict first (it accepts list-of-lists or DataFrame)
        preds = mlflow_model.predict([request.features])
        # preds may be numpy array; convert to int for JSON serialization
        pred = int(preds[0])
    except Exception:
        # fallback: if it's a scikit-learn estimator loaded via joblib
        try:
            pred = int(mlflow_model.predict([request.features])[0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": "prediction failed", "details": str(e)}

    return {"prediction": pred}

# Middleware to record Prometheus metrics
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
    # Return Prometheus-compatible response
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
