from logging_config import get_logger

logger=get_logger(__name__)

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import numpy as np
import pandas as pd
import os

import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic    import BaseModel,RootModel
import joblib
import uvicorn

from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Request
import time

REQUEST_COUNT = Counter("request_count", "Total number of requests", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of requests", ["endpoint"])

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("DecisionTree_Experiment")

logger.info('loading dataset')

X,y=load_iris(return_X_y=True)

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

    input_example = [X_test[0].tolist()]
    signature = mlflow.models.infer_signature(X_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model,name='decision_tree_model',signature=signature,input_example=input_example)

    joblib.dump(model,'model.pkl')
    logger.info("Model saved as model.pkl")

app=FastAPI(title="decision tree classifier api")

class PredictRequest(RootModel[list[float]]):
    """Single sample for prediction"""

@app.post("/predict")
def predict(request: PredictRequest):
    model = joblib.load("model.pkl")
    prediction = model.predict([request.root])
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
