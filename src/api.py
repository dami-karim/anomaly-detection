import joblib
import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from preprocess import FEATURES

BASE_DIR   = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

app = FastAPI(
    title="Cloud Anomaly Detection API",
    description=(
        "Random Forest (supervisé) + "
        "Isolation Forest (non supervisé)"
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Charger les deux modèles
rf     = joblib.load(os.path.join(
    MODELS_DIR, 'random_forest.pkl'
))
iso    = joblib.load(os.path.join(
    MODELS_DIR, 'isolation_forest.pkl'
))
scaler = joblib.load(os.path.join(
    MODELS_DIR, 'scaler.pkl'
))
print("RF + IF chargés avec succès.")

# Statistiques globales de session
stats = {
    "total_flows":   0,
    "total_attacks": 0,
    "rf_attacks":    0,
    "iso_attacks":   0,
    "both_attacks":  0,
    "only_rf":       0,
    "only_iso":      0
}


class FlowRecord(BaseModel):
    features: List[float]


class BatchRequest(BaseModel):
    flows: List[FlowRecord]


def predict_one(features: list) -> dict:
    X  = np.array(features).reshape(1, -1)
    Xs = scaler.transform(X)

    # Random Forest
    rf_pred  = int(rf.predict(Xs)[0])
    rf_proba = float(rf.predict_proba(Xs)[0][1])

    # Isolation Forest
    iso_pred  = int(iso.predict(Xs)[0] == -1)
    iso_score = float(-iso.score_samples(Xs)[0])

    # Décision combinée
    is_attack = bool(rf_pred == 1 or iso_pred == 1)

    if rf_pred == 1 and iso_pred == 1:
        detector = "RF + IF"
    elif rf_pred == 1:
        detector = "RF"
    elif iso_pred == 1:
        detector = "IF"
    else:
        detector = "aucun"

    # Mise à jour stats
    stats["total_flows"] += 1
    if is_attack:
        stats["total_attacks"] += 1
    if rf_pred == 1:
        stats["rf_attacks"] += 1
    if iso_pred == 1:
        stats["iso_attacks"] += 1
    if rf_pred == 1 and iso_pred == 1:
        stats["both_attacks"] += 1
    if rf_pred == 1 and iso_pred == 0:
        stats["only_rf"] += 1
    if rf_pred == 0 and iso_pred == 1:
        stats["only_iso"] += 1

    return {
        "rf_prediction":  rf_pred,
        "rf_confidence":  round(rf_proba, 4),
        "iso_prediction": iso_pred,
        "iso_score":      round(iso_score, 4),
        "is_attack":      is_attack,
        "detector":       detector
    }


@app.get("/")
def root():
    return {
        "status":  "API opérationnelle",
        "models":  {
            "random_forest":    "supervisé",
            "isolation_forest": "non supervisé"
        },
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features():
    return {
        "features": FEATURES,
        "count":    len(FEATURES)
    }


@app.post("/predict")
def predict(req: FlowRecord):
    return predict_one(req.features)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return [
        predict_one(f.features) for f in req.flows
    ]


@app.get("/stats")
def get_stats():
    rate = 0.0
    if stats["total_flows"] > 0:
        rate = round(
            stats["total_attacks"]
            / stats["total_flows"] * 100, 2
        )
    return {**stats, "attack_rate_percent": rate}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data   = await websocket.receive_text()
            result = predict_one(
                json.loads(data)["features"]
            )
            await websocket.send_text(
                json.dumps(result)
            )
    except Exception:
        pass


if __name__ == '__main__':
    uvicorn.run(
        app, host="0.0.0.0",
        port=8000, reload=False
    )