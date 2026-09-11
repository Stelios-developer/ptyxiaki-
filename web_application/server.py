import json
import uuid
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS = ["Random Forest", "Logistic Regression", "Linear SVM", "kNN", "ANN"]
FILES = {}
PIPELINES = {}
CONFIG = {
    "CICIDS2017": {
        "artifact": "CICIDS2017",
        "models": {
            "Logistic Regression": ROOT / "models" / "5.3.1_Logistic_Regression" / "CICIDS2017_logistic_regression.joblib",
            "Random Forest": ROOT / "models" / "5.3.2_Random_Forest" / "CICIDS2017_random_forest.joblib",
            "Linear SVM": ROOT / "models" / "5.3.3_Linear_SVM" / "CICIDS2017_linear_svm.joblib",
            "kNN": ROOT / "models" / "5.3.4_kNN" / "CICIDS2017_knn.joblib",
            "ANN": ROOT / "models" / "5.3.5_ANN" / "CICIDS2017_ann.joblib",
        },
        "preview": ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"],
    },
    "UNSW-NB15": {
        "artifact": "UNSW_NB15",
        "models": {
            "Logistic Regression": ROOT / "models" / "5.3.1_Logistic_Regression" / "UNSW_NB15_logistic_regression.joblib",
            "Random Forest": ROOT / "models" / "5.3.2_Random_Forest" / "UNSW_NB15_random_forest.joblib",
            "Linear SVM": ROOT / "models" / "5.3.3_Linear_SVM" / "UNSW_NB15_linear_svm.joblib",
            "kNN": ROOT / "models" / "5.3.4_kNN" / "UNSW_NB15_knn.joblib",
            "ANN": ROOT / "models" / "5.3.5_ANN" / "UNSW_NB15_ann.joblib",
        },
        "preview": ["proto", "service", "state", "dur", "sbytes", "dbytes"],
    },
}


def load_pipeline(dataset):
    if dataset in PIPELINES:
        return PIPELINES[dataset]
    artifact = CONFIG[dataset]["artifact"]
    encoding = ROOT / "data" / "4.2.2_Categorical_Encoding" / "artifacts"
    normalization = ROOT / "data" / "4.2.3_Feature_Normalization" / "artifacts"
    selection = ROOT / "data" / "4.2.5_Feature_Selection" / "artifacts"
    pipeline = {
        "features": json.loads((encoding / f"{artifact}_raw_feature_names.json").read_text(encoding="utf-8")),
        "encoder": joblib.load(encoding / f"{artifact}_encoder.joblib"),
        "scaler": joblib.load(normalization / f"{artifact}_minmax_scaler.joblib"),
        "selector": joblib.load(selection / f"{artifact}_feature_selector.joblib"),
        "models": {name: joblib.load(path) for name, path in CONFIG[dataset]["models"].items()},
    }
    PIPELINES[dataset] = pipeline
    return pipeline


def prepare_data(data, pipeline):
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()
    data = data.replace([np.inf, -np.inf], np.nan)
    usable = data.loc[:, pipeline["features"]].dropna()
    encoded = pipeline["encoder"].transform(usable)
    scaled = pipeline["scaler"].transform(np.asarray(encoded, dtype=np.float32))
    selected = scaled[:, pipeline["selector"]["selected_indices"]]
    return data.loc[usable.index].copy(), np.asarray(selected, dtype=np.float32)


def predictions(features, pipeline, choice):
    names = MODELS if choice == "Compare all models" else [choice]
    return {name: pipeline["models"][name].predict(features).astype(int) for name in names}


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
async def analyze(dataset: str = Form(...), model: str = Form(...), file: UploadFile = File(...)):
    try:
        data = pd.read_csv(BytesIO(await file.read()), low_memory=False)
        pipeline = load_pipeline(dataset)
        usable, features = prepare_data(data, pipeline)
        if len(usable) == 0:
            raise ValueError()
        results = predictions(features, pipeline, model)
    except Exception:
        raise HTTPException(status_code=400, detail="Δεν ήταν δυνατή η ανάλυση του αρχείου.")

    output = usable.copy()
    summary = []
    prediction_columns = []
    for name, labels in results.items():
        column = "Prediction" if len(results) == 1 else f"Prediction_{name.replace(' ', '_')}"
        prediction_columns.append(column)
        output[column] = np.where(labels == 1, "Attack", "Normal")
        attacks = int(labels.sum())
        summary.append({
            "model": name,
            "normal": int(len(labels) - attacks),
            "attack": attacks,
            "attack_rate": round(attacks / len(labels) * 100, 2),
        })

    preview_columns = prediction_columns + [column for column in CONFIG[dataset]["preview"] if column in output.columns]
    preview = output.loc[:, preview_columns]
    token = uuid.uuid4().hex
    FILES[token] = output.to_csv(index=False).encode("utf-8-sig")
    return {
        "rows": int(len(output)),
        "summary": summary,
        "preview_columns": preview.columns.tolist(),
        "preview_rows": json.loads(preview.to_json(orient="records")),
        "download": f"/api/download/{token}",
    }


@app.get("/api/download/{token}")
def download(token: str):
    if token not in FILES:
        raise HTTPException(status_code=404, detail="Το αρχείο δεν είναι διαθέσιμο.")
    return Response(
        content=FILES[token],
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=network_attack_predictions.csv"},
    )
