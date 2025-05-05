from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, model_validator
from fastapi.responses import HTMLResponse
from typing import List
import pandas as pd
import joblib
import os
import json
import wandb

# --- Config ---
artifact_model_name = "attendance_face_recognition/model_export:latest"
model_path = "model_export/model.pkl"

artifact_json_name = "attendance_face_recognition/students.json:latest"

# --- Init App ---
app = FastAPI()

def load_model():
    global model
    run = wandb.init(project="attendance_face_recognition", job_type="api")
    model_export_path = run.use_artifact(artifact_model_name).download()
    model = joblib.load(os.path.join(model_export_path, "model_export"))
    print("-- Model loaded into memory. --")

def load_json_data(artifact_json_name):
    global data_json
    run = wandb.init(project="attendance_face_recognition", job_type="api")
    json_artifact_path = run.use_artifact(artifact_json_name).download()
    json_file_path = os.path.join(json_artifact_path, "students.json")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    print("-- JSON data loaded into memory. --")
    return data_json

# --- Pydantic Schema ---
class EmbeddingInput(BaseModel):
    embedding: List[float]

    @model_validator(mode="before")
    def check_length(cls, values):
        emb = values.get("embedding")
        if emb is None or len(emb) != 512:
            raise ValueError(f"‘embedding’ phải có đúng 512 giá trị (đã nhận {len(emb) if emb is not None else 0})")
        return values

    class Config:
        schema_extra = {
            "example": {
                "embedding": [0.1] * 512  # Dummy example
            }
        }

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Facial Recognition API</h1><p>Ready to predict 512-dim embeddings.</p>"

@app.post("/load_model")
def load_latest_model():
    load_model()
    return {"message": "Model loaded successfully."}

@app.post("/load_json_data")
def load_latest_json():
    students = load_json_data(artifact_json_name)
    return students

@app.post("/predict")
async def predict_student(input: EmbeddingInput):
    try:
        df = pd.DataFrame([input.embedding])
        pred = model.predict(df)[0]
        conf = max(model.predict_proba(df)[0])
        return {"student_id": int(pred), "confidence": round(conf, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
