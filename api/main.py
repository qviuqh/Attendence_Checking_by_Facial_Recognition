from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, model_validator
from fastapi.responses import HTMLResponse
from typing import Dict, List, Any
import pandas as pd
import joblib
import os
import json
import wandb
from fastapi.responses import StreamingResponse
import io

# --- Init WandB ---
run = wandb.init(project="attendance_face_recognition", job_type="api")

# --- Config ---
artifact_model_name = "attendance_face_recognition/model_export:latest"
model_path = "model_export/model.pkl"

artifact_json_name = "attendance_face_recognition/students.json:latest"
artifact_data_name = "attendance_face_recognition/embedding_data.csv:latest"

# --- Init App ---
app = FastAPI()

def load_model():
    global model
    model_export_path = run.use_artifact(artifact_model_name).download()
    model = joblib.load(os.path.join(model_export_path, "model_export"))
    print("-- Model loaded into memory. --")

def load_json_data(artifact_json_name):
    global data_json
    json_artifact_path = run.use_artifact(artifact_json_name).download()
    json_file_path = os.path.join(json_artifact_path, "students.json")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    print("-- JSON data loaded into memory. --")
    return data_json

def load_data(artifact_data_name):
    global data
    data_artifact_path = run.use_artifact(artifact_data_name).download()
    data_file_path = os.path.join(data_artifact_path, "embedding_data.csv")
    data = pd.read_csv(data_file_path, encoding='utf-8')
    print("-- CSV data loaded into memory. --")
    return data

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

@app.get("/load_json_data")
def load_latest_json():
    students = load_json_data(artifact_json_name)
    return students

@app.get("/load_data")
def load_latest_data():
    data = load_data(artifact_data_name)
    stream = io.StringIO()
    data.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=embedding_data.csv"})

@app.get("/load_log")
def load_log():
    # Check if artifact exists
    artifact_name = "attendance_face_recognition/log:latest"
    try:
        latest_artifact = run.use_artifact(artifact_name)
        latest_log_path = latest_artifact.download()
        latest_log_file = os.path.join(latest_log_path, "log.csv")
        if os.path.exists(latest_log_file):
            return StreamingResponse(open(latest_log_file, mode="rb"), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=log.csv"})
        else:
            raise HTTPException(status_code=404, detail="Log file not found.")
    except wandb.errors.CommError:
        raise HTTPException(status_code=404, detail="Artifact not found.")

@app.post("/save_json")
async def save_json(students: dict):
    try:
        latest_json = load_json_data(artifact_json_name)
        # Merge new data with existing data
        latest_json.update(students)
        # Save the updated data back to the JSON file and upload it to W&B
        artifact = wandb.Artifact(name="students.json", type="data")
        with open("students.json", "w", encoding='utf-8') as f:
            json.dump(latest_json, f, ensure_ascii=False, indent=4)
        artifact.add_file("students.json")
        run.log_artifact(artifact)
        os.remove("students.json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save_data")
async def save_data(data: Dict[int, List[float]]):
    try:
        latest_data = load_data(artifact_data_name)
        df = pd.DataFrame(data)
        merge_data = pd.concat([df, latest_data], ignore_index=True)
        # Save the updated data back to the CSV file and upload it to W&B
        artifact = wandb.Artifact(name="embedding_data.csv", type="data")
        merge_data.to_csv("embedding_data.csv", index=False, encoding='utf-8')
        artifact.add_file("embedding_data.csv")
        run.log_artifact(artifact)
        os.remove("embedding_data.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save_log")
async def save_file_log(log: Dict[str, List[Any]]):
    try:
        # Save log to a local file
        df = pd.DataFrame(log)
        # Upload log file as a W&B artifact
        # Check if artifact exists
        artifact_name = "attendance_face_recognition/log:latest"
        try:
            latest_artifact = run.use_artifact(artifact_name)
            latest_log_path = latest_artifact.download()
            latest_log_file = os.path.join(latest_log_path, "log.csv")
            if os.path.exists(latest_log_file):
                latest_df = pd.read_csv(latest_log_file, encoding='utf-8')
                df = pd.concat([latest_df, df], ignore_index=True)
                df.to_csv("log.csv", index=False, encoding='utf-8')
                artifact = wandb.Artifact(name="log", type="data")
                artifact.add_file("log.csv")
                run.log_artifact(artifact)
                # os.remove("log.csv")
        except wandb.errors.CommError:
            # Artifact does not exist, proceed to create new
            artifact = wandb.Artifact(name="log", type="data")
            df.to_csv("log.csv", index=False, encoding='utf-8')
            artifact.add_file("log.csv")
            run.log_artifact(artifact)
            # os.remove("log.csv")
        return {"message": "Log saved and uploaded to W&B successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict_student(input: EmbeddingInput):
    try:
        df = pd.DataFrame([input.embedding])
        pred = model.predict(df)[0]
        conf = max(model.predict_proba(df)[0])
        return {"student_id": int(pred), "confidence": round(conf, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
