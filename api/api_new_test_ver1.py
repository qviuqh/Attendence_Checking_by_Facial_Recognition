import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

import pandas as pd
import joblib
import wandb
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ConfigDict

import io
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class Config:
    """Application configuration"""
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "attendance_face_recognition")
    WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "your-secret-key")
    
    MODEL_ARTIFACT: str = "attendance_face_recognition/model_export:latest"
    STUDENTS_ARTIFACT: str = "attendance_face_recognition/students.json:latest"
    EMBEDDING_ARTIFACT: str = "attendance_face_recognition/embedding_data.csv:latest"
    LOG_ARTIFACT: str = "attendance_face_recognition/log:latest"
    
    EMBEDDING_DIMENSION: int = 512
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        if not cls.WANDB_API_KEY:
            raise ValueError("WANDB_API_KEY environment variable is required")
        if not cls.API_SECRET_KEY:
            raise ValueError("API_SECRET_KEY environment variable is required")

config = Config()

class ModelManager:
    """Thread-safe model management"""
    
    def __init__(self):
        self._model = None
        self._students_data = {}
        self._embedding_data = None
        self._wandb_run = None
        self._lock = threading.RLock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize WandB and load model"""
        with self._lock:
            if self._initialized:
                return
            
            try:
                # Initialize WandB in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    self._wandb_run = await loop.run_in_executor(
                        executor, self._init_wandb
                    )
                
                await self.load_model()
                await self.load_students_data()
                await self.load_embedding_data()
                
                self._initialized = True
                logger.info("ModelManager initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize ModelManager: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service initialization failed"
                )
    
    def _init_wandb(self):
        """Initialize WandB (runs in thread pool)"""
        return wandb.init(
            project=config.WANDB_PROJECT,
            job_type="api",
            reinit=True
        )
    
    async def load_model(self):
        """Load ML model from WandB artifact"""
        with self._lock:
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    artifact_path = await loop.run_in_executor(
                        executor, 
                        lambda: self._wandb_run.use_artifact(config.MODEL_ARTIFACT).download()
                    )
                    
                    model_file = Path(artifact_path) / "model_export"
                    self._model = await loop.run_in_executor(
                        executor, joblib.load, str(model_file)
                    )
                
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model loading failed"
                )
    
    async def load_students_data(self):
        """Load students data from WandB artifact"""
        with self._lock:
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    artifact_path = await loop.run_in_executor(
                        executor,
                        lambda: self._wandb_run.use_artifact(config.STUDENTS_ARTIFACT).download()
                    )
                    
                    json_file = Path(artifact_path) / "students.json"
                    with open(json_file, 'r', encoding='utf-8') as f:
                        self._students_data = json.load(f)
                
                logger.info("Students data loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load students data: {e}")
                self._students_data = {}
    
    async def load_embedding_data(self):
        """Load embedding data from WandB artifact"""
        with self._lock:
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    artifact_path = await loop.run_in_executor(
                        executor,
                        lambda: self._wandb_run.use_artifact(config.EMBEDDING_ARTIFACT).download()
                    )
                    
                    csv_file = Path(artifact_path) / "embedding_data.csv"
                    self._embedding_data = await loop.run_in_executor(
                        executor, pd.read_csv, str(csv_file)
                    )
                
                logger.info("Embedding data loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load embedding data: {e}")
                self._embedding_data = pd.DataFrame()
    
    def get_model(self):
        """Get the loaded model (thread-safe)"""
        with self._lock:
            if self._model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            return self._model
    
    def get_students_data(self) -> dict:
        """Get students data (thread-safe)"""
        with self._lock:
            return self._students_data.copy()
    
    def get_embedding_data(self) -> pd.DataFrame:
        """Get embedding data (thread-safe)"""
        with self._lock:
            return self._embedding_data.copy() if self._embedding_data is not None else pd.DataFrame()
    
    async def save_students_data(self, new_data: dict):
        """Save updated students data to WandB"""
        with self._lock:
            try:
                # Update local data
                self._students_data.update(new_data)
                
                # Save to temporary file and upload
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(self._students_data, f, ensure_ascii=False, indent=2)
                    temp_path = f.name
                
                # Upload to WandB
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    await loop.run_in_executor(
                        executor, self._upload_json_artifact, temp_path
                    )
                
                # Cleanup
                os.unlink(temp_path)
                logger.info("Students data saved successfully")
                
            except Exception as e:
                logger.error(f"Failed to save students data: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to save students data"
                )
    
    def _upload_json_artifact(self, file_path: str):
        """Upload JSON artifact to WandB"""
        artifact = wandb.Artifact(name="students.json", type="data")
        artifact.add_file(file_path, name="students.json")
        self._wandb_run.log_artifact(artifact)
    
    async def predict(self, embedding: List[float]) -> tuple[int, float]:
        """Make prediction using the loaded model"""
        model = self.get_model()
        
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                df = pd.DataFrame([embedding])
                prediction = await loop.run_in_executor(
                    executor, model.predict, df
                )
                probabilities = await loop.run_in_executor(
                    executor, model.predict_proba, df
                )
            
            student_id = int(prediction[0])
            confidence = float(max(probabilities[0]))
            
            return student_id, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed"
            )

# Global model manager
model_manager = ModelManager()

# Pydantic Models
class EmbeddingInput(BaseModel):
    """Input model for embedding prediction"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding": [0.1] * config.EMBEDDING_DIMENSION
            }
        }
    )
    
    embedding: List[float] = Field(
        ...,
        min_length=config.EMBEDDING_DIMENSION,
        max_length=config.EMBEDDING_DIMENSION,
        description=f"Face embedding vector with exactly {config.EMBEDDING_DIMENSION} dimensions"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    student_id: int = Field(..., description="Predicted student ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    prediction_id: str = Field(..., description="Unique prediction identifier")

class StudentsUpdateRequest(BaseModel):
    """Request model for updating students data"""
    students: Dict[str, Any] = Field(..., description="Students data to update")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    service_id: str

# Security functions
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API authentication token"""
    if credentials.credentials != config.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    try:
        config.validate()
        await model_manager.initialize()
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Application shutting down")

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Secure API for face recognition attendance system",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"]
)



# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    return """
    <h1>Face Recognition API v2.0</h1>
    <p>Secure API for attendance face recognition system</p>
    <ul>
        <li><a href="/docs">API Documentation</a></li>
        <li><a href="/health">Health Check</a></li>
    </ul>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager._model is not None,
        service_id=str(uuid.uuid4())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_student(
    input_data: EmbeddingInput,
    _: str = Depends(verify_token)
):
    """Predict student from face embedding"""
    try:
        student_id, confidence = await model_manager.predict(input_data.embedding)
        
        return PredictionResponse(
            student_id=student_id,
            confidence=round(confidence, 4),
            prediction_id=str(uuid.uuid4())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/students")
async def get_students(
    _: str = Depends(verify_token)
):
    """Get current students data"""
    try:
        return {"students": model_manager.get_students_data()}
    except Exception as e:
        logger.error(f"Get students error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve students data"
        )

@app.post("/students")
async def update_students(
    update_request: StudentsUpdateRequest,
    _: str = Depends(verify_token)
):
    """Update students data"""
    try:
        await model_manager.save_students_data(update_request.students)
        return {"message": "Students data updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update students error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update students data"
        )

@app.get("/embeddings/download")
async def download_embeddings(
    _: str = Depends(verify_token)
):
    """Download embedding data as CSV"""
    try:
        data = model_manager.get_embedding_data()
        
        # Create CSV stream
        stream = io.StringIO()
        data.to_csv(stream, index=False)
        stream.seek(0)
        
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=embeddings.csv"}
        )
        
    except Exception as e:
        logger.error(f"Download embeddings error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download embeddings"
        )

@app.post("/model/reload")
async def reload_model(
    _: str = Depends(verify_token)
):
    """Reload model from latest artifact"""
    try:
        await model_manager.load_model()
        return {"message": "Model reloaded successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
