from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from . import wandb_client

app = FastAPI(title="Attendance Tracking API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WandB only when needed
wandb_run = None

@app.get("/api/refresh-data")
async def refresh_data():
    """Refresh attendance data by downloading the latest log from WandB"""
    global wandb_run
    
    try:
        # Initialize WandB connection if not already done
        if wandb_run is None:
            wandb_run = wandb_client.init_wandb()
        
        # Download and overwrite the local log file
        output_path = wandb_client.download_latest_log(wandb_run)
        
        return {"status": "success", "message": "Data refreshed successfully", "file_path": output_path}
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup WandB resources on shutdown"""
    global wandb_run
    if wandb_run:
        wandb_run.finish()