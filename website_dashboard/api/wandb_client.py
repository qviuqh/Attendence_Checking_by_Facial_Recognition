import os
import wandb
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from dotenv import load_dotenv
load_dotenv() 
wandb_api_key = os.environ.get('WANDB_API_KEY') 

def init_wandb():
    """Initialize WandB connection"""
    try:
        wandb.login(key=wandb_api_key) 
        run = wandb.init(project="attendance_face_recognition", job_type="api")
        return run
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to WandB: {str(e)}")

def download_latest_log(run, output_path='data/attendance_log.csv'):
    """Download the latest log file from WandB and overwrite the local file"""
    artifact_name = "attendance_face_recognition/log:latest"
    
    try:
        latest_artifact = run.use_artifact(artifact_name)
        latest_log_path = latest_artifact.download()
        latest_log_file = os.path.join(latest_log_path, "log.csv")
        
        if os.path.exists(latest_log_file):
            # Ensure the data directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Copy the file content to overwrite the existing file
            with open(latest_log_file, 'rb') as source, open(output_path, 'wb') as dest:
                dest.write(source.read())
                
            return output_path
        else:
            raise HTTPException(status_code=404, detail="Log file not found in artifact.")
    except wandb.errors.CommError as e:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading log: {str(e)}")