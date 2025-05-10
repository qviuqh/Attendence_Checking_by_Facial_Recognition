import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-development-key')
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    
    # API configuration
    API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')
    
    # Data configuration
    DATA_FILE_PATH = os.environ.get('DATA_FILE_PATH', 'data/attendance_log.csv')
    
    # WandB configuration
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY', '')
    WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'attendance_face_recognition')

