from flask import Flask, render_template, jsonify, request
import pandas as pd
import requests
import json
import os
from datetime import datetime
from config import Config
import wandb
import shutil
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv() 
wandb_api_key = os.environ.get('WANDB_API_KEY') 

app = Flask(__name__)
app.config.from_object(Config)

# Ensure data directory exists
os.makedirs(os.path.dirname(Config.DATA_FILE_PATH), exist_ok=True)


# Thêm hàm kết nối WandB
def init_wandb():
    """Initialize WandB connection"""
    try:
        wandb.login(key=wandb_api_key)
        run = wandb.init(project="attendance_face_recognition", job_type="api")
        return run
    except Exception as e:
        print(f"Failed to connect to WandB: {str(e)}")
        return None

# Thêm hàm tải tệp WandB
def download_latest_log(run, output_path=Config.DATA_FILE_PATH):
    """Download the latest log file from WandB"""
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
            return None
    except Exception as e:
        print(f"Error downloading log: {str(e)}")
        return None

# Sửa lại hàm refresh_data() trong app.py
@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """API endpoint to trigger data refresh directly from WandB"""
    try:
        # Khởi tạo WandB và tải file
        wandb_run = init_wandb()
        if not wandb_run:
            return jsonify({
                'status': 'error',
                'message': "Failed to connect to WandB"
            }), 500
            
        output_path = download_latest_log(wandb_run)
        wandb_run.finish()  # Đóng kết nối WandB
        
        if output_path:
            return jsonify({
                'status': 'success',
                'message': 'Data refreshed successfully',
                'file_path': output_path
            })
        else:
            return jsonify({
                'status': 'error',
                'message': "Failed to download log file"
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error refreshing data: {str(e)}"
        }), 500

def get_attendance_data():
    """Read and process attendance data from CSV file"""
    try:
        if os.path.exists(Config.DATA_FILE_PATH):
            df = pd.read_csv(Config.DATA_FILE_PATH)
            # Convert timestamp column to datetime if it's not already
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return pd.DataFrame(columns=['student_id', 'student_name', 'timestamp', 'status'])
    except Exception as e:
        print(f"Error reading attendance data: {str(e)}")
        return pd.DataFrame(columns=['student_id', 'student_name', 'timestamp', 'status'])

@app.route('/')
def home():
    """Render home page"""
    return render_template('home.html')

@app.route('/log')
def log_page():
    """Render log page"""
    return render_template('log.html')

@app.route('/api/attendance-data')
def attendance_data():
    """API endpoint to get attendance data"""
    df = get_attendance_data()
    
    # Convert DataFrame to dict for JSON serialization
    if not df.empty:
        # Convert timestamp to string for JSON serialization
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'status': 'success',
            'data': df.to_dict(orient='records')
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No data available'
        })

@app.route('/api/attendance-summary')
def attendance_summary():
    """API endpoint to get attendance summary statistics"""
    df = get_attendance_data()
    
    if df.empty:
        return jsonify({
            'status': 'success',
            'data': {
                'total_students': 0,
                'successful_recognitions': 0,
                'failed_recognitions': 0,
                'success_rate': 0,
                'unique_students': 0
            }
        })
    
    # Convert status to numeric if it's not already
    df['status'] = pd.to_numeric(df['status'])
    
    # Calculate summary statistics
    total_records = len(df)
    successful_recognitions = int(df[df['status'] == 1]['status'].count())
    failed_recognitions = int(df[df['status'] == 0]['status'].count())
    success_rate = round((successful_recognitions / total_records * 100), 2) if total_records > 0 else 0
    unique_students = df['student_id'].nunique()
    
    return jsonify({
        'status': 'success',
        'data': {
            'total_records': total_records,
            'successful_recognitions': successful_recognitions,
            'failed_recognitions': failed_recognitions,
            'success_rate': success_rate,
            'unique_students': unique_students
        }
    })

@app.route('/api/attendance-by-date')
def attendance_by_date():
    """API endpoint to get attendance data grouped by date"""
    df = get_attendance_data()
    
    if df.empty:
        return jsonify({
            'status': 'success',
            'data': []
        })
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['date'] = df['timestamp'].dt.date.astype(str)
    else:
        return jsonify({
            'status': 'error',
            'message': 'Timestamp column not found'
        })
    
    # Convert status to numeric if it's not already
    df['status'] = pd.to_numeric(df['status'])
    
    # Group by date and count success/failure
    daily_stats = df.groupby('date').agg(
        successful=('status', lambda x: (x == 1).sum()),
        failed=('status', lambda x: (x == 0).sum())
    ).reset_index()
    
    return jsonify({
        'status': 'success',
        'data': daily_stats.to_dict(orient='records')
    })

@app.route('/api/student-success-rate')
def student_success_rate():
    """API endpoint to get success rate by student"""
    df = get_attendance_data()
    
    if df.empty:
        return jsonify({
            'status': 'success',
            'data': []
        })
    
    # Convert status to numeric if it's not already
    df['status'] = pd.to_numeric(df['status'])
    
    # Group by student and calculate success rate
    student_stats = df.groupby(['student_id', 'student_name']).agg(
        successful=('status', lambda x: (x == 1).sum()),
        failed=('status', lambda x: (x == 0).sum()),
        total=('status', 'count')
    ).reset_index()
    
    student_stats['success_rate'] = (student_stats['successful'] / student_stats['total'] * 100).round(2)
    
    return jsonify({
        'status': 'success',
        'data': student_stats.to_dict(orient='records')
    })



if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)