import requests
import json

# Remember to run 
# uvicorn api.main:app --reload
# in the terminal to start the FastAPI server before running this test script.

# The API address running on localhost
BASE_URL = "http://127.0.0.1:8000"

# Test endpoint: /load_model (POST request)
def test_load_model():
    url = f"{BASE_URL}/load_model"
    response = requests.post(url)
    if response.status_code == 200:
        print(f"[Test] /load_model: SUCCESS")
    else:
        print(f"[Test] /load_model: FAILED with status code {response.status_code}")

# Test endpoint: /predict (POST request)
def test_predict():
    url = f"{BASE_URL}/predict"
    headers = {"Content-Type": "application/json"}
    data = {"embedding": [0.1] * 512}  # Simulated embedding array
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        print(f"[Test] /predict: SUCCESS - Student ID: {result['student_id']}, Confidence: {result['confidence']}")
    else:
        print(f"[Test] /predict: FAILED with status code {response.status_code}")

# Test endpoint: /load_json_data (GET request)
def test_load_json_data():
    url = f"{BASE_URL}/load_json_data"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"[Test] /load_json_data: SUCCESS")
    else:
        print(f"[Test] /load_json_data: FAILED with status code {response.status_code}")

# Test endpoint: /load_data (GET request)
def test_load_data():
    url = f"{BASE_URL}/load_data"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"[Test] /load_data: SUCCESS")
    else:
        print(f"[Test] /load_data: FAILED with status code {response.status_code}")

# Test endpoint: /save_json (POST request)
def test_save_json():
    url = f"{BASE_URL}/save_json"
    headers = {"Content-Type": "application/json"}
    data = {"123": {"name": "John Doe", "id": 123}}  # Example student data
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print(f"[Test] /save_json: SUCCESS")
    else:
        print(f"[Test] /save_json: FAILED with status code {response.status_code}")

# Test endpoint: /save_data (POST request)
def test_save_data():
    url = f"{BASE_URL}/save_data"
    headers = {"Content-Type": "application/json"}
    data = {123: [0.1] * 512}  # Simulated embedding data
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print(f"[Test] /save_data: SUCCESS")
    else:
        print(f"[Test] /save_data: FAILED with status code {response.status_code}")

# Test endpoint: /save_log (POST request)
def test_save_log():
    url = f"{BASE_URL}/save_log"
    headers = {"Content-Type": "application/json"}
    data = {"log": [{"student_id": 123, "status": "present"}]}  # Example log
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print(f"[Test] /save_log: SUCCESS")
    else:
        print(f"[Test] /save_log: FAILED with status code {response.status_code}")

if __name__ == "__main__":
    # Check each endpoint
    test_load_model()
    test_predict()
    test_load_json_data()
    test_load_data()
    test_save_json()
    test_save_data()
    test_save_log()
