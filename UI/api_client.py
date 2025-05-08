import requests
import json

class APIClient:
    """Client for calling remote face recognition API"""
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}

    def load_model(self):
        url = f"{self.base_url}/load_model"
        response = requests.post(url)
        response.raise_for_status()

    def load_json_data(self):
        url = f"{self.base_url}/load_json_data"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def push_json_data(self, json_data):
        url = f"{self.base_url}/save_json"
        response = requests.post(url, json=json_data)
        response.raise_for_status()

    def push_data(self, data):
        try:
            resp = requests.post(f"{self.base_url}/save_data", json=data.to_dict(orient="list"))
            resp.raise_for_status()
            return resp.json()   # chứa message + rows_before/after
        except requests.HTTPError as err:
            print("Push failed:", resp.status_code, resp.text)
            raise err

    def push_log(self, log_data):
        url = f"{self.base_url}/save_log"
        response = requests.post(url, json=log_data.to_dict(orient="list"))
        response.raise_for_status()

    def predict(self, vec_embedding):
        url = f"{self.base_url}/predict"
        files = {"embedding": vec_embedding.tolist()}
        response = requests.post(url, json=files)
        response.raise_for_status()
        return response