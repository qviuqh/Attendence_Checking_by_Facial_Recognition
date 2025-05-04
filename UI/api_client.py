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

    def predict(self, vec_embedding):
        url = f"{self.base_url}/predict"
        files = {"embedding": vec_embedding.tolist()}
        response = requests.post(url, json=files)
        response.raise_for_status()
        return response