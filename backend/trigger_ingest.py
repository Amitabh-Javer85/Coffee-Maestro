import os

import requests

url = "http://localhost:8000/api/ingest"
api_key = os.environ.get("API_SECRET_KEY")
if not api_key:
    raise SystemExit("API_SECRET_KEY must be set before triggering ingestion.")

headers = {
    "Content-Type": "application/json",
    "X-API-Key": api_key,
}

response = requests.post(url, headers=headers, json={})
print(response.status_code)
print(response.json())
