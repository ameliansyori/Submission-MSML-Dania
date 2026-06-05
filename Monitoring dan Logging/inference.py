import requests
import time
import random

url = "http://localhost:5002/predict"
payload = {"age": 52, "sex": 1, "cp": 0, "trestbps": 125}

print("Memulai pengiriman data test ke API Model untuk memicu metrik...")
for i in range(20):
    try:
        res = requests.post(url, json=payload)
        print(f"Request ke-{i+1} status: {res.status_code}")
    except Exception as e:
        print("Pastikan prometheus_exporter.py sudah dijalankan!")
    time.sleep(0.5)
