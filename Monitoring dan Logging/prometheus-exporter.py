import time
import psutil
import random
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest, REGISTRY


app = Flask(__name__)

# Mendaftarkan 5 Metriks Berbeda untuk syarat SKILLED
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('api_model_latency_seconds', 'Latensi API Model dalam Detik')
CPU_USAGE = Gauge('system_cpu_usage', 'Penggunaan CPU dalam Persen')
RAM_USAGE = Gauge('system_ram_usage', 'Penggunaan RAM dalam Megabyte')
MODEL_ERROR_COUNT = Counter('model_prediction_errors_total', 'Total Eror pada Prediksi Model')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    # pembacaan sistem monitoring
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    try:
        data = request.get_json()
        time.sleep(0.1) 
        
        if not data:
            raise ValueError("Data kosong")
            
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        return jsonify({"status": "success", "prediction": 1})
        
    except Exception as e:
        MODEL_ERROR_COUNT.inc()
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    # Biar pas Prometheus ngetok, cpu & ram langsung update otomatis tanpa nunggu /predict
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    #start_http_server(8000)
    app.run(host='0.0.0.0', port=5002)
