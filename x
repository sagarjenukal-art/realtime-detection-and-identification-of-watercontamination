import time
import threading
import random
import numpy as np
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier

# ================================
# Flask App
# ================================
app = Flask(__name__)

latest_data = {}
latest_result = {}
lock = threading.Lock()

# ================================
# Sensor Data (Simulated)
# ================================
def read_sensors():
    return {
        "pH": round(random.uniform(5.5, 9.5), 2),
        "turbidity": round(random.uniform(0, 50), 2),
        "tds": round(random.uniform(50, 1500), 2),
        "temperature": round(random.uniform(10, 40), 2),
        "orp": round(random.uniform(100, 500), 2)
    }

# ================================
# ML Model Training
# ================================
def train_model():
    X = []
    y = []

    for _ in range(1000):
        ph = random.uniform(5.5, 9.5)
        turb = random.uniform(0, 50)
        tds = random.uniform(50, 1500)
        temp = random.uniform(10, 40)
        orp = random.uniform(100, 500)

        label = int(
            ph < 6.5 or ph > 8.5 or
            turb > 5 or
            tds > 500 or
            orp < 250
        )

        X.append([ph, turb, tds, temp, orp])
        y.append(label)

    model = RandomForestClassifier(n_estimators=50, random_state=1)
    model.fit(X, y)
    return model

model = train_model()

# ================================
# Background Monitoring Thread
# ================================
def monitor():
    global latest_data, latest_result

    while True:
        data = read_sensors()

        features = np.array([[ 
            data["pH"],
            data["turbidity"],
            data["tds"],
            data["temperature"],
            data["orp"]
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        with lock:
            latest_data = data
            latest_result = {
                "contaminated": bool(prediction),
                "confidence": round(probability, 2),
                "status": "UNSAFE" if prediction else "SAFE"
            }

        if prediction:
            print("⚠️ Water contamination detected")

        time.sleep(2)

# ================================
# API Endpoints
# ================================
@app.route("/data")
def get_data():
    with lock:
        return jsonify(latest_data)

@app.route("/result")
def get_result():
    with lock:
        return jsonify(latest_result)

@app.route("/status")
def status():
    return jsonify({"system": "running"})

# ================================
# Main
# ================================
if __name__ == "__main__":
    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
