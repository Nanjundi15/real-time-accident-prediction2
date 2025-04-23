from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
import joblib
from keras.models import load_model
import os
import sqlite3
from datetime import datetime
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# Set model directory
model_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
logistic_model = joblib.load(os.path.join(model_dir, 'logistic_model.pkl'))
decision_tree_model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
random_forest_model = joblib.load(os.path.join(model_dir, 'random_forest_model_compressed.pkl'))
xgboost_model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
lstm_model = load_model(os.path.join(model_dir, 'lstm_model.h5'))
scaler = joblib.load(os.path.join(model_dir, 'final_scaler.pkl'))

class_names = {
    0: 'No Accident',
    1: 'Minor Accident',
    2: 'Moderate Accident',
    3: 'Severe Accident'
}

# Database setup
db_path = os.path.join(model_dir, 'predictions.db')
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    features TEXT,
                    latitude REAL,
                    longitude REAL,
                    logistic_pred TEXT,
                    decision_tree_pred TEXT,
                    random_forest_pred TEXT,
                    xgboost_pred TEXT,
                    lstm_pred TEXT,
                    logistic_probs TEXT,
                    decision_tree_probs TEXT,
                    random_forest_probs TEXT,
                    xgboost_probs TEXT,
                    lstm_probs TEXT
                )''')
conn.commit()

# Email sending function
def send_email(subject, body, recipient_email):
    sender_email = "Nanjundi9731@gmail.com"
    sender_password = "wvpl ovah dkgc dclg"  # Use Gmail app password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# Empty template (replace with real HTML if needed)
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>🚦 Real-Time Traffic Accident Prediction | AI-Powered</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {
            padding: 30px;
            background: linear-gradient(to bottom, #e0f7fa, #fff);
            font-family: 'Segoe UI', sans-serif;
        }
        .card {
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .prediction-block {
            margin-top: 30px;
        }
        .table-container {
            margin-top: 50px;
        }
        .header-icon {
            width: 60px;
            height: 60px;
        }
        .emoji-card {
            font-size: 28px;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="text-center mb-4">
        <img src="https://cdn-icons-png.flaticon.com/512/854/854894.png" alt="Traffic Icon" class="header-icon">
        <h1>🚦 AI-Powered Real-Time Traffic Accident Predictor 🚗💥</h1>
        <p class="text-muted">Predict accidents before they happen using ML + Deep Learning ⚙️🤖</p>
    </div>

    <form method="POST" action="/result" class="bg-white p-4 rounded shadow-sm">
    <div class="mb-3">
        <label for="features" class="form-label">🔢 Enter Features (comma separated):</label>
        <input type="text" class="form-control" id="features" name="features" placeholder="e.g. 0.2, 0.5, 1.3, ..." required>
        <small class="form-text text-muted">Each feature represents a specific aspect of the traffic situation. For example:</small>
        <ul>
            <li><strong>Feature 1:</strong> Traffic Density (e.g., 0.2 - low, 1.0 - high)</li>
            <li><strong>Feature 2:</strong> Road Condition (e.g., 0.1 - good, 1.0 - poor)</li>
            <li><strong>Feature 3:</strong> Weather Condition (e.g., 0.5 - clear, 1.0 - heavy rain)</li>
            <li><strong>Feature 4:</strong> Time of Day (e.g., 0.0 - morning, 1.0 - night)</li>
        </ul>
    </div>
    <div class="mb-3">
        <label for="latitude" class="form-label">🌍 Latitude:</label>
        <input type="text" class="form-control" id="latitude" name="latitude" placeholder="e.g. 12.9716" required>
    </div>
    <div class="mb-3">
        <label for="longitude" class="form-label">🧭 Longitude:</label>
        <input type="text" class="form-control" id="longitude" name="longitude" placeholder="e.g. 77.5946" required>
    </div>
    <button type="submit" class="btn btn-primary">🚀 Predict Now</button>
    <a href="/dashboard" class="btn btn-success">📊 Dashboard</a>
    <a href="/download" class="btn btn-warning">⬇️ Download CSV</a>
    </form>


    {% if predictions %}
        <div class="prediction-block">
            <h2 class="mt-5 text-center">🎯 Model Predictions</h2>
            <div class="row">
                {% for model, pred in predictions.items() %}
                    <div class="col-md-4">
                        <div class="card p-4 text-center emoji-card bg-light mt-3">
                            <h5>{{ model }}</h5>
                            <p><strong>{{ pred }}</strong> 💡</p>
                        </div>
                    </div>
                {% endfor %}
            </div>
        </div>
    {% endif %}

    {% if model_probs %}
    <h3 class="mt-5 text-center">📈 Model Probabilities</h3>
    <ul class="list-group">
        {% for model, probs in model_probs.items() %}
            <li class="list-group-item">
                <strong>{{ model }}:</strong>
                <ul>
                    {% for label, prob in zip(class_labels, probs) %}
                        <li>{{ label }}: {{ '%.2f' | format(prob * 100) }}%</li>
                    {% endfor %}
                </ul>
            </li>
        {% endfor %}
    </ul>

    <div class="my-5">
        <h4 class="text-center mb-3">📊 Probability Comparison Chart</h4>
        <canvas id="probabilityChart" height="120"></canvas>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const modelLabels = {{ model_probs.keys() | list | tojson }};
        const classLabels = {{ class_labels | tojson }};
        const modelData = {{ model_probs.values() | list | tojson }};

        const datasets = classLabels.map((label, idx) => ({
            label: label,
            data: modelData.map(p => p[idx] * 100),
            backgroundColor: `hsl(${(idx * 90) % 360}, 70%, 60%)`
        }));

        new Chart(document.getElementById('probabilityChart'), {
            type: 'bar',
            data: {
                labels: modelLabels,
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    title: { display: true, text: 'Model Probability (%) Across Classes' }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Probability (%)' }
                    }
                }
            }
        });
    </script>
{% endif %}


    {% if rows %}
        <div class="table-container">
            <h2 class="text-center">📜 Prediction History</h2>
            <table class="table table-bordered table-hover table-striped">
                <thead class="table-dark">
                    <tr>
                        <th>ID</th>
                        <th>🕒 Timestamp</th>
                        <th>📊 Features</th>
                        <th>🌍 Lat</th>
                        <th>🧭 Lon</th>
                        <th>📘 Logistic</th>
                        <th>🌳 Decision Tree</th>
                        <th>🌲 Random Forest</th>
                        <th>⚡ XGBoost</th>
                        <th>🧠 LSTM</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in rows %}
                        <tr>
                            <td>{{ row[0] }}</td>
                            <td>{{ row[1] }}</td>
                            <td>{{ row[2] }}</td>
                            <td>{{ row[3] }}</td>
                            <td>{{ row[4] }}</td>
                            <td>{{ row[5] }}</td>
                            <td>{{ row[6] }}</td>
                            <td>{{ row[7] }}</td>
                            <td>{{ row[8] }}</td>
                            <td>{{ row[9] }}</td>
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    {% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template)

@app.route('/result', methods=['POST'])
def result():
    features_raw = request.form.get("features")
    lat_raw = request.form.get("latitude")
    lon_raw = request.form.get("longitude")

    if not features_raw or not lat_raw or not lon_raw:
        return render_template_string(html_template, predictions={"Error": "Please fill all fields."}, zip=zip)

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
        features = list(map(float, features_raw.strip().split(',')))

        input_array = np.array(features).reshape(1, -1)
        predictions = {}
        model_probs = {}

        models = {
            "Logistic Regression": logistic_model,
            "Decision Tree": decision_tree_model,
            "Random Forest": random_forest_model,
            "XGBoost": xgboost_model
        }

        for model_name, model in models.items():
            pred = int(model.predict(input_array)[0])
            predictions[model_name] = class_names.get(pred)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_array)[0]
                model_probs[model_name] = list(map(float, probs))

        lstm_input = np.array(features).reshape(1, 1, -1)
        lstm_probs = lstm_model.predict(lstm_input, verbose=0)[0]
        lstm_pred = int(np.argmax(lstm_probs))
        predictions["LSTM"] = class_names.get(lstm_pred)
        model_probs["LSTM"] = list(map(float, lstm_probs))

        # Email if severe
        if "Severe Accident" in predictions.values():
            subject = "🚨 Accident Alert: Severe Prediction"
            body = f"A Severe Accident was predicted!\n\nLatitude: {lat}\nLongitude: {lon}\n\nModel Predictions:\n" + \
                   "\n".join(f"{k}: {v}" for k, v in predictions.items())
            send_email(subject, body, "chetanj1005@gmail.com")

        cursor.execute('''INSERT INTO predictions (
            timestamp, features, latitude, longitude,
            logistic_pred, decision_tree_pred, random_forest_pred,
            xgboost_pred, lstm_pred,
            logistic_probs, decision_tree_probs, random_forest_probs, xgboost_probs, lstm_probs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                features_raw, lat, lon,
                predictions.get("Logistic Regression"),
                predictions.get("Decision Tree"),
                predictions.get("Random Forest"),
                predictions.get("XGBoost"),
                predictions.get("LSTM"),
                str(model_probs.get("Logistic Regression")),
                str(model_probs.get("Decision Tree")),
                str(model_probs.get("Random Forest")),
                str(model_probs.get("XGBoost")),
                str(model_probs.get("LSTM"))
            ))
        conn.commit()

        return render_template_string(
            html_template,
            predictions=predictions,
            model_probs=model_probs,
            lstm_probs=list(map(float, lstm_probs)),
            class_labels=list(class_names.values()),
            zip=zip
        )

    except ValueError:
        return render_template_string(html_template, predictions={"Error": "Invalid input! Ensure fields are numbers."}, zip=zip)

@app.route('/predict', methods=['POST'])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type. Set Content-Type: application/json"}), 415

    data = request.get_json()

    try:
        features = data.get("features")
        latitude = float(data.get("latitude"))
        longitude = float(data.get("longitude"))

        if not isinstance(features, list):
            raise ValueError("Features should be a list of numbers.")

        input_array = np.array(features).reshape(1, -1)
        predictions = {}
        model_probs = {}

        models = {
            "Logistic Regression": logistic_model,
            "Decision Tree": decision_tree_model,
            "Random Forest": random_forest_model,
            "XGBoost": xgboost_model
        }

        for model_name, model in models.items():
            pred = int(model.predict(input_array)[0])
            predictions[model_name] = class_names.get(pred)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_array)[0]
                model_probs[model_name] = list(map(float, probs))

        lstm_input = np.array(features).reshape(1, 1, -1)
        lstm_probs = lstm_model.predict(lstm_input, verbose=0)[0]
        lstm_pred = int(np.argmax(lstm_probs))
        predictions["LSTM"] = class_names.get(lstm_pred)
        model_probs["LSTM"] = list(map(float, lstm_probs))

        if "Severe Accident" in predictions.values():
            subject = "🚨 Accident Alert: Severe Prediction"
            body = f"Severe Accident Predicted!\n\nLat: {latitude}\nLon: {longitude}\nPredictions:\n" + \
                   "\n".join(f"{k}: {v}" for k, v in predictions.items())
            send_email(subject, body, "chetanj1005@gmail.com")

        cursor.execute('''INSERT INTO predictions (
            timestamp, features, latitude, longitude,
            logistic_pred, decision_tree_pred, random_forest_pred,
            xgboost_pred, lstm_pred,
            logistic_probs, decision_tree_probs, random_forest_probs, xgboost_probs, lstm_probs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                str(features), latitude, longitude,
                predictions.get("Logistic Regression"),
                predictions.get("Decision Tree"),
                predictions.get("Random Forest"),
                predictions.get("XGBoost"),
                predictions.get("LSTM"),
                str(model_probs.get("Logistic Regression")),
                str(model_probs.get("Decision Tree")),
                str(model_probs.get("Random Forest")),
                str(model_probs.get("XGBoost")),
                str(model_probs.get("LSTM"))
            ))
        conn.commit()

        return jsonify({
            "predictions": predictions,
            "probabilities": model_probs,
            "class_labels": list(class_names.values())
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

@app.route('/dashboard')
def dashboard():
    cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 10')
    rows = cursor.fetchall()
    return render_template_string(html_template, rows=rows)

@app.route('/download')
def download_csv():
    cursor.execute('SELECT * FROM predictions')
    rows = cursor.fetchall()

    csv_path = os.path.join(model_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([col[0] for col in cursor.description])
        writer.writerows(rows)

    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
