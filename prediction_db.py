from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
from keras.models import load_model
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

model_dir = r"D:/cyber security/major project/real time/Project"

# Load models
logistic_model = joblib.load(os.path.join(model_dir, 'logistic_model.pkl'))
decision_tree_model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
random_forest_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
xgboost_model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
lstm_model = load_model(os.path.join(model_dir, 'lstm_model.h5'))

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

html_template = '''
<!doctype html>
<html>
<head>
    <title>🚦 Accident Severity Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f4f6f9; padding: 30px; }
        .container { background: white; padding: 30px; border-radius: 12px; max-width: 900px; margin: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; }
        input, button { padding: 10px; width: 100%; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; }
        button { background-color: #3498db; color: white; border: none; font-weight: bold; }
        .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px; }
        .card { background: #ecf0f1; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); text-align: center; }
        canvas { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚦 Real-Time Traffic Accident Prediction</h1>
        <form method="POST" action="/result">
            <label>Enter Features (comma-separated):</label>
            <input type="text" name="features" placeholder="e.g. 1,0,1,2,1,0" required>
            <button type="submit">Predict</button>
        </form>

        {% if predictions %}
            <h2>🧠 Model Predictions</h2>
            <div class="card-grid">
                {% for model, pred in predictions.items() %}
                    <div class="card">
                        <h3>{{ model }}</h3>
                        <p><strong>{{ pred }}</strong></p>
                    </div>
                {% endfor %}
            </div>
        {% endif %}

        {% if model_probs %}
            <h2>📊 Prediction Probabilities</h2>
            <canvas id="allModelChart" width="400" height="300"></canvas>
        {% endif %}

        {% if lstm_probs %}
            <h2>🔍 LSTM Class Probabilities</h2>
            <canvas id="lstmChart" width="400" height="300"></canvas>
        {% endif %}
    </div>

<script>
    {% if model_probs %}
    new Chart(document.getElementById('allModelChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: {{ class_labels|safe }},
            datasets: [
                {% for model, probs in model_probs.items() %}
                {
                    label: '{{ model }}',
                    data: {{ probs|safe }},
                    backgroundColor: 'rgba({{ 100+loop.index0*40 }}, 99, 132, 0.6)',
                    borderWidth: 1
                },
                {% endfor %}
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    {% endif %}

    {% if lstm_probs %}
    new Chart(document.getElementById('lstmChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: {{ class_labels|safe }},
            datasets: [{
                label: 'LSTM Class Probabilities',
                data: {{ lstm_probs|safe }},
                backgroundColor: ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    {% endif %}
</script>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template)


@app.route('/result', methods=['POST'])
def browser_predict():
    raw = request.form.get("features")
    try:
        features = list(map(float, raw.strip().split(',')))
        input_array = np.array(features).reshape(1, -1)

        predictions = {}
        model_probs = {}

        models = {
            "Logistic Regression": logistic_model,
            "Decision Tree": decision_tree_model,
            "Random Forest": random_forest_model,
            "XGBoost": xgboost_model
        }

        probs_dict = {}

        for model_name, model in models.items():
            pred = int(model.predict(input_array)[0])
            predictions[model_name] = class_names.get(pred)

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_array)[0]
                probs_dict[model_name] = probs
                model_probs[model_name] = list(map(float, probs))

        lstm_input = np.array(features).reshape(1, 1, -1)
        lstm_probs = lstm_model.predict(lstm_input, verbose=0)[0]
        lstm_pred = int(np.argmax(lstm_probs))
        predictions["LSTM"] = class_names.get(lstm_pred)
        model_probs["LSTM"] = list(map(float, lstm_probs))

        # Save to database
        cursor.execute('''INSERT INTO predictions (
            timestamp, features, logistic_pred, decision_tree_pred,
            random_forest_pred, xgboost_pred, lstm_pred,
            logistic_probs, decision_tree_probs,
            random_forest_probs, xgboost_probs, lstm_probs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                raw,
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
            class_labels=list(class_names.values())
        )

    except Exception as e:
        return render_template_string(html_template, predictions={"Error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
