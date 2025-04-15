# **🚦 Real-Time Traffic Accident Prediction Web App**

## **📌 Overview**
- Predicts accident severity using ML & DL models.
- Real-time predictions with input features and GPS coordinates.
- Built with Flask, SQLite, and Leaflet.js.
- Includes visual graphs, mail alert

---
🛠️ Tech Stack
Frontend: HTML, CSS, JS, Chart.js, Leaflet.js
Backend: Flask, Gunicorn
ML/DL: Scikit-learn, XGBoost, TensorFlow/Keras
Database: SQLite
Deployment: Render.com / Azure

📦 Future Upgrades
Add weather and road condition APIs.
SMS/Push notifications.
Admin dashboard for analytics.

## **📁 Project Structure**
- **`reaal_time_heat.py`** – Main Flask app.
- **`models/`** – Contains `.pkl` and `.h5` files for all trained models.
- **`final_scaler.pkl`** – Preprocessing scaler.
- **`predictions.db`** – SQLite database for logging.
- **`static/`** – JS, CSS, custom icons, audio.
- **`templates/`** – HTML files for dashboard.
- **`requirements.txt`** – Project dependencies.

---

## **✨ Key Features**
- 🔍 Multiple prediction models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - LSTM
- 📊 Model comparison with probability bar charts.
- 🗺️ Real-time accident location heatmap (Leaflet.js).
- 🔔 Sound alert for **Severe Accidents**.
- 📬 Email notification for emergency cases.
- 💾 Save predictions to SQLite database.
- 📥 Export predictions to CSV.

---
2. Create Virtual Environment
  python -m venv venv
  venv\Scripts\activate  # (Windows)
3. Install Dependencies
  pip install -r requirements.txt
4. Run the App
  python reaal_time_heat.py
Open browser at http://127.0.0.1:5000

📡 API Endpoint (Postman Support)

URL: POST /predict

Request JSON:

{
  "features": "12.0,22.1,34.5,...",
  "latitude": "12.9716",
  "longitude": "77.5946"
}

Response JSON:

{
  "predictions": {
    "Logistic Regression": "Minor Accident",
    "Decision Tree": "No Accident",
    ...
  },
  "probabilities": {
    "LSTM": [0.1, 0.2, 0.3, 0.4],
    ...
  }
}
 🚀 Deployment (Render / Azure / GCP / AWS)
   gunicorn reaal_time_heat:app
Make sure port is read from the environment:port = int(os.environ.get("PORT", 5000))

overwiew:
The Real-Time Traffic Accident Prediction Web App is an intelligent dashboard that leverages machine learning and deep learning to predict the severity of road accidents based on real-time inputs like traffic data, environmental conditions, and location coordinates.

🔍 Key Highlights:
✅ Uses five different models for prediction: Logistic Regression, Decision Tree, Random Forest, XGBoost, and LSTM.

🌐 Built using Flask, with a visually appealing frontend powered by Chart.js and Leaflet.js for interactive mapping.

🧠 Accepts input from forms or APIs and provides predictions with model-wise confidence scores.

📊 Displays real-time bar charts, pie charts, and an accident heatmap based on location.

🔔 Triggers audio alerts for severe accident predictions and logs all inputs/results to a local SQLite database.

💡 Ideal for smart city dashboards, traffic management systems, and emergency response services.

