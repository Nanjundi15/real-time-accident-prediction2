import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = r"D:\cyber security\major project\real time\Project\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path, engine='python')

# Drop unnecessary columns
df.drop(columns=['Accident_Index', 'Accident Date', 'Time', 'Datetime'], inplace=True)

# Map weather conditions to numeric
weather_mapping = {
    "Clear": 1, "Sunny": 1,
    "Partly Cloudy": 2, "Cloudy": 3,
    "Fog": 4, "Mist": 5,
    "Rainy": 6, "Snowy": 7, "Stormy": 8
}
df["Weather_Conditions_Num"] = df["Weather_Conditions"].map(weather_mapping).fillna(0)

# Encode target column
le = LabelEncoder()
df["Accident_Severity_Num"] = le.fit_transform(df["Accident_Severity"])

# Features and target
features = ['Latitude', 'Longitude', 'Number_of_Casualties',
            'Number_of_Vehicles', 'Speed_limit', 'Weather_Conditions_Num']
X = df[features]
y = df["Accident_Severity_Num"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "final_scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train smaller Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=30,          # reduced from 100
    max_depth=10,             # added depth limit
    min_samples_leaf=4,       # prevent deep trees
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save highly compressed model
joblib.dump(rf_model, "random_forest_model_compressed.pkl", compress=9)
print("✅ Compressed Random Forest model saved as 'random_forest_model_compressed.pkl'")
