# ✅ STEP 1: Load Dataset and Preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = r"D:\cyber security\major project\real time\Project\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path, engine='python')

# Drop non-useful columns
df.drop(columns=['Accident_Index', 'Accident Date', 'Time', 'Datetime'], inplace=True)

# Encode weather conditions
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

# Select features
features = ['Latitude', 'Longitude', 'Number_of_Casualties',
            'Number_of_Vehicles', 'Speed_limit', 'Weather_Conditions_Num']
X = df[features]
y = df['Accident_Severity_Num']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
import joblib
joblib.dump(scaler, 'final_scaler.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ✅ STEP 2A: Train Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
joblib.dump(log_model, 'logistic_model.pkl')
print("✅ Logistic Regression model trained and saved.")


# ✅ STEP 2B: Train Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'decision_tree_model.pkl')
print("✅ Decision Tree model trained and saved.")


# ✅ STEP 2C: Train XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(verbosity=0)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgboost_model.pkl')
print("✅ XGBoost model trained and saved.")
